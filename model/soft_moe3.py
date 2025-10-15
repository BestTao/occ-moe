# 做测试用的，对softmoe.py的softmoe5修改，主要对比原方法的jaax代码进行修改
import torch
from torch.nn import Module
import torch.nn.functional as F
import torch.distributed as dist
from torch import nn, einsum, Tensor
import itertools
from einops import rearrange, pack, unpack

from .soft_moe_distributed import (
    AllGather,
    split_by_rank,
    gather_sizes,
    has_only_one_value
)
import math

# helper functions

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def divisible_by(num, den):
    return (num % den) == 0

def chunk_num(num, chunks):
    num_per_chunk, remainder = divmod(num, chunks)

    out = []
    for i in range(chunks):
        n = num_per_chunk
        out.append(n + int(i < remainder))

    return out

def pack_one(t, pattern):
    return pack([t], pattern)

def unpack_one(t, ps, pattern):
    return unpack(t, ps, pattern)[0]

def l2norm(t):
    return F.normalize(t, dim = - 1)

def cumsum_exclusive(t, dim = -3):
    assert dim < 0
    num_pad_dims = -dim - 1
    pre_padding = (0, 0) * num_pad_dims
    return F.pad(t, (*pre_padding, 1, -1)).cumsum(dim = dim)

def log(t, eps = 1e-20):
    return torch.log(t.clamp(min = eps))

def gumbel_noise(t):
    noise = torch.zeros_like(t).uniform_(0, 1)
    return -log(-log(noise))

# norm

class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.register_buffer("beta", torch.zeros(dim))

    def forward(self, x):
        return F.layer_norm(x, x.shape[-1:], self.gamma, self.beta)

class RMSNorm(Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return l2norm(x) * self.scale * self.gamma

# expert

# def FeedForward(dim, num_experts, dropout=0.):
#     hidden_features = 4 * dim  # 强制 hidden_features = 4 * dim
#     dim_hidden = hidden_features // num_experts
#     assert dim_hidden * num_experts == hidden_features, "num_experts 必须能整除 4*dim"
#     return nn.Sequential(
#         nn.Linear(dim, dim_hidden),
#         nn.GELU(),
#         nn.Dropout(dropout),
#         nn.Linear(dim_hidden, dim)
#     )

def FeedForward(dim, num_experts, dropout=0.):
    hidden_dim_total = 4 * dim  # 与原MLP一致（3072当dim=768）
    hidden_dim_per_expert = hidden_dim_total // num_experts
    assert hidden_dim_total % num_experts == 0, "num_experts必须整除4*dim"
    
    return nn.Sequential(
        nn.Linear(dim, hidden_dim_per_expert),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim_per_expert, dim)
    )


class GEGLU(Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim = -1)
        return x * F.gelu(gate)

def GLUFeedForward(
    dim,
    mult = 4,
    dropout = 0.
):
    dim_hidden = int(dim * mult * 2 / 3)

    return nn.Sequential(
        nn.Linear(dim, dim_hidden * 2),
        GEGLU(),
        nn.Dropout(dropout),
        nn.Linear(dim_hidden, dim)
    )

def cosine_psim(
    x: torch.Tensor,
    batch_axes: tuple,
    contract_axes: tuple,
    eps: float = 1e-9) -> torch.Tensor:
    """PyTorch版余弦相似度矩阵计算（支持多维度收缩）"""
    # 1. 沿contract_axes归一化
    norm = x.pow(2).sum(dim=contract_axes, keepdim=True).add(eps).rsqrt()
    x_norm = x * norm
    
    # 2. 重组维度：[保留维度, contract_axes]
    all_dims = list(range(x_norm.dim()))
    preserved_dims = [d for d in all_dims if d not in contract_axes]
    x_perm = x_norm.permute(*(preserved_dims + list(contract_axes)))
    
    # 3. 展平保留的非batch维度
    batch_size = x_perm.shape[:len(batch_axes)]
    preserved_non_batch = x_perm.shape[len(batch_axes):-len(contract_axes)]
    flat_preserved = torch.prod(torch.tensor(preserved_non_batch)).item()
    
    # 4. 展平contract维度并计算点积
    x_flat = x_perm.reshape(*batch_size, flat_preserved, -1)
    dot_product = torch.matmul(x_flat, x_flat.transpose(-1, -2))
    
    return dot_product  # shape: (B, flat_preserved, flat_preserved)

class SoftMoE5(nn.Module):
    def __init__(self, dim, num_experts=3, capacity_factor=1.0, noise_std=0.0, occ_dim=64,
                 deterministic=False, compute_metrics=True, multiple_of=4,grad_clip_val=1.0,max_seq_len=256,
                #  expert_groups=(3,3,2)
                 expert_groups=(1,1,1)
                 ):
        super().__init__()
        self.dim = dim
        self.num_experts = num_experts
        self.capacity_factor = capacity_factor
        self.noise_std = noise_std
        self.deterministic = deterministic
        self.compute_metrics = compute_metrics
        self.multiple_of = multiple_of
        self.grad_clip_val = grad_clip_val
        # 新增参数预定义最大序列长度
        self.max_seq_len = max_seq_len
        self.multiple_of = 4
        # 输入归一化层
        self.norm = nn.LayerNorm(dim)
        self.expert_groups = expert_groups  # (可见专家数, 修复专家数, 推理专家数)
        # print(num_experts)
        # print(sum(expert_groups))
        assert sum(expert_groups) == num_experts, "专家分组数与总数不匹配"
        # 遮挡权重适配器
        self.occ_adapter = nn.Sequential(
            nn.Linear(1, occ_dim),
            nn.GELU(),
            nn.Linear(occ_dim, num_experts)
        )
        # 可见区域增强的Slot生成器
        # self.visible_slot_gen = nn.Linear(dim + occ_dim, dim)
        self.visible_slot_gen = nn.Linear(768 + 1, 768)
        # 遮挡修复Slot生成器
        # self.occ_slot_gen = nn.Linear(dim + occ_dim, dim)
        self.occ_slot_gen = nn.Linear(768 + 1, 768)
        # 可学习参数（对应论文中的mu和scale）
        fan_in = dim
        std = math.sqrt(1.0 / fan_in)
        max_slots = self._compute_max_slots(max_seq_len, num_experts)
        # self.mu = nn.Parameter(torch.empty(dim, num_experts, 1))  # 初始shape需动态调整
        # self.mu = nn.Parameter(torch.empty(dim, num_experts, max_slots))
        self.mu = nn.Parameter(torch.empty(num_experts, dim, max_slots))  # [8,768,32]
        # print(f"self.mu:{self.mu.shape}")
        nn.init.normal_(self.mu, mean=0, std=1/math.sqrt(dim))
        # nn.init.normal_(self.mu, mean=0, std=std)  # 手动实现lecun_normal
        self.scale = nn.Parameter(torch.ones(1))
        
        
        # 专家网络
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, 4*dim//self.num_experts),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(4*dim//self.num_experts, dim),
                nn.Dropout(0.1),)
            for _ in range(num_experts)])
        for expert in self.experts:
            nn.init.kaiming_normal_(expert[0].weight, nonlinearity='relu')
            nn.init.zeros_(expert[0].bias)
            nn.init.xavier_normal_(expert[3].weight)
            nn.init.zeros_(expert[3].bias)
        # 初始化参数
        # nn.init.lecun_normal_(self.mu)
        # nn.init.ones_(self.scale)

    def compute_capacity(self, num_tokens: int) -> int:
        """确保 num_experts * num_slots >= num_tokens"""
        num_slots = math.ceil(num_tokens / self.num_experts)
        num_slots = num_slots + (-num_slots) % self.multiple_of
        return max(num_slots, 1)
    
    def _compute_max_slots(self, max_seq_len, num_experts):
        """预计算最大可能的slot数量"""
        slots = math.ceil(max_seq_len / num_experts)
        return slots + (-slots % self.multiple_of)
    
    def get_metrics(
        self,
        combine_weights: torch.Tensor,
        dispatch_weights: torch.Tensor,
        mu: torch.Tensor) -> dict:
        
        B, N, E, S = combine_weights.shape
        # print(B, N, E, S)
        metrics = {
            'auxiliary_loss': torch.tensor(0.0),  # 论文中没有辅助损失
            'combine_weights_min_mean': combine_weights.min(dim=2).values.min(dim=2).values.mean(),
            'combine_weights_max_mean': combine_weights.max(dim=2).values.max(dim=2).values.mean(),
            'dispatch_weights_min_mean': dispatch_weights.min(dim=1).values.mean(),
            'dispatch_weights_max_mean': dispatch_weights.max(dim=1).values.mean(),
        }
        
        if self.compute_metrics:
            # Combine weights相似度 (B, N, N)
            cw_sim = cosine_psim(
                combine_weights,
                batch_axes=(0,),  # 按batch分组计算
                contract_axes=(2,3)  # 压缩专家和slot维度
            )
            assert cw_sim.shape == (B, N, N), f"Combine权重形状错误: {cw_sim.shape}"
            # 创建掩码并排除对角线
            eye = torch.eye(N, device=combine_weights.device)[None]  # (1, N, N)
            cw_sim_masked = (1 - eye) * cw_sim
            max_values = cw_sim_masked.amax(dim=(1,2), keepdim=True)  # (B,1,1)
            cw_sim_masked += eye * max_values
            
            metrics.update({
                'combine_weights_similarity_min': cw_sim_masked.min(),
                'combine_weights_similarity_max': cw_sim_masked.max(),
                'combine_weights_similarity_mean': (
                    cw_sim.sum() - B*N) / (B*N*(N-1))  # 排除对角线
            })
            
            # Dispatch weights相似度 (B, E, S, E, S)
            # dw_sim = cosine_psim(
            #     dispatch_weights,
            #     batch_axes=(0,),
            #     contract_axes=(1,)  # 压缩输入序列维度
            # ).view(B, E, S, E, S)
            dw_sim = cosine_psim(
                dispatch_weights.permute(0,2,3,1),  # [B,E,S,N] -> [B,N,E,S]
                batch_axes=(0,),
                contract_axes=(1,2)  # 压缩输入序列维度
            )            
            assert dw_sim.shape == (B, E, S, E, S), f"Dispatch权重形状错误: {dw_sim.shape}"
            # 创建5D eye mask
            eye = torch.eye(E*S, device=dispatch_weights.device).view(1, E, S, E, S)
            dw_sim_masked = (1 - eye) * dw_sim
            max_values = dw_sim_masked.amax(dim=(2,3,4), keepdim=True)
            dw_sim_masked += eye * max_values
            
            metrics.update({
                'dispatch_weights_similarity_min': dw_sim_masked.min(),
                'dispatch_weights_similarity_max': dw_sim_masked.max(),
                'dispatch_weights_similarity_mean': (
                    dw_sim.sum() - B*E*S) / (B*E*S*(E*S-1))
            })
            
            # Mu参数相似度 (E, S, E, S)
            mu_perm = mu.permute(1,2,0)
            mu_sim = cosine_psim(
                mu_perm,  
                batch_axes=(0,1), 
                contract_axes=(2,)  # 压缩特征维度
            )
            assert mu_sim.shape == (E, S, E, S), f"Mu相似度形状错误: {mu_sim.shape}"
            eye = torch.eye(E*S, device=mu.device).view(E, S, E, S)
            mu_sim_masked = (1 - eye) * mu_sim
            max_values = mu_sim_masked.amax(dim=(2,3), keepdim=True)
            mu_sim_masked += eye * max_values
            
            metrics.update({
                'mu_similarity_min': mu_sim_masked.min(),
                'mu_similarity_max': mu_sim_masked.max(),
                'mu_similarity_mean': (mu_sim.sum() - E*S) / (E*S*(E*S-1))
            })
            
        return metrics
    
    def _get_expert_mask(self, attn_weight):
        B, N = attn_weight.shape  # N=128
        device = attn_weight.device
        mask = torch.zeros(B, N, self.num_experts, device=device)
        
        # 划分遮挡等级
        vis_mask = (attn_weight > 0.7)        # [B,128]
        part_mask = (attn_weight >= 0.3) & (attn_weight <= 0.7)
        occ_mask = (attn_weight < 0.3)
        
        # 生成专家范围掩码
        expert_idx = torch.arange(self.num_experts, device=device)  # [E]
        
        # 可见专家分配 (前 expert_groups[0] 个)
        vis_expert_mask = (expert_idx < self.expert_groups[0])                      # [E]
        mask[vis_mask.unsqueeze(-1) & vis_expert_mask.view(1,1,-1)] = 1             # 三维布尔索引
        
        # 修复专家分配 (中间 expert_groups[1] 个)
        repair_start = self.expert_groups[0]
        repair_expert_mask = (expert_idx >= repair_start) & \
                            (expert_idx < repair_start + self.expert_groups[1])    # [E]
        mask[part_mask.unsqueeze(-1) & repair_expert_mask.view(1,1,-1)] = 1
        
        # 推理专家分配 (最后 expert_groups[2] 个)
        infer_start = repair_start + self.expert_groups[1]
        infer_expert_mask = (expert_idx >= infer_start)                            # [E]
        mask[occ_mask.unsqueeze(-1) & infer_expert_mask.view(1,1,-1)] = 1
        
        return mask
    def _process_tokens(self, img_tokens, attn_weight, mode):
        """ 处理图像块的核心路由逻辑 """
        B, N, D = img_tokens.shape  # N=128
        E, S = self.num_experts, self.compute_capacity(N)
        cls_feat = img_tokens[:, 0, :]  # [B, D] 从img_tokens获取CLS
        # === 输入归一化（仅处理图像块）===
        img_tokens_norm = self.norm(img_tokens)  # 关键修改：仅归一化图像块

        # 生成slot偏置输入
        if mode == 'original':
            attn_feat = attn_weight.mean(dim=1, keepdim=True)  # [B,1]
        else:
            attn_feat = 1 - attn_weight.mean(dim=1, keepdim=True)
            
        slot_input = torch.cat([cls_feat, attn_feat], dim=-1)  # [B, D+1]
        print(f"slot_input.shape:{slot_input.shape}") # torch.Size([64, 769])
        # 生成slot偏置
        if mode == 'original':
            slot_bias = self.visible_slot_gen(slot_input)  # [B, D]
        else:
            slot_bias = self.occ_slot_gen(slot_input)
        # print(f"slot_bias.shape:{slot_bias.shape}") # slot_bias.shape:torch.Size([64, 768])
        # 调整mu形状以匹配slot_bias
        # print(f"self.mu:{self.mu.shape}") # self.mu:torch.Size([8, 768, 32])
        mu = self.mu[:, :, :S]  # [experts=8, dim=768, slots=S]
        # print(f"mu.shape:{mu.shape}") # mu.shape:torch.Size([8, 768, 16])
        mu = mu.permute(1, 0, 2)  
        # print(f"mu.shape:{mu.shape}") # mu.shape:torch.Size([768, 8, 16]) # [dim=768, experts=8, slots=S]
        mu = mu.unsqueeze(0)   
        # print(f"mu.shape:{mu.shape}")  # [1, 768, 8, S]
        slot_bias = slot_bias.unsqueeze(1).unsqueeze(3)  # [B,768,1,1]
        slot_bias =slot_bias.permute(0,2,1,3)
        # print(f"slot_bias.shape:{slot_bias.shape}") # slot_bias.shape:torch.Size([64, 768, 1, 1])
        adjusted_mu = mu + slot_bias    # 广播后 [B,768,8,S]
        
        # === 分发权重计算 ===
        logits = torch.einsum('bnd,bdes->bnes', 
                            img_tokens_norm * self.scale, 
                            adjusted_mu)  # [B, 128, E, S]
        
        # 应用专家掩码
        print(attn_weight.shape) # torch.Size([64, 128])
        expert_mask = self._get_expert_mask(attn_weight)  # [B, 128, E]
        # print(f"expert_mask:{expert_mask.shape}")
        logits = logits * expert_mask.unsqueeze(-1)  # [B, 128, E, S]
        # print(f"logits:{logits.shape}")
        return self._moe_forward(img_tokens, logits)
    
    def _moe_forward(self, x, logits):
        """ 封装原有MoE计算流程 """
        B, N, D = x.shape
        E = self.num_experts
        S = self.compute_capacity(N)  # 动态计算slot数量
        if self.training and self.noise_std > 0:
            logits += torch.randn_like(logits) * self.noise_std
        def stable_softmax(logits, dim):
            logits = logits - logits.amax(dim=dim, keepdim=True).detach()
            return torch.softmax(logits, dim=dim)       
        # 保持原有softmax计算
        dispatch_weights = stable_softmax(logits, dim=1)
        combine_weights_flat = stable_softmax(logits.view(B, N, E*S), dim=-1)   
        # Slot构造与专家处理
        slot_inputs = einsum('bnd,bnes->besd', x, dispatch_weights)
        # print(f"slot_inputs:{slot_inputs.shape}") # slot_inputs:torch.Size([64, 4, 36, 768])
        # === 专家处理 ===
        slot_outputs = []
        for e in range(E):
            expert_input = slot_inputs[:, e]  # [B, S, D]
            expert_output = self.experts[e](expert_input)
            slot_outputs.append(expert_output)
        slot_outputs = torch.stack(slot_outputs, dim=1)  # [B, E, S, D]
        
        # === 输出聚合 ===
        slot_outputs_flat = rearrange(slot_outputs, 'b e s d -> b (e s) d')
        outputs = einsum('bsd,bns->bnd', slot_outputs_flat, combine_weights_flat)
        # outputs.register_hook(lambda grad: torch.clamp(grad, -self.grad_clip_val, self.grad_clip_val))
        if self.training:  # 仅在训练阶段注册梯度钩子
            outputs.register_hook(lambda grad: torch.clamp(grad, -self.grad_clip_val, self.grad_clip_val))        
        
        return outputs  # [B, N, D]
    def forward(self, x, attn_weight=None, mode='original'):
        B, N, D = x.shape
        print(f"输入到softmoe的：{x.shape}") # torch.Size([64, 129, 768])
        E = self.num_experts
        S = self.compute_capacity(N)  # 动态计算slot数量
        # print(S)
        # 更新mu参数形状
        # self.mu.data = self.mu.data.expand(D, E, S).contiguous()
        cls_token = x[:, 0:1, :]    # [B, 1, D]
        img_tokens = x[:, 1:, :]    # [B, 128, D]
        # === 动态Slot生成（关键改进1）===
        if attn_weight is not None:
            # 生成遮挡感知的slot偏置
            assert attn_weight.shape == (B, 128), "attn_weight必须匹配图像块数"
            occ_bias = self.occ_adapter(attn_weight.unsqueeze(-1))  # [B, N, E]
            # print(f"img_tokens.shape:{img_tokens.shape}")
            img_output = self._process_tokens(img_tokens, attn_weight, mode)
            # 分支特定的Slot增强
            if mode == 'original':
                cls_token = cls_token + img_output.mean(dim=1, keepdim=True)
                        # 合并特征
            outputs = torch.cat([cls_token, img_output], dim=1)
        else: 
            # 无遮挡处理模式s
            outputs, _ = self._moe_forward(x, None)
 
        metrics = {'auxiliary_loss': torch.tensor(0.0)}   
        return outputs, metrics
    



class SoftMoE6(nn.Module):
    def __init__(self, dim, num_experts=3, capacity_factor=1.0, noise_std=0.0, occ_dim=64,dropout_rate=0.1,
                 deterministic=False, compute_metrics=True, multiple_of=4,grad_clip_val=1.0,max_seq_len=256,
                #  expert_groups=(3,3,2)
                 expert_groups=(1,1,1)
                 ):
        super().__init__()
        self.dim = dim
        self.num_experts = num_experts
        self.capacity_factor = capacity_factor
        self.noise_std = noise_std
        self.deterministic = deterministic
        self.compute_metrics = compute_metrics
        self.multiple_of = multiple_of
        self.grad_clip_val = grad_clip_val
        # 新增参数预定义最大序列长度
        self.max_seq_len = max_seq_len
        self.multiple_of = 4
        # 输入归一化层
        self.norm = nn.LayerNorm(dim)
        self.expert_groups = expert_groups  # (可见专家数, 修复专家数, 推理专家数)
        # print(num_experts)
        # print(sum(expert_groups))
        assert sum(expert_groups) == num_experts, "专家分组数与总数不匹配"
        # 遮挡权重适配器
        self.occ_adapter = nn.Sequential(
            nn.Linear(1, occ_dim),
            nn.GELU(),
            nn.Linear(occ_dim, num_experts)
        )
        # 可见区域增强的Slot生成器
        # self.visible_slot_gen = nn.Linear(dim + occ_dim, dim)
        self.visible_slot_gen = nn.Linear(768 + 1, 768)
        # 遮挡修复Slot生成器
        # self.occ_slot_gen = nn.Linear(dim + occ_dim, dim)
        self.occ_slot_gen = nn.Linear(768 + 1, 768)
        # 可学习参数（对应论文中的mu和scale）
        fan_in = dim
        std = math.sqrt(1.0 / fan_in)
        max_slots = self._compute_max_slots(max_seq_len, num_experts)
        # self.mu = nn.Parameter(torch.empty(dim, num_experts, 1))  # 初始shape需动态调整
        # self.mu = nn.Parameter(torch.empty(dim, num_experts, max_slots))
        self.mu = nn.Parameter(torch.empty(num_experts, dim, max_slots))  # [8,768,32]
        # print(f"self.mu:{self.mu.shape}")
        nn.init.normal_(self.mu, mean=0, std=1/math.sqrt(dim))
        # nn.init.normal_(self.mu, mean=0, std=std)  # 手动实现lecun_normal
        self.scale = nn.Parameter(torch.ones(1))
        
        
        # 专家网络
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, 4*dim//self.num_experts),
                nn.GELU(),
                nn.Dropout(dropout_rate),
                nn.Linear(4*dim//self.num_experts, dim),
                nn.Dropout(dropout_rate),)
            for _ in range(num_experts)])
        
        self.cls_experts =  nn.Sequential(
                nn.Linear(dim, 4*dim),
                nn.GELU(),
                nn.Dropout(dropout_rate),
                nn.Linear(4*dim, dim),
                nn.Dropout(dropout_rate))
        for expert in self.experts:
            nn.init.kaiming_normal_(expert[0].weight, nonlinearity='relu')
            nn.init.zeros_(expert[0].bias)
            nn.init.xavier_normal_(expert[3].weight)
            nn.init.zeros_(expert[3].bias)
        nn.init.kaiming_normal_(self.cls_experts[0].weight, nonlinearity='relu')
        nn.init.zeros_(self.cls_experts[0].bias)    
        nn.init.kaiming_normal_(self.cls_experts[3].weight, nonlinearity='relu')
        nn.init.zeros_(self.cls_experts[3].bias)            
        # 初始化参数
        # nn.init.lecun_normal_(self.mu)
        # nn.init.ones_(self.scale)

    def compute_capacity(self, num_tokens: int) -> int:
        """确保 num_experts * num_slots >= num_tokens"""
        num_slots = math.ceil(num_tokens / self.num_experts)
        num_slots = num_slots + (-num_slots) % self.multiple_of
        return max(num_slots, 1)
    
    def _compute_max_slots(self, max_seq_len, num_experts):
        """预计算最大可能的slot数量"""
        slots = math.ceil(max_seq_len / num_experts)
        return slots + (-slots % self.multiple_of)
    
    def get_metrics(
        self,
        combine_weights: torch.Tensor,
        dispatch_weights: torch.Tensor,
        mu: torch.Tensor) -> dict:
        
        B, N, E, S = combine_weights.shape
        # print(B, N, E, S)
        metrics = {
            'auxiliary_loss': torch.tensor(0.0),  # 论文中没有辅助损失
            'combine_weights_min_mean': combine_weights.min(dim=2).values.min(dim=2).values.mean(),
            'combine_weights_max_mean': combine_weights.max(dim=2).values.max(dim=2).values.mean(),
            'dispatch_weights_min_mean': dispatch_weights.min(dim=1).values.mean(),
            'dispatch_weights_max_mean': dispatch_weights.max(dim=1).values.mean(),
        }
        
        if self.compute_metrics:
            # Combine weights相似度 (B, N, N)
            cw_sim = cosine_psim(
                combine_weights,
                batch_axes=(0,),  # 按batch分组计算
                contract_axes=(2,3)  # 压缩专家和slot维度
            )
            assert cw_sim.shape == (B, N, N), f"Combine权重形状错误: {cw_sim.shape}"
            # 创建掩码并排除对角线
            eye = torch.eye(N, device=combine_weights.device)[None]  # (1, N, N)
            cw_sim_masked = (1 - eye) * cw_sim
            max_values = cw_sim_masked.amax(dim=(1,2), keepdim=True)  # (B,1,1)
            cw_sim_masked += eye * max_values
            
            metrics.update({
                'combine_weights_similarity_min': cw_sim_masked.min(),
                'combine_weights_similarity_max': cw_sim_masked.max(),
                'combine_weights_similarity_mean': (
                    cw_sim.sum() - B*N) / (B*N*(N-1))  # 排除对角线
            })
            
            # Dispatch weights相似度 (B, E, S, E, S)
            # dw_sim = cosine_psim(
            #     dispatch_weights,
            #     batch_axes=(0,),
            #     contract_axes=(1,)  # 压缩输入序列维度
            # ).view(B, E, S, E, S)
            dw_sim = cosine_psim(
                dispatch_weights.permute(0,2,3,1),  # [B,E,S,N] -> [B,N,E,S]
                batch_axes=(0,),
                contract_axes=(1,2)  # 压缩输入序列维度
            )            
            assert dw_sim.shape == (B, E, S, E, S), f"Dispatch权重形状错误: {dw_sim.shape}"
            # 创建5D eye mask
            eye = torch.eye(E*S, device=dispatch_weights.device).view(1, E, S, E, S)
            dw_sim_masked = (1 - eye) * dw_sim
            max_values = dw_sim_masked.amax(dim=(2,3,4), keepdim=True)
            dw_sim_masked += eye * max_values
            
            metrics.update({
                'dispatch_weights_similarity_min': dw_sim_masked.min(),
                'dispatch_weights_similarity_max': dw_sim_masked.max(),
                'dispatch_weights_similarity_mean': (
                    dw_sim.sum() - B*E*S) / (B*E*S*(E*S-1))
            })
            
            # Mu参数相似度 (E, S, E, S)
            mu_perm = mu.permute(1,2,0)
            mu_sim = cosine_psim(
                mu_perm,  
                batch_axes=(0,1), 
                contract_axes=(2,)  # 压缩特征维度
            )
            assert mu_sim.shape == (E, S, E, S), f"Mu相似度形状错误: {mu_sim.shape}"
            eye = torch.eye(E*S, device=mu.device).view(E, S, E, S)
            mu_sim_masked = (1 - eye) * mu_sim
            max_values = mu_sim_masked.amax(dim=(2,3), keepdim=True)
            mu_sim_masked += eye * max_values
            
            metrics.update({
                'mu_similarity_min': mu_sim_masked.min(),
                'mu_similarity_max': mu_sim_masked.max(),
                'mu_similarity_mean': (mu_sim.sum() - E*S) / (E*S*(E*S-1))
            })
            
        return metrics
    
    def _get_expert_mask(self, attn_weight):
        B, N = attn_weight.shape  # N=128
        device = attn_weight.device
        mask = torch.zeros(B, N, self.num_experts, device=device)
        
        # 划分遮挡等级
        vis_mask = (attn_weight > 0.7)        # [B,128]
        part_mask = (attn_weight >= 0.3) & (attn_weight <= 0.7)
        occ_mask = (attn_weight < 0.3)
        
        # 生成专家范围掩码
        expert_idx = torch.arange(self.num_experts, device=device)  # [E]
        
        # 可见专家分配 (前 expert_groups[0] 个)
        vis_expert_mask = (expert_idx < self.expert_groups[0])                      # [E]
        mask[vis_mask.unsqueeze(-1) & vis_expert_mask.view(1,1,-1)] = 1             # 三维布尔索引
        
        # 修复专家分配 (中间 expert_groups[1] 个)
        repair_start = self.expert_groups[0]
        repair_expert_mask = (expert_idx >= repair_start) & \
                            (expert_idx < repair_start + self.expert_groups[1])    # [E]
        mask[part_mask.unsqueeze(-1) & repair_expert_mask.view(1,1,-1)] = 1
        
        # 推理专家分配 (最后 expert_groups[2] 个)
        infer_start = repair_start + self.expert_groups[1]
        infer_expert_mask = (expert_idx >= infer_start)                            # [E]
        mask[occ_mask.unsqueeze(-1) & infer_expert_mask.view(1,1,-1)] = 1
        
        return mask
    def  _process_tokens(self, x, attn_weight, mode):
        # cls_token = x[:, 0:1, :]    # [B, 1, D]
        img_tokens = x[:, 1:, :]    # [B, 128, D]
        """ 处理图像块的核心路由逻辑 """
        B, N, D = img_tokens.shape  # N=128
        E, S = self.num_experts, self.compute_capacity(N)
        cls_feat = x[:, 0, :]  # [B, D] 从img_tokens获取CLS
        # === 输入归一化（仅处理图像块）===
        img_tokens_norm = self.norm(img_tokens)  # 关键修改：仅归一化图像块

        # 生成slot偏置输入
        if mode == 'original':
            attn_feat = attn_weight.mean(dim=1, keepdim=True)  # [B,1]
        else:
            attn_feat = 1 - attn_weight.mean(dim=1, keepdim=True)
            
        slot_input = torch.cat([cls_feat, attn_feat], dim=-1)  # [B, D+1]
        # print(f"slot_input.shape:{slot_input.shape}")
        # 生成slot偏置
        if mode == 'original':
            slot_bias = self.visible_slot_gen(slot_input)  # [B, D]
        else:
            slot_bias = self.occ_slot_gen(slot_input)
        # print(f"slot_bias.shape:{slot_bias.shape}") # slot_bias.shape:torch.Size([64, 768])
        # 调整mu形状以匹配slot_bias
        # print(f"self.mu:{self.mu.shape}") # self.mu:torch.Size([8, 768, 32])
        mu = self.mu[:, :, :S]  # [experts=8, dim=768, slots=S]
        # print(f"mu.shape:{mu.shape}") # mu.shape:torch.Size([8, 768, 16])
        mu = mu.permute(1, 0, 2)  
        # print(f"mu.shape:{mu.shape}") # mu.shape:torch.Size([768, 8, 16]) # [dim=768, experts=8, slots=S]
        mu = mu.unsqueeze(0)   
        # print(f"mu.shape:{mu.shape}")  # [1, 768, 8, S]
        slot_bias = slot_bias.unsqueeze(1).unsqueeze(3)  # [B,768,1,1]
        slot_bias =slot_bias.permute(0,2,1,3)
        # print(f"slot_bias.shape:{slot_bias.shape}") # slot_bias.shape:torch.Size([64, 768, 1, 1])
        adjusted_mu = mu + slot_bias    # 广播后 [B,768,8,S]
        
        # === 分发权重计算 ===
        logits = torch.einsum('bnd,bdes->bnes', 
                            img_tokens_norm * self.scale, 
                            adjusted_mu)  # [B, 128, E, S]
        # print(f"logits1:{logits.shape}") # logits:torch.Size([64, 128, 8, 16])
        # 应用专家掩码
        print(attn_weight.shape) # torch.Size([64, 128])
        expert_mask = self._get_expert_mask(attn_weight)  # [B, 128, E]
        # print(f"expert_mask:{expert_mask.shape}")
        logits = logits * expert_mask.unsqueeze(-1)  # [B, 128, E, S]
        # print(f"logits:{logits.shape}") # logits:torch.Size([64, 128, 8, 16])
        # print(f"x.shape:{x.shape}") # torch.Size([64, 128])
        return self._moe_forward(img_tokens, logits)
    
    def _moe_forward(self, x, logits):
        """ 封装原有MoE计算流程 """
        B, N, D = x.shape
        # N=128
        # print(f"B:{B}")
        # print(f"N:{N}")
        E = self.num_experts
        S = self.compute_capacity(N)  # 动态计算slot数量
        # print(f"E:{E}")
        # print(f"S:{S}")
        if self.training and self.noise_std > 0:
            logits += torch.randn_like(logits) * self.noise_std
        def stable_softmax(logits, dim):
            logits = logits - logits.amax(dim=dim, keepdim=True).detach()
            return torch.softmax(logits, dim=dim)       
        # 保持原有softmax计算
        dispatch_weights = stable_softmax(logits, dim=1)
        # print(f"dispatch_weights:{dispatch_weights.shape}") # torch.Size([64, 128, 8, 16])
        # print(f"_moe_forward的logits:{logits.shape}") # torch.Size([64, 128, 8, 16])
        combine_weights_flat = stable_softmax(logits.view(B, N, E*S), dim=-1)   
        # Slot构造与专家处理
        slot_inputs = einsum('bnd,bnes->besd', x, dispatch_weights)
        # print(f"slot_inputs:{slot_inputs.shape}") # slot_inputs:torch.Size([64, 4, 36, 768])
        # === 专家处理 ===
        slot_outputs = []
        for e in range(E):
            expert_input = slot_inputs[:, e]  # [B, S, D]
            expert_output = self.experts[e](expert_input)
            # print(f"expert_output.shape:{expert_output.shape}") # torch.Size([64, 16, 768])
            slot_outputs.append(expert_output)
        slot_outputs = torch.stack(slot_outputs, dim=1)  # [B, E, S, D]
        # print(f"slot_outputs.shape:{slot_outputs.shape}") # ([64, 8, 16, 768])
        # === 输出聚合 ===
        slot_outputs_flat = rearrange(slot_outputs, 'b e s d -> b (e s) d')
        # print(f"slot_outputs_flat.shape:{slot_outputs_flat.shape}") # ([64, 128, 768])
        outputs = einsum('bsd,bns->bnd', slot_outputs_flat, combine_weights_flat)
        # print(f"outputs:{outputs.shape}") # outputs:torch.Size([64, 128, 768])
        # outputs.register_hook(lambda grad: torch.clamp(grad, -self.grad_clip_val, self.grad_clip_val))
        if self.training:  # 仅在训练阶段注册梯度钩子
            outputs.register_hook(lambda grad: torch.clamp(grad, -self.grad_clip_val, self.grad_clip_val))        
        
        return outputs  # [B, N, D]
    def forward(self, x, attn_weight=None, mode='original'):
        B, N, D = x.shape
        # print(f"输入到softmoe的：{x.shape}")
        E = self.num_experts
        S = self.compute_capacity(N)  # 动态计算slot数量
        cls_token =x[:, 0, :]
        # self.cls_experts
        # print(S)
        # 更新mu参数形状
        # self.mu.data = self.mu.data.expand(D, E, S).contiguous()
        # === 动态Slot生成（关键改进1）===
        if attn_weight is not None:
            # 生成遮挡感知的slot偏置
            assert attn_weight.shape == (B, 128), "attn_weight必须匹配图像块数"
            occ_bias = self.occ_adapter(attn_weight.unsqueeze(-1))  # [B, N, E]
            # print(f"img_tokens.shape:{img_tokens.shape}")
            img_output = self._process_tokens(x, attn_weight, mode)
            # 分支特定的Slot增强
            # if mode == 'original':
            #     cls_token = cls_token + img_output.mean(dim=1, keepdim=True)
                        # 合并特征
            # outputs = torch.cat([cls_token, img_output], dim=1)
            # cls_feat = self.cls_experts(cls_token)
            # outputs = torch.cat([cls_feat, img_output], dim=1)
            # outputs =img_output
        else: 
            # 无遮挡处理模式s
            # img_output, _ = self._moe_forward(x, None)
            img_output, _ = self._process_tokens(x,mode='occ')
        
        # cls_feat = self.cls_experts(cls_token)
        cls_token = cls_token.unsqueeze(1)
        # print(f"cls_feat:{cls_feat.shape}")
        # print(f"img_output:{img_output.shape}")
        all_feats = torch.cat([cls_token, img_output], dim=1)  
        outputs = self.cls_experts(all_feats)
 
        metrics = {'auxiliary_loss': torch.tensor(0.0)}   
        return outputs, metrics






# def test_image_input():
#     batch_size = 64
#     num_patches = 129
#     dim = 768
#     x = torch.randn(batch_size, num_patches, dim)
#     model5 = SoftMoE5(
#     dim=dim,
#     num_experts=3,
#     capacity_factor=1.0,
#     noise_std=0.1,
#     compute_metrics=True
# )
#     # print(model5)
#     output, metrics = model5(x)
#     for name, param in model5.named_parameters():
#         if "mu" in name:
#             print("存在")
#             print(f"{name}: {param.shape}")
#     print(model5)
#     print(output.shape)  # (64, 129, 768)
#     print(metrics.keys()) # 输出统计指标


# if __name__ == "__main__":
#     test_image_input()



# 在6的基础上修改，使得原图不需要输入Attention weight
class SoftMoE7(nn.Module):
    def __init__(self, dim, num_experts=3, capacity_factor=1.0, noise_std=0.0, occ_dim=64,dropout_rate=0.1,
                 deterministic=False, compute_metrics=True, multiple_of=4,grad_clip_val=1.0,max_seq_len=256,
                #  expert_groups=(3,3,2)
                 expert_groups=(1,1,1)
                 ):
        super().__init__()
        self.dim = dim
        self.num_experts = num_experts
        self.capacity_factor = capacity_factor
        self.noise_std = noise_std
        self.deterministic = deterministic
        self.compute_metrics = compute_metrics
        self.multiple_of = multiple_of
        self.grad_clip_val = grad_clip_val
        # 新增参数预定义最大序列长度
        self.max_seq_len = max_seq_len
        self.multiple_of = 4
        # 输入归一化层
        # self.norm = nn.LayerNorm(dim)
        self.expert_groups = expert_groups  # (可见专家数, 修复专家数, 推理专家数)
        # print(num_experts)
        # print(sum(expert_groups))
        assert sum(expert_groups) == num_experts, "专家分组数与总数不匹配"
        # 遮挡权重适配器
        self.occ_adapter = nn.Sequential(
            nn.Linear(1, occ_dim),
            nn.GELU(),
            nn.Linear(occ_dim, num_experts)
        )
        # 可见区域增强的Slot生成器
        # self.visible_slot_gen = nn.Linear(dim + occ_dim, dim)
        # self.visible_slot_gen = nn.Linear(768 + 1, 768)
        # 遮挡修复Slot生成器
        # self.occ_slot_gen = nn.Linear(dim + occ_dim, dim)
        self.occ_slot_gen = nn.Linear(768 + 1, 768)
        # 可学习参数（对应论文中的mu和scale）
        fan_in = dim
        std = math.sqrt(1.0 / fan_in)
        max_slots = self._compute_max_slots(max_seq_len, num_experts)
        # self.mu = nn.Parameter(torch.empty(dim, num_experts, 1))  # 初始shape需动态调整
        # self.mu = nn.Parameter(torch.empty(dim, num_experts, max_slots))
        self.mu = nn.Parameter(torch.empty(num_experts, dim, max_slots))  # [8,768,32]
        # print(f"self.mu:{self.mu.shape}")
        nn.init.normal_(self.mu, mean=0, std=1/math.sqrt(dim))
        # nn.init.normal_(self.mu, mean=0, std=std)  # 手动实现lecun_normal
        self.scale = nn.Parameter(torch.ones(1))
        
        
        # 专家网络
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, 4*dim//self.num_experts),
                nn.GELU(),
                nn.Dropout(dropout_rate),
                nn.Linear(4*dim//self.num_experts, dim),
                nn.Dropout(dropout_rate),)
            for _ in range(num_experts)])
        
        self.cls_experts =  nn.Sequential(
                nn.Linear(dim, 4*dim),
                nn.GELU(),
                nn.Dropout(dropout_rate),
                nn.Linear(4*dim, dim),
                nn.Dropout(dropout_rate))
        for expert in self.experts:
            nn.init.kaiming_normal_(expert[0].weight, nonlinearity='relu')
            nn.init.zeros_(expert[0].bias)
            nn.init.xavier_normal_(expert[3].weight)
            nn.init.zeros_(expert[3].bias)
        nn.init.kaiming_normal_(self.cls_experts[0].weight, nonlinearity='relu')
        nn.init.zeros_(self.cls_experts[0].bias)    
        nn.init.kaiming_normal_(self.cls_experts[3].weight, nonlinearity='relu')
        nn.init.zeros_(self.cls_experts[3].bias)            
        # 初始化参数
        # nn.init.lecun_normal_(self.mu)
        # nn.init.ones_(self.scale)

    def compute_capacity(self, num_tokens: int) -> int:
        """确保 num_experts * num_slots >= num_tokens"""
        num_slots = math.ceil(num_tokens / self.num_experts)
        num_slots = num_slots + (-num_slots) % self.multiple_of
        return max(num_slots, 1)
    
    def _compute_max_slots(self, max_seq_len, num_experts):
        """预计算最大可能的slot数量"""
        slots = math.ceil(max_seq_len / num_experts)
        return slots + (-slots % self.multiple_of)
    
    def get_metrics(
        self,
        combine_weights: torch.Tensor,
        dispatch_weights: torch.Tensor,
        mu: torch.Tensor) -> dict:
        
        B, N, E, S = combine_weights.shape
        # print(B, N, E, S)
        metrics = {
            'auxiliary_loss': torch.tensor(0.0),  # 论文中没有辅助损失
            'combine_weights_min_mean': combine_weights.min(dim=2).values.min(dim=2).values.mean(),
            'combine_weights_max_mean': combine_weights.max(dim=2).values.max(dim=2).values.mean(),
            'dispatch_weights_min_mean': dispatch_weights.min(dim=1).values.mean(),
            'dispatch_weights_max_mean': dispatch_weights.max(dim=1).values.mean(),
        }
        
        if self.compute_metrics:
            # Combine weights相似度 (B, N, N)
            cw_sim = cosine_psim(
                combine_weights,
                batch_axes=(0,),  # 按batch分组计算
                contract_axes=(2,3)  # 压缩专家和slot维度
            )
            assert cw_sim.shape == (B, N, N), f"Combine权重形状错误: {cw_sim.shape}"
            # 创建掩码并排除对角线
            eye = torch.eye(N, device=combine_weights.device)[None]  # (1, N, N)
            cw_sim_masked = (1 - eye) * cw_sim
            max_values = cw_sim_masked.amax(dim=(1,2), keepdim=True)  # (B,1,1)
            cw_sim_masked += eye * max_values
            
            metrics.update({
                'combine_weights_similarity_min': cw_sim_masked.min(),
                'combine_weights_similarity_max': cw_sim_masked.max(),
                'combine_weights_similarity_mean': (
                    cw_sim.sum() - B*N) / (B*N*(N-1))  # 排除对角线
            })
            
            # Dispatch weights相似度 (B, E, S, E, S)
            # dw_sim = cosine_psim(
            #     dispatch_weights,
            #     batch_axes=(0,),
            #     contract_axes=(1,)  # 压缩输入序列维度
            # ).view(B, E, S, E, S)
            dw_sim = cosine_psim(
                dispatch_weights.permute(0,2,3,1),  # [B,E,S,N] -> [B,N,E,S]
                batch_axes=(0,),
                contract_axes=(1,2)  # 压缩输入序列维度
            )            
            assert dw_sim.shape == (B, E, S, E, S), f"Dispatch权重形状错误: {dw_sim.shape}"
            # 创建5D eye mask
            eye = torch.eye(E*S, device=dispatch_weights.device).view(1, E, S, E, S)
            dw_sim_masked = (1 - eye) * dw_sim
            max_values = dw_sim_masked.amax(dim=(2,3,4), keepdim=True)
            dw_sim_masked += eye * max_values
            
            metrics.update({
                'dispatch_weights_similarity_min': dw_sim_masked.min(),
                'dispatch_weights_similarity_max': dw_sim_masked.max(),
                'dispatch_weights_similarity_mean': (
                    dw_sim.sum() - B*E*S) / (B*E*S*(E*S-1))
            })
            
            # Mu参数相似度 (E, S, E, S)
            mu_perm = mu.permute(1,2,0)
            mu_sim = cosine_psim(
                mu_perm,  
                batch_axes=(0,1), 
                contract_axes=(2,)  # 压缩特征维度
            )
            assert mu_sim.shape == (E, S, E, S), f"Mu相似度形状错误: {mu_sim.shape}"
            eye = torch.eye(E*S, device=mu.device).view(E, S, E, S)
            mu_sim_masked = (1 - eye) * mu_sim
            max_values = mu_sim_masked.amax(dim=(2,3), keepdim=True)
            mu_sim_masked += eye * max_values
            
            metrics.update({
                'mu_similarity_min': mu_sim_masked.min(),
                'mu_similarity_max': mu_sim_masked.max(),
                'mu_similarity_mean': (mu_sim.sum() - E*S) / (E*S*(E*S-1))
            })
            
        return metrics
    
    def _get_expert_mask(self, attn_weight):
        B, N = attn_weight.shape  # N=128
        device = attn_weight.device
        mask = torch.zeros(B, N, self.num_experts, device=device)
        
        # 划分遮挡等级
        vis_mask = (attn_weight > 0.7)        # [B,128]
        part_mask = (attn_weight >= 0.3) & (attn_weight <= 0.7)
        occ_mask = (attn_weight < 0.3)
        
        # 生成专家范围掩码
        expert_idx = torch.arange(self.num_experts, device=device)  # [E]
        
        # 可见专家分配 (前 expert_groups[0] 个)
        vis_expert_mask = (expert_idx < self.expert_groups[0])                      # [E]
        mask[vis_mask.unsqueeze(-1) & vis_expert_mask.view(1,1,-1)] = 1             # 三维布尔索引
        
        # 修复专家分配 (中间 expert_groups[1] 个)
        repair_start = self.expert_groups[0]
        repair_expert_mask = (expert_idx >= repair_start) & \
                            (expert_idx < repair_start + self.expert_groups[1])    # [E]
        mask[part_mask.unsqueeze(-1) & repair_expert_mask.view(1,1,-1)] = 1
        
        # 推理专家分配 (最后 expert_groups[2] 个)
        infer_start = repair_start + self.expert_groups[1]
        infer_expert_mask = (expert_idx >= infer_start)                            # [E]
        mask[occ_mask.unsqueeze(-1) & infer_expert_mask.view(1,1,-1)] = 1
        
        return mask
    def  _process_tokens(self, x,mode, attn_weight=None):
        # cls_token = x[:, 0:1, :]    # [B, 1, D]
        img_tokens = x[:, 1:, :]    # [B, 128, D]
        """ 处理图像块的核心路由逻辑 """
        B, N, D = img_tokens.shape  # N=128
        E, S = self.num_experts, self.compute_capacity(N)
        cls_feat = x[:, 0, :]  # [B, D] 从img_tokens获取CLS
        # === 输入归一化（仅处理图像块）===
        # img_tokens_norm = self.norm(img_tokens)  # 关键修改：仅归一化图像块
        img_tokens_norm = img_tokens
        # 生成slot偏置输入
        if mode == 'occlusion':
            attn_feat = 1 - attn_weight.mean(dim=1, keepdim=True)
            # attn_feat = attn_weight.mean(dim=1, keepdim=True)  # [B,1]
        # else:
        #     attn_feat = 1 - attn_weight.mean(dim=1, keepdim=True)
            
            slot_input = torch.cat([cls_feat, attn_feat], dim=-1)  # [B, D+1]
        else:
            slot_input = cls_feat
        mu = self.mu[:, :, :S]  # [experts=8, dim=768, slots=S]
        # print(f"mu.shape:{mu.shape}") # mu.shape:torch.Size([8, 768, 16])
        mu = mu.permute(1, 0, 2)  
        # print(f"mu.shape:{mu.shape}") # mu.shape:torch.Size([768, 8, 16]) # [dim=768, experts=8, slots=S]
        mu = mu.unsqueeze(0)   
        # print(f"slot_input.shape:{slot_input.shape}")
        # 生成slot偏置
        if mode == 'occlusion':
              # [B, D]
            slot_bias = self.occ_slot_gen(slot_input)
            slot_bias = slot_bias.unsqueeze(1).unsqueeze(3)  # [B,768,1,1]
            slot_bias =slot_bias.permute(0,2,1,3)
            adjusted_mu = mu + slot_bias
        else:
            adjusted_mu = mu
        # else:
            # slot_bias = self.visible_slot_gen(slot_input)

        # print(f"slot_bias.shape:{slot_bias.shape}") # slot_bias.shape:torch.Size([64, 768])
        # 调整mu形状以匹配slot_bias
        # print(f"self.mu:{self.mu.shape}") # self.mu:torch.Size([8, 768, 32])

        # print(f"mu.shape:{mu.shape}")  # [1, 768, 8, S]
        # print(f"slot_bias.shape:{slot_bias.shape}") # slot_bias.shape:torch.Size([64, 768, 1, 1])

        
        # === 分发权重计算 ===
        logits = torch.einsum('bnd,bdes->bnes', 
                            img_tokens_norm * self.scale, 
                            adjusted_mu)  # [B, 128, E, S]
        # print(f"logits1:{logits.shape}") # logits:torch.Size([64, 128, 8, 16])
        # 应用专家掩码
        # print(attn_weight.shape) # torch.Size([64, 128])
        if mode == 'occlusion':
            expert_mask = self._get_expert_mask(attn_weight)  # [B, 128, E]
            # print(f"expert_mask:{expert_mask.shape}")
            logits = logits * expert_mask.unsqueeze(-1)  # [B, 128, E, S]
        # print(f"logits:{logits.shape}") # logits:torch.Size([64, 128, 8, 16])
        # print(f"x.shape:{x.shape}") # torch.Size([64, 128])
        return self._moe_forward(img_tokens, logits)
    
    def _moe_forward(self, x, logits):
        """ 封装原有MoE计算流程 """
        B, N, D = x.shape
        # N=128
        # print(f"B:{B}")
        # print(f"N:{N}")
        E = self.num_experts
        S = self.compute_capacity(N)  # 动态计算slot数量
        # print(f"E:{E}")
        # print(f"S:{S}")
        if self.training and self.noise_std > 0:
            logits += torch.randn_like(logits) * self.noise_std
        def stable_softmax(logits, dim):
            logits = logits - logits.amax(dim=dim, keepdim=True).detach()
            return torch.softmax(logits, dim=dim)       
        # 保持原有softmax计算
        dispatch_weights = stable_softmax(logits, dim=1)
        # print(f"dispatch_weights:{dispatch_weights.shape}") # torch.Size([64, 128, 8, 16])
        # print(f"_moe_forward的logits:{logits.shape}") # torch.Size([64, 128, 8, 16])
        combine_weights_flat = stable_softmax(logits.view(B, N, E*S), dim=-1)   
        # Slot构造与专家处理
        slot_inputs = einsum('bnd,bnes->besd', x, dispatch_weights)
        # print(f"slot_inputs:{slot_inputs.shape}") # slot_inputs:torch.Size([64, 4, 36, 768])
        # === 专家处理 ===
        slot_outputs = []
        for e in range(E):
            expert_input = slot_inputs[:, e]  # [B, S, D]
            expert_output = self.experts[e](expert_input)
            # print(f"expert_output.shape:{expert_output.shape}") # torch.Size([64, 16, 768])
            slot_outputs.append(expert_output)
        slot_outputs = torch.stack(slot_outputs, dim=1)  # [B, E, S, D]
        # print(f"slot_outputs.shape:{slot_outputs.shape}") # ([64, 8, 16, 768])
        # === 输出聚合 ===
        slot_outputs_flat = rearrange(slot_outputs, 'b e s d -> b (e s) d')
        # print(f"slot_outputs_flat.shape:{slot_outputs_flat.shape}") # ([64, 128, 768])
        outputs = einsum('bsd,bns->bnd', slot_outputs_flat, combine_weights_flat)
        # print(f"outputs:{outputs.shape}") # outputs:torch.Size([64, 128, 768])
        # outputs.register_hook(lambda grad: torch.clamp(grad, -self.grad_clip_val, self.grad_clip_val))
        if self.training:  # 仅在训练阶段注册梯度钩子
            outputs.register_hook(lambda grad: torch.clamp(grad, -self.grad_clip_val, self.grad_clip_val))        
        
        return outputs  # [B, N, D]
    def forward(self, x, attn_weight=None, mode='original'):
        B, N, D = x.shape
        # print(f"输入到softmoe的：{x.shape}")
        E = self.num_experts
        S = self.compute_capacity(N)  # 动态计算slot数量
        cls_token =x[:, 0, :]
        # self.cls_experts
        # print(S)
        # 更新mu参数形状
        # self.mu.data = self.mu.data.expand(D, E, S).contiguous()
        # === 动态Slot生成（关键改进1）===
        if mode == 'occlusion':
            # 生成遮挡感知的slot偏置
            # print(f"当前的attn_weight.shape：{attn_weight.shape}")
            # assert attn_weight.shape == (B, 128), "attn_weight必须匹配图像块数"
            # occ_bias = self.occ_adapter(attn_weight.unsqueeze(-1))  # [B, N, E]
            # print(f"img_tokens.shape:{img_tokens.shape}")
            img_output = self._process_tokens(x, mode,attn_weight=attn_weight)
            # 分支特定的Slot增强
            # if mode == 'original':
            #     cls_token = cls_token + img_output.mean(dim=1, keepdim=True)
                        # 合并特征
            # outputs = torch.cat([cls_token, img_output], dim=1)
            # cls_feat = self.cls_experts(cls_token)
            # outputs = torch.cat([cls_feat, img_output], dim=1)
            # outputs =img_output
        else: 
            # 无遮挡处理模式s
            # img_output, _ = self._moe_forward(x, None)
            img_output= self._process_tokens(x,mode=mode,attn_weight=None)
        
        # cls_feat = self.cls_experts(cls_token)
        cls_token = cls_token.unsqueeze(1)
        # print(f"cls_feat:{cls_feat.shape}")
        # print(f"img_output:{img_output.shape}")
        # all_feats = torch.cat([cls_token, img_output], dim=1)  
        outputs = self.cls_experts(torch.cat([cls_token, img_output], dim=1))
 
        metrics = {'auxiliary_loss': torch.tensor(0.0)}   
        return outputs, metrics
    


# def test_softmoe():
#     B, N, D = 64, 129, 768
#     attn_weight = torch.rand(B, 128)
    
#     model = SoftMoE7(dim=D, num_experts=3, expert_groups=(1,1,1))
#     # model = SoftMoE7(dim=D, num_experts=8, expert_groups=(3,3,2))
#     print(model)
#     # 原图分支前向
#     orig_output, _ = model(torch.randn(B, N, D),  mode='original')
#     assert orig_output.shape == (B, N, D), f"原图分支输出形状错误: {orig_output.shape}"
#     print(f"orig_output:{orig_output.shape}")  # orig_output:torch.Size([64, 129, 768])
#     # 遮挡分支前向
#     occ_output, _ = model(torch.randn(B, N, D), attn_weight, mode='occlusion')
#     assert occ_output.shape == (B, N, D), f"遮挡分支输出形状错误: {occ_output.shape}"
#     print(f"occ_output:{occ_output.shape}")  # occ_output:torch.Size([64, 129, 768]
#     # 检查CLS融合
#     cls_diff = (occ_output[:,0] - orig_output[:,0]).abs().mean()
#     print(f"CLS差异均值: {cls_diff.item():.4f}")  # 应有明显差异

# test_softmoe()
