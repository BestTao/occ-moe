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

# def cosine_psim(
#     x: torch.Tensor,
#     batch_axes: tuple,
#     contract_axes: tuple,
#     eps: float = 1e-9) -> torch.Tensor:
#     """PyTorch版余弦相似度矩阵计算（支持多维度收缩）"""
#     # 1. 沿contract_axes归一化
#     norm = x.pow(2).sum(dim=contract_axes, keepdim=True).add(eps).rsqrt()
#     x_norm = x * norm
    
#     # 2. 重组维度：[保留维度, contract_axes]
#     all_dims = list(range(x_norm.dim()))
#     preserved_dims = [d for d in all_dims if d not in contract_axes]
#     x_perm = x_norm.permute(*(preserved_dims + list(contract_axes)))
    
#     # 3. 展平保留的非batch维度
#     batch_size = x_perm.shape[:len(batch_axes)]
#     preserved_non_batch = x_perm.shape[len(batch_axes):-len(contract_axes)]
#     flat_preserved = torch.prod(torch.tensor(preserved_non_batch)).item()
    
#     # 4. 展平contract维度并计算点积
#     x_flat = x_perm.reshape(*batch_size, flat_preserved, -1)
#     dot_product = torch.matmul(x_flat, x_flat.transpose(-1, -2))
    
#     return dot_product  # shape: (B, flat_preserved, flat_preserved)

class SoftMoE5(nn.Module):
    def __init__(self, dim, num_experts, capacity_factor=1.0, noise_std=0.0, dropout_rate=0.1,
                 deterministic=False, compute_metrics=True, multiple_of=4,grad_clip_val=1.0,max_seq_len=256):
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
        
        # 可学习参数（对应论文中的mu和scale）
        fan_in = dim
        std = math.sqrt(1.0 / fan_in)
        max_slots = self._compute_max_slots(max_seq_len, num_experts)
        # self.mu = nn.Parameter(torch.empty(dim, num_experts, 1))  # 初始shape需动态调整
        self.mu = nn.Parameter(torch.empty(dim, num_experts, max_slots))
        print(f"self.mu:{self.mu.shape}")
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
        for expert in self.experts:
            nn.init.kaiming_normal_(expert[0].weight, nonlinearity='relu')
            # nn.init.orthogonal_(expert[0].weight)
            # expert[0].weight.data.mul_(math.sqrt(2 / (1 + math.sqrt(2))))  # GELU增益
                # 第二层：零初始化避免初始阶段输出偏移
            # nn.init.zeros_(expert[3].weight)
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

    # def compute_capacity(self, num_tokens: int) -> int:
    #     num_slots = (num_tokens + self.num_experts - 1) // self.num_experts
    #     remainder = num_slots % self.multiple_of
    #     return num_slots + (self.multiple_of - remainder) % self.multiple_of


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
    def forward(self, x, mask=None):
        B, N, D = x.shape
        print(f"输入到softmoe的：{x.shape}")
        E = self.num_experts
        S = self.compute_capacity(N)  # 动态计算slot数量
        # print(S)
        # 更新mu参数形状
        # self.mu.data = self.mu.data.expand(D, E, S).contiguous()
        mu = self.mu[:, :, :S]  # [D, E, S]  # [1, D, E, 1]
        # print(f"mu:{mu.shape}") # mu:torch.Size([768, 4, 36])
        # mu = mu.unsqueeze(0).expand(B, -1, -1, -1)  # [B, D, E, S]
        
        # === 输入归一化 ===
        x = self.norm(x)
        # logits = torch.einsum('bnd,des->bnes', x * self.scale, self.mu[:, :, :S])
        x_norm = torch.nn.functional.normalize(x, p=2, dim=-1)
        mu_norm = torch.nn.functional.normalize(mu, dim=0)
        
        # print(f"x_norm:{x_norm.shape}") # x_norm:torch.Size([64, 129, 768])
        # print(f"mu_norm:{mu_norm.shape}") # mu_norm:torch.Size([768, 4, 36])
        # === 分发权重计算 ===
        # 计算logits [B, N, E, S]
        # logits = einsum('bnd,bdes->bnes', x_norm * self.scale, mu_norm)
        logits = torch.einsum('bnd,des->bnes', x_norm* self.scale, mu_norm)
        if x.numel() < mu.numel():
            x_norm = x_norm * self.scale
        else:
            mu_norm = mu_norm * self.scale

        # logits = einsum('bnd,denp->bnes', x_norm, mu_norm)              
        # print(f"logits:{logits.shape}")
        # 添加噪声
        if self.training and self.noise_std > 0:
            logits += torch.randn_like(logits) * self.noise_std
        def stable_softmax(logits, dim):
            logits = logits - logits.amax(dim=dim, keepdim=True).detach()
            return torch.softmax(logits, dim=dim)
        # Softmax维度调整
        dispatch_weights = stable_softmax(logits, dim=1)
        # print(f"logits.shape:{logits.shape}") # logits.shape:torch.Size([64, 129, 4, 36])
        # print(f"B, N, E, S:{B, N, E, S}")
        combine_weights_flat = stable_softmax(logits.view(B, N, E*S), dim=-1)
        # combine_weights = combine_weights_flat.view(B, N, E, S)  # [B, N, E, S]
        # print(combine_weights.shape)
        # === Slot输入构造 ===
        slot_inputs = einsum('bnd,bnes->besd', x, dispatch_weights)
        # print(f"slot_inputs:{slot_inputs.shape}") # slot_inputs:torch.Size([64, 4, 36, 768])

        # === 专家处理 ===
        slot_outputs = []
        for e in range(E):
            expert_input = slot_inputs[:, e]  # [B, S, D]
            expert_output = self.experts[e](expert_input)
            slot_outputs.append(expert_output)
        slot_outputs = torch.stack(slot_outputs, dim=1)  # [B, E, S, D]
        
        # 前向计算
        # slot_inputs = rearrange(slot_inputs, 'b e s d -> (b e s) d')
        # slot_outputs = torch.vmap(lambda expert, x: expert(x))(self.experts, slot_inputs)
        # slot_outputs = rearrange(slot_outputs, '(b e s) d -> b e s d', b=B, e=E)
        # === 输出聚合 ===
        slot_outputs_flat = rearrange(slot_outputs, 'b e s d -> b (e s) d')
        outputs = einsum('bsd,bns->bnd', slot_outputs_flat, combine_weights_flat)
        # outputs.register_hook(lambda grad: torch.clamp(grad, -self.grad_clip_val, self.grad_clip_val))
        if self.training:  # 仅在训练阶段注册梯度钩子
            outputs.register_hook(lambda grad: torch.clamp(grad, -self.grad_clip_val, self.grad_clip_val))
        # 指标收集
        # if self.compute_metrics:
        #     metrics = self.get_metrics(combine_weights, dispatch_weights, self.mu)
        # else:
        #     metrics = {}        
        metrics = {'auxiliary_loss': torch.tensor(0.0)}   
        return outputs, metrics
    
def cosine_psim(x, contract_axes=(2, 3)):
    # 归一化输入张量（沿特征维度）
    x_normalized = F.normalize(x, p=2, dim=contract_axes)
    
    # 将特征维度展平（例如将 dim1 和 dim2 合并）
    batch_size = x.shape[0]
    # 计算特征维度总大小（contract_axes 各维度乘积）
    feature_size = 1
    for dim in contract_axes:
        feature_size *= x_normalized.size(dim)
    
    # 将所有非批次和非特征维度合并到第二维
    permute_dims = [0] + list(contract_axes) + \
                  [d for d in range(1, x.dim()) if d not in contract_axes]
    x_flat = x_normalized.permute(*permute_dims).contiguous()
    x_flat = x_flat.view(batch_size, feature_size, -1)  # (batch, feature_size, other_dims)
    
    # Step 3: 计算余弦相似度矩阵
    similarity = torch.bmm(x_flat, x_flat.transpose(1, 2))  # (batch, feature_size, feature_size)
    return similarity


# def cosine_psim(
#     x: torch.Tensor,
#     batch_axes: tuple = (0,),
#     contract_axes: tuple = (2, 3),
#     eps: float = 1e-9
# ) -> torch.Tensor:
#     """PyTorch 版余弦相似度矩阵计算，完全对齐 JAX 逻辑"""
#     # --------------------------
#     # Step 1: 沿 contract_axes 归一化
#     # --------------------------
#     # 计算 L2 范数（对齐 JAX 的 rsqart(x^2.sum + eps)）
#     norm = torch.sqrt(x.pow(2).sum(dim=contract_axes, keepdim=True) + eps)
#     x_normalized = x / norm
    
#     # --------------------------
#     # Step 2: 重组张量维度，准备矩阵乘法
#     # --------------------------
#     # 确定所有维度
#     all_dims = list(range(x.dim()))
#     batch_dims = list(batch_axes)
#     contract_dims = list(contract_axes)
    
#     # 非批次和非收缩维度（将被展平）
#     other_dims = [
#         d for d in all_dims
#         if d not in batch_dims + contract_dims
#     ]
    
#     # 重组维度顺序：[batch_dims, contract_dims, other_dims]
#     permute_dims = batch_dims + contract_dims + other_dims
#     x_permuted = x_normalized.permute(*permute_dims)
    
#     # 展平非关键维度
#     batch_shape = x_permuted.size()[:len(batch_dims)]
#     contract_size = torch.prod(torch.tensor(x_permuted.size()[len(batch_dims):len(batch_dims)+len(contract_dims)]))
#     other_size = torch.prod(torch.tensor(x_permuted.size()[len(batch_dims)+len(contract_dims):]))
    
#     x_flat = x_permuted.reshape(*batch_shape, contract_size, other_size)
    
#     # --------------------------
#     # Step 3: 执行矩阵乘法（对齐 JAX 的 dot_general）
#     # --------------------------
#     # Einstein 求和约定：对齐 JAX 的维度收缩逻辑
#     # JAX: dot_general((contract, batch), (contract, batch)) -> (batch, batch, other, other)
#     # PyTorch 等效实现：
#     cw_sim = torch.einsum('...ik,...jk->...ij', x_flat, x_flat)
    
#     return cw_sim



# 25.4.13，针对说上面的softmoe5出现后续训练的不稳定问题，重新使用jax代码修正
# 修改后的损失计算
class SoftMoE7(nn.Module):
    def __init__(self, dim, num_experts, capacity_factor=1.0, noise_std=0.0, dropout_rate=0.1,
                 deterministic=False, compute_metrics=True, multiple_of=4,grad_clip_val=1.0,max_seq_len=256,
                 compute_similarity_metrics=True, precision=torch.float32):
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
        self.compute_similarity_metrics = True
        self.precision = precision
        # 输入归一化层
        # self.norm = nn.LayerNorm(dim)
        
        # 可学习参数（对应论文中的mu和scale）
        fan_in = dim
        std = math.sqrt(1.0 / fan_in)
        max_slots = self._compute_max_slots(max_seq_len, num_experts)
        # self.mu = nn.Parameter(torch.empty(dim, num_experts, 1))  # 初始shape需动态调整
        self.mu = nn.Parameter(torch.empty(dim, num_experts, max_slots))
        # print(f"self.mu:{self.mu.shape}")
        # nn.init.normal_(self.mu, mean=0, std=1/math.sqrt(dim))
        nn.init.kaiming_normal_(self.mu, mode='fan_in', nonlinearity='linear')
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
        for expert in self.experts:
            nn.init.kaiming_normal_(expert[0].weight, nonlinearity='relu')
            # nn.init.orthogonal_(expert[0].weight)
            # expert[0].weight.data.mul_(math.sqrt(2 / (1 + math.sqrt(2))))  # GELU增益
                # 第二层：零初始化避免初始阶段输出偏移
            # nn.init.zeros_(expert[3].weight)
            nn.init.zeros_(expert[0].bias)
            nn.init.xavier_normal_(expert[3].weight)
            nn.init.zeros_(expert[3].bias)
        # 初始化参数
        # nn.init.lecun_normal_(self.mu)
        # nn.init.ones_(self.scale)

    # def compute_capacity(self, num_tokens: int) -> int:
    #     """确保 num_experts * num_slots >= num_tokens"""
    #     num_slots = math.ceil(num_tokens / self.num_experts)
    #     num_slots = num_slots + (-num_slots) % self.multiple_of
    #     return max(num_slots, 1)

    # def compute_capacity(self, num_tokens: int) -> int:
    #     num_slots = (num_tokens + self.num_experts - 1) // self.num_experts
    #     remainder = num_slots % self.multiple_of
    #     return num_slots + (self.multiple_of - remainder) % self.multiple_of
    def compute_capacity(
        self,
        num_tokens: int,
        ceil_or_round: str = "ceil"
    ) -> int:
        """修正后的容量计算"""
        # 确保必要的参数存在
        # assert hasattr(self, 'capacity_factor'), "需要定义capacity_factor"
        # assert hasattr(self, 'multiple_of'), "需要定义multiple_of"

        # 应用capacity_factor
        scaled_tokens = num_tokens * self.capacity_factor
        
        # 计算基础容量
        if ceil_or_round == "ceil":
            capacity = math.ceil(scaled_tokens / self.num_experts)
        elif ceil_or_round == "round":
            capacity = round(scaled_tokens / self.num_experts)
        else:
            raise ValueError(f"Invalid ceil_or_round: {ceil_or_round}")

        # 确保最小容量
        capacity = max(capacity, 1)

        # 对齐到multiple_of
        if self.multiple_of > 1:
            remainder = capacity % self.multiple_of
            if remainder != 0:
                capacity += self.multiple_of - remainder

        # 输出警告（可选）
        actual_factor = capacity * self.num_experts / num_tokens
        if abs(actual_factor - self.capacity_factor) > 1e-6:
            print(f"Warning: Target cap_factor {self.capacity_factor}, Actual {actual_factor:.2f}")

        return capacity
    def cosine_psim(self, x, contract_dims, batch_dims, eps=1e-9):
        """PyTorch版本的cosine_psim函数"""
        # 沿contract_dims归一化
        norm = torch.rsqrt((x**2).sum(dim=contract_dims, keepdim=True) + eps)
        x_norm = x * norm
        
        # 重新排列维度以进行批处理矩阵乘法
        all_dims = list(range(x_norm.ndim))
        keep_dims = [d for d in all_dims if d not in contract_dims]
        # 之前的
        # x_flat = x_norm.permute(*keep_dims, *contract_dims).flatten(start_dim=len(keep_dims))
        # 修改的
        x_flat = x_norm.reshape(*x_norm.shape[:len(keep_dims)], -1)
        # 计算批处理点积
        similarity = torch.einsum('...i,...j->...ij', x_flat, x_flat)
        return similarity
    def _compute_max_slots(self, max_seq_len, num_experts):
        """预计算最大可能的slot数量"""
        slots = math.ceil(max_seq_len / num_experts)
        return slots + (-slots % self.multiple_of)
    def get_metrics(self, combine_weights, dispatch_weights, mu):
        B, N, E, S = combine_weights.shape
        device = combine_weights.device
        metrics = {}

        if self.compute_similarity_metrics:
            # ===== Combine Weights相似度 =====
            with torch.autocast(device_type='cuda', dtype=self.precision):  # 混合精度支持
                # 计算相似度矩阵 [B, N, N]
                cw_sim = self.cosine_psim(
                    combine_weights,
                    contract_dims=(2, 3),  # 对应E,S维度
                    batch_dims=(0,)        # B维度作为批处理
                )
                
                # 创建对角线掩码
                # eye = torch.eye(N, device=device).expand(B, -1, -1)
                # 创建对角线掩码并替换对角线元素为最大值
                # eye = torch.eye(E*S, device=device).expand(B, -1, -1).bool()
                # max_values = dw_sim.max(dim=2, keepdim=True)[0].max(dim=1, keepdim=True)[0]  # [B, 1, 1]
                # dw_sim_masked = torch.where(eye, max_values, dw_sim)    
                # sum_total = dw_sim_masked.sum()            
                # 计算均值（排除对角线）
                sum_total = cw_sim.sum()
                sum_diag = torch.diagonal(cw_sim, dim1=1, dim2=2).sum()
                metrics['combine_weights_similarity_mean'] = (sum_total - sum_diag) / (B * N * (N - 1))

            # ===== Dispatch Weights相似度 =====
            with torch.autocast(device_type='cuda', dtype=self.precision):
                # 重新排列维度为[B, E, S, N]
                dw_permuted = dispatch_weights.permute(0, 2, 3, 1)
                
                # 计算相似度矩阵 [B, E*S, E*S]
                dw_sim = self.cosine_psim(
                    dw_permuted,
                    contract_dims=(3,),    # N维度
                    batch_dims=(0,)        # B维度
                )
                
                # 创建对角线掩码
                eye = torch.eye(E*S, device=device).expand(B, -1, -1)
                
                # 计算均值（排除对角线）
                sum_total = dw_sim.sum()
                sum_diag = torch.diagonal(dw_sim, dim1=1, dim2=2).sum()
                metrics['dispatch_weights_similarity_mean'] = (sum_total - sum_diag) / (B * E * S * (E*S - 1))

        return metrics
    # ######33##############################
    # def get_metrics(self, combine_weights, dispatch_weights, mu):
    #     B, N, E, S = combine_weights.shape
    #     device = combine_weights.device
    #     metrics = {
    #         'auxiliary_loss': torch.tensor(0.0, device=device),
    #         # 'combine_weights_min_mean': combine_weights.min(dim=3)[0].min(dim=2)[0].mean(),
    #         # 'combine_weights_max_mean': combine_weights.max(dim=3)[0].max(dim=2)[0].mean(),
    #         # 'dispatch_weights_min_mean': dispatch_weights.min(dim=1)[0].mean(),
    #         # 'dispatch_weights_max_mean': dispatch_weights.max(dim=1)[0].mean(),
    #     }

    #     if self.compute_similarity_metrics:
    #         # Combine weights similarity
    #         combine_flat = combine_weights.view(B, N, -1)
    #         similarity = torch.cosine_similarity(
    #             combine_flat.unsqueeze(2), 
    #             combine_flat.unsqueeze(1),
    #             dim=3
    #         )
    #         eye = torch.eye(N, device=device).expand(B, -1, -1)
    #         masked_sim = similarity * (1 - eye) + eye * similarity.amax(dim=(1,2), keepdim=True)
    #         print("combine_weights_similarity_mean:")
    #         print((similarity.sum() - B*N) / (B*N*(N-1)))
    #         metrics.update({
    #             # 'combine_weights_similarity_min': masked_sim.min(),
    #             # 'combine_weights_similarity_max': masked_sim.max(),
    #             'combine_weights_similarity_mean': (similarity.sum() - B*N) / (B*N*(N-1))
    #         })

    #         # Dispatch weights similarity
    #         dispatch_flat = dispatch_weights.permute(0,2,3,1).reshape(B, -1, N)
    #         similarity = torch.cosine_similarity(
    #             dispatch_flat.unsqueeze(2),
    #             dispatch_flat.unsqueeze(1),
    #             dim=3
    #         )
    #         eye = torch.eye(E*S, device=device).expand(B, -1, -1)
    #         masked_sim = similarity * (1 - eye) + eye * similarity.amax(dim=(1,2), keepdim=True)
            
    #         metrics.update({
    #             # 'dispatch_weights_similarity_min': masked_sim.min(),
    #             # 'dispatch_weights_similarity_max': masked_sim.max(),
    #             'dispatch_weights_similarity_mean': (similarity.sum() - B*E*S) / (B*E*S*(E*S-1))
    #         })

    #         # Mu similarity
    #         mu_flat = mu.view(-1, self.dim)
    #         similarity = torch.cosine_similarity(
    #             mu_flat.unsqueeze(1),
    #             mu_flat.unsqueeze(0),
    #             dim=2
    #         )
    #         eye = torch.eye(E*S, device=device)
    #         masked_sim = similarity * (1 - eye) + eye * similarity.amax()
            
    #         # metrics.update({
    #         #     'mu_similarity_min': masked_sim.min(),
    #         #     'mu_similarity_max': masked_sim.max(),
    #         #     'mu_similarity_mean': (similarity.sum() - E*S) / (E*S*(E*S-1))
    #         # })

    #     return metrics
    # ######33##############################
            
            # # Mu参数相似度 (E, S, E, S)
            # mu_perm = mu.permute(1,2,0)
            # mu_sim = cosine_psim(
            #     mu_perm,  
            #     batch_axes=(0,1), 
            #     contract_axes=(2,)  # 压缩特征维度
            # )
            # # assert mu_sim.shape == (E, S, E, S), f"Mu相似度形状错误: {mu_sim.shape}"
            # eye = torch.eye(E*S, device=mu.device).view(E, S, E, S)
            # mu_sim_masked = (1 - eye) * mu_sim
            # max_values = mu_sim_masked.amax(dim=(2,3), keepdim=True)
            # mu_sim_masked += eye * max_values
            
            # metrics.update({
            #     'mu_similarity_min': mu_sim_masked.min(),
            #     'mu_similarity_max': mu_sim_masked.max(),
            #     'mu_similarity_mean': (mu_sim.sum() - E*S) / (E*S*(E*S-1))
            # })
            
        # return metrics
    def forward(self, x, mask=None):
        B, N, D = x.shape
        print(f"输入到softmoe的：{x.shape}")
        E = self.num_experts
        S = self.compute_capacity(N)  # 动态计算slot数量
        print(f"计算的槽位数：{S}")
        # print(S)
        # 更新mu参数形状
        # self.mu.data = self.mu.data.expand(D, E, S).contiguous()
        mu = self.mu[:, :, :S]  # [D, E, S]  # [1, D, E, 1]
        # print(f"mu:{mu.shape}") # mu:torch.Size([768, 4, 36])
        # mu = mu.unsqueeze(0).expand(B, -1, -1, -1)  # [B, D, E, S]
        
        # === 输入归一化 ===
        # x = self.norm(x)
        # logits = torch.einsum('bnd,des->bnes', x * self.scale, self.mu[:, :, :S])
        x_norm = torch.nn.functional.normalize(x, p=2, dim=-1)
        mu_norm = torch.nn.functional.normalize(mu, p=2,dim=0)
        
        # print(f"x_norm:{x_norm.shape}") # x_norm:torch.Size([64, 129, 768])
        # print(f"mu_norm:{mu_norm.shape}") # mu_norm:torch.Size([768, 4, 36])
        # === 分发权重计算 ===
        # 计算logits [B, N, E, S]
        # logits = einsum('bnd,bdes->bnes', x_norm * self.scale, mu_norm)
        
        if x.numel() < mu.numel():
            x_norm = x_norm * self.scale
        else:
            mu_norm = mu_norm * self.scale
        logits = torch.einsum('bnd,des->bnes', x_norm, mu_norm)
        # logits = einsum('bnd,denp->bnes', x_norm, mu_norm)              
        # print(f"logits:{logits.shape}")
        # 添加噪声
        if self.training and self.noise_std > 0:
            logits += torch.randn_like(logits) * self.noise_std
        def stable_softmax(logits, dim):
            logits = logits - logits.amax(dim=dim, keepdim=True).detach()
            return torch.softmax(logits, dim=dim)
        # Softmax维度调整
        dispatch_weights = stable_softmax(logits, dim=1)
        # dispatch_weights = F.softmax(logits, dim=1)
        global_max = logits.max().item()   # 全局最大值
        global_min = logits.min().item()   # 全局最小值
        global_mean = logits.mean().item() # 全局平均值

        print(f"Global max: {global_max}")
        print(f"Global min: {global_min}")
        print(f"Global mean: {global_mean}")

        global_max_d = dispatch_weights.max().item()   # 全局最大值
        global_min_d = dispatch_weights.min().item()   # 全局最小值
        global_mean_d = dispatch_weights.mean().item() # 全局平均值

        print(f"d max: {global_max_d}")
        print(f"d min: {global_min_d}")
        print(f"d mean: {global_mean_d}")


        # print(f"logits.shape:{logits.shape}") # logits.shape:torch.Size([64, 129, 4, 36])
        print(f"B, N, E, S:{B, N, E, S}")
        combine_weights_flat = stable_softmax(logits.view(B, N, E*S), dim=-1)
        # combine_weights_flat = F.softmax(logits.view(B, N, E*S), dim=-1)
        # combine_weights_flat = F.softmax(logits.flatten(2), dim=-1).view_as(logits)


        combine_weights = combine_weights_flat.view(B, N, E, S)  # [B, N, E, S]
        # print(combine_weights.shape)
        # === Slot输入构造 ===
        slot_inputs = einsum('bnd,bnes->besd', x, dispatch_weights)
        print(f"slot_inputs:{slot_inputs.shape}") # slot_inputs:torch.Size([64, 4, 36, 768])

        # === 专家处理 ===
        slot_outputs = []
        for e in range(E):
            expert_input = slot_inputs[:, e]  # [B, S, D]
            expert_output = self.experts[e](expert_input)
            slot_outputs.append(expert_output)
        slot_outputs = torch.stack(slot_outputs, dim=1)  # [B, E, S, D]
        
        # 前向计算
        # slot_inputs = rearrange(slot_inputs, 'b e s d -> (b e s) d')
        # slot_outputs = torch.vmap(lambda expert, x: expert(x))(self.experts, slot_inputs)
        # slot_outputs = rearrange(slot_outputs, '(b e s) d -> b e s d', b=B, e=E)
        # === 输出聚合 ===
        slot_outputs_flat = rearrange(slot_outputs, 'b e s d -> b (e s) d')
        outputs = einsum('bsd,bns->bnd', slot_outputs_flat, combine_weights_flat)
        # outputs.register_hook(lambda grad: torch.clamp(grad, -self.grad_clip_val, self.grad_clip_val))
        if self.training:  # 仅在训练阶段注册梯度钩子
            outputs.register_hook(lambda grad: torch.clamp(grad, -self.grad_clip_val, self.grad_clip_val))
        # 指标收集
        if self.compute_metrics:
            metrics = self.get_metrics(combine_weights, dispatch_weights, mu_norm)
      
        # else:
        #     metrics = {}        
        # metrics = {'auxiliary_loss': torch.tensor(0.0)}   
        return outputs, metrics

# def test_image_input():
#     batch_size = 64
#     num_patches = 129
#     dim = 768
#     x = torch.randn(batch_size, num_patches, dim)
#     model5 = SoftMoE5(
#     dim=dim,
#     num_experts=4,
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
    # print(metrics.keys()) # 输出统计指标


# if __name__ == "__main__":
#     test_image_input()


# 25.4.28 增加输出logits，用于损失函数
class SoftMoE8(nn.Module):
    def __init__(self, dim, num_experts, capacity_factor=1.0, noise_std=0.0, dropout_rate=0.1,num_slots=None,
                 deterministic=False, compute_metrics=True, multiple_of=4,grad_clip_val=1.0,max_seq_len=256,precision=torch.float32):
        super().__init__()
        self.dim = dim
        self.num_experts = num_experts
        self.capacity_factor = capacity_factor
        self.noise_std = noise_std
        self.deterministic = deterministic
        self.compute_metrics = compute_metrics
        self.num_slots = num_slots
        self.multiple_of = multiple_of
        self.grad_clip_val = grad_clip_val
        # 新增参数预定义最大序列长度
        self.max_seq_len = max_seq_len
        self.multiple_of = 4
        self.alpha = 0.4  # 遮挡区域缩放下限
        self.beta = 1.2   # 重要区域缩放上限
        self.compute_similarity_metrics = True
        self.precision = precision
        # 输入归一化层
        # self.norm = nn.LayerNorm(dim)
        
        # 可学习参数（对应论文中的mu和scale）
        fan_in = dim
        std = math.sqrt(1.0 / fan_in)
        max_slots = self._compute_max_slots(max_seq_len, num_experts)
        # self.mu = nn.Parameter(torch.empty(dim, num_experts, 1))  # 初始shape需动态调整
        self.mu = nn.Parameter(torch.empty(dim, num_experts, max_slots))
        # print(f"self.mu:{self.mu.shape}")
        nn.init.normal_(self.mu, mean=0, std=1/math.sqrt(dim))
        # nn.init.normal_(self.mu, mean=0, std=std)  # 手动实现lecun_normal
        self.scale = nn.Parameter(torch.ones(1))
        self.threshold=0.4
        
        # 专家网络
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, 4*dim//self.num_experts),
                nn.GELU(),
                nn.Dropout(dropout_rate),
                nn.Linear(4*dim//self.num_experts, dim),
                nn.Dropout(dropout_rate),)
            for _ in range(num_experts)])
        for expert in self.experts:
            nn.init.kaiming_normal_(expert[0].weight, nonlinearity='relu')
            nn.init.zeros_(expert[0].bias)
            nn.init.xavier_normal_(expert[3].weight)
            nn.init.zeros_(expert[3].bias)
        # 初始化参数
        # nn.init.lecun_normal_(self.mu)
        # nn.init.ones_(self.scale)

    # def compute_capacity(self, num_tokens: int) -> int:
    #     """确保 num_experts * num_slots >= num_tokens"""
    #     num_slots = math.ceil(num_tokens / self.num_experts)
    #     num_slots = num_slots + (-num_slots) % self.multiple_of
    #     return max(num_slots, 1)

    # def compute_capacity(self, num_tokens: int) -> int:
    #     num_slots = (num_tokens + self.num_experts - 1) // self.num_experts
    #     remainder = num_slots % self.multiple_of
    #     return num_slots + (self.multiple_of - remainder) % self.multiple_of
    def compute_capacity(
        self,
        num_tokens: int,
        ceil_or_round: str = "ceil"
    ) -> int:
        """修正后的容量计算"""
        scaled_tokens = num_tokens * self.capacity_factor
        
        # 计算基础容量
        if ceil_or_round == "ceil":
            capacity = math.ceil(scaled_tokens / self.num_experts)
        elif ceil_or_round == "round":
            capacity = round(scaled_tokens / self.num_experts)
        else:
            raise ValueError(f"Invalid ceil_or_round: {ceil_or_round}")

        # 确保最小容量
        capacity = max(capacity, 1)

        # 对齐到multiple_of
        if self.multiple_of > 1:
            remainder = capacity % self.multiple_of
            if remainder != 0:
                capacity += self.multiple_of - remainder

        # 输出警告（可选）
        actual_factor = capacity * self.num_experts / num_tokens
        if abs(actual_factor - self.capacity_factor) > 1e-6:
            print(f"Warning: Target cap_factor {self.capacity_factor}, Actual {actual_factor:.2f}")

        return capacity

    def cosine_psim(self, x, contract_dims, batch_dims, eps=1e-9):
        """PyTorch版本的cosine_psim函数"""
        # 沿contract_dims归一化
        norm = torch.rsqrt((x**2).sum(dim=contract_dims, keepdim=True) + eps)
        x_norm = x * norm
        
        # 重新排列维度以进行批处理矩阵乘法
        all_dims = list(range(x_norm.ndim))
        keep_dims = [d for d in all_dims if d not in contract_dims]
        x_flat = x_norm.permute(*keep_dims, *contract_dims).flatten(start_dim=len(keep_dims))
        
        # 计算批处理点积
        similarity = torch.einsum('...i,...j->...ij', x_flat, x_flat)
        return similarity
    def _compute_max_slots(self, max_seq_len, num_experts):
        """预计算最大可能的slot数量"""
        slots = math.ceil(max_seq_len / num_experts)
        return slots + (-slots % self.multiple_of)
    def get_metrics(self, combine_weights, dispatch_weights, mu):
        B, N, E, S = combine_weights.shape
        device = combine_weights.device
        metrics = {}

        if self.compute_similarity_metrics:
            # ===== Combine Weights相似度 =====
            with torch.autocast(device_type='cuda', dtype=self.precision):  # 混合精度支持
                # 计算相似度矩阵 [B, N, N]
                cw_sim = self.cosine_psim(
                    combine_weights,
                    contract_dims=(2, 3),  # 对应E,S维度
                    batch_dims=(0,)        # B维度作为批处理
                )
                
                # 创建对角线掩码
                eye = torch.eye(N, device=device).expand(B, -1, -1)
                
                # 计算均值（排除对角线）
                sum_total = cw_sim.sum()
                sum_diag = torch.diagonal(cw_sim, dim1=1, dim2=2).sum()
                metrics['combine_weights_similarity_mean'] = (sum_total - sum_diag) / (B * N * (N - 1))

            # ===== Dispatch Weights相似度 =====
            with torch.autocast(device_type='cuda', dtype=self.precision):
                # 重新排列维度为[B, E, S, N]
                dw_permuted = dispatch_weights.permute(0, 2, 3, 1)
                
                # 计算相似度矩阵 [B, E*S, E*S]
                dw_sim = self.cosine_psim(
                    dw_permuted,
                    contract_dims=(3,),    # N维度
                    batch_dims=(0,)        # B维度
                )
                
                # 创建对角线掩码
                eye = torch.eye(E*S, device=device).expand(B, -1, -1)
                
                # 计算均值（排除对角线）
                sum_total = dw_sim.sum()
                sum_diag = torch.diagonal(dw_sim, dim1=1, dim2=2).sum()
                metrics['dispatch_weights_similarity_mean'] = (sum_total - sum_diag) / (B * E * S * (E*S - 1))

        return metrics
    def forward(self, x,attn_weight, mode='original'):
        B, N, D = x.shape
        print(mode)
        if attn_weight is None:
            print("传入softmoe的attn_weight为None")
        else:
            print("attn_weight.max:")
            print(attn_weight.max().item())
            print("attn_weight.min:")
            print(attn_weight.min().item())
            print("attn_weight.mean:")
            print(attn_weight.mean().item())
        # attn_weight_full = torch.cat([torch.ones(B, 1).to(device), attn_weight], dim=1)  
        if mode !='original':
            attn_weight_full = torch.cat([torch.ones(B, 1).to(attn_weight.device), attn_weight], dim=1) 
            scaling_factor = torch.where(
            attn_weight_full < self.threshold,  # 条件：低于阈值
            attn_weight_full,              # 满足条件时：缩放系数 = attn_weight_full
            torch.ones_like(attn_weight_full)  # 不满足条件时：缩放系数 = 1.0
        )
            # attn_weight_scaled = self.alpha + (self.beta - self.alpha) * attn_weight_full
            # attn_weight_expanded = attn_weight_scaled.unsqueeze(-1).unsqueeze(-1)
            attn_weight_expanded = scaling_factor.unsqueeze(-1).unsqueeze(-1)
            # attn_weight_expanded = attn_weight_full.unsqueeze(-1).unsqueeze(-1)
            
            print(f"attn_weight_expanded.max:{attn_weight_expanded.max().item()}")
            print(f"attn_weight_expanded.min:{attn_weight_expanded.min().item()}")
            print(f"attn_weight_expanded.mean:{attn_weight_expanded.mean().item()}")
        B, N, D = x.shape
        E = self.num_experts
        if self.num_slots is None:
            S = self.compute_capacity(N)  # 动态计算slot数量
        else:
            S = self.num_slots
        print(f"计算的槽位数：{S}")
        # print(S)
        # 更新mu参数形状
        # self.mu.data = self.mu.data.expand(D, E, S).contiguous()
        mu = self.mu[:, :, :S]  # [D, E, S]  # [1, D, E, 1]
        # print(f"mu:{mu.shape}") # mu:torch.Size([768, 4, 36])
        # mu = mu.unsqueeze(0).expand(B, -1, -1, -1)  # [B, D, E, S]
        
        # === 输入归一化 ===
        # x = self.norm(x)
        # logits = torch.einsum('bnd,des->bnes', x * self.scale, self.mu[:, :, :S])
        x_norm = torch.nn.functional.normalize(x, p=2, dim=-1)
        mu_norm = torch.nn.functional.normalize(mu, p=2,dim=0)
        
        # print(f"x_norm:{x_norm.shape}") # x_norm:torch.Size([64, 129, 768])
        # print(f"mu_norm:{mu_norm.shape}") # mu_norm:torch.Size([768, 4, 36])
        # === 分发权重计算 ===
        # 计算logits [B, N, E, S]
        # logits = einsum('bnd,bdes->bnes', x_norm * self.scale, mu_norm)
        
        if x.numel() < mu.numel():
            x_norm = x_norm * self.scale
        else:
            mu_norm = mu_norm * self.scale
        logits = torch.einsum('bnd,des->bnes', x_norm, mu_norm)
        if mode !='original':
            logits = logits * attn_weight_expanded

        # logits = einsum('bnd,denp->bnes', x_norm, mu_norm)              
        # print(f"logits:{logits.shape}")
        # 添加噪声
        if self.training and self.noise_std > 0:
            logits += torch.randn_like(logits) * self.noise_std
        def stable_softmax(logits, dim):
            logits = logits - logits.amax(dim=dim, keepdim=True).detach()
            return torch.softmax(logits, dim=dim)
        # Softmax维度调整
        dispatch_weights = stable_softmax(logits, dim=1)
        print("修改前：---------")
        global_max_d = dispatch_weights.max().item()   # 全局最大值
        global_min_d = dispatch_weights.min().item()   # 全局最小值
        global_mean_d = dispatch_weights.mean().item() # 全局平均值

        print(f"d max: {global_max_d}")
        print(f"d min: {global_min_d}")
        print(f"d mean: {global_mean_d}")
        if mode !='original':
            dispatch_weights_weighted = dispatch_weights * attn_weight_expanded
            dispatch_weights = dispatch_weights_weighted

            print("修改后：---------")
            print(f"d max: {dispatch_weights.max().item()}")
            print(f"d min: {dispatch_weights.min().item()}")
            print(f"d mean: {dispatch_weights.mean().item()}")

        # dispatch_weights = F.softmax(logits, dim=1)
        # print(f"logits.shape:{logits.shape}") # logits.shape:torch.Size([64, 129, 4, 36])
        print(f"B, N, E, S:{B, N, E, S}")
        combine_weights_flat = stable_softmax(logits.view(B, N, E*S), dim=-1)
        combine_weights = combine_weights_flat.view(B, N, E, S)
        # combine_weights_flat = F.softmax(logits.view(B, N, E*S), dim=-1)
        # combine_weights_flat = F.softmax(logits.flatten(2), dim=-1).view_as(logits)


        # combine_weights = combine_weights_flat.view(B, N, E, S)  # [B, N, E, S]
        # print(combine_weights.shape)
        # === Slot输入构造 ===
        slot_inputs = einsum('bnd,bnes->besd', x, dispatch_weights)
        print(f"slot_inputs:{slot_inputs.shape}") # slot_inputs:torch.Size([64, 4, 36, 768])

        # === 专家处理 ===
        slot_outputs = []
        for e in range(E):
            expert_input = slot_inputs[:, e]  # [B, S, D]
            expert_output = self.experts[e](expert_input)
            slot_outputs.append(expert_output)
        slot_outputs = torch.stack(slot_outputs, dim=1)  # [B, E, S, D]
        
        # 前向计算
        # slot_inputs = rearrange(slot_inputs, 'b e s d -> (b e s) d')
        # slot_outputs = torch.vmap(lambda expert, x: expert(x))(self.experts, slot_inputs)
        # slot_outputs = rearrange(slot_outputs, '(b e s) d -> b e s d', b=B, e=E)
        # === 输出聚合 ===
        slot_outputs_flat = rearrange(slot_outputs, 'b e s d -> b (e s) d')
        outputs = einsum('bsd,bns->bnd', slot_outputs_flat, combine_weights_flat)
        # outputs.register_hook(lambda grad: torch.clamp(grad, -self.grad_clip_val, self.grad_clip_val))
        if self.training:  # 仅在训练阶段注册梯度钩子
            outputs.register_hook(lambda grad: torch.clamp(grad, -self.grad_clip_val, self.grad_clip_val))
        # 指标收集
        if self.compute_metrics:
            metrics = self.get_metrics(combine_weights, dispatch_weights, self.mu)
        # else:
        #     metrics = {}        
        # metrics = {'auxiliary_loss': torch.tensor(0.0)}   
        return outputs, metrics
    




# 25.4.29 在SoftMoE7的基础上，额外返回一个特征以便损失计算
class SoftMoE9(nn.Module):
    def __init__(self, dim, num_experts, capacity_factor=1.0, noise_std=0.0, dropout_rate=0.1,
                 deterministic=False, compute_metrics=True, multiple_of=4,grad_clip_val=1.0,max_seq_len=256):
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
        
        # 可学习参数（对应论文中的mu和scale）
        fan_in = dim
        std = math.sqrt(1.0 / fan_in)
        max_slots = self._compute_max_slots(max_seq_len, num_experts)
        # self.mu = nn.Parameter(torch.empty(dim, num_experts, 1))  # 初始shape需动态调整
        self.mu = nn.Parameter(torch.empty(dim, num_experts, max_slots))
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
        for expert in self.experts:
            nn.init.kaiming_normal_(expert[0].weight, nonlinearity='relu')
            # nn.init.orthogonal_(expert[0].weight)
            # expert[0].weight.data.mul_(math.sqrt(2 / (1 + math.sqrt(2))))  # GELU增益
                # 第二层：零初始化避免初始阶段输出偏移
            # nn.init.zeros_(expert[3].weight)
            nn.init.zeros_(expert[0].bias)
            nn.init.xavier_normal_(expert[3].weight)
            nn.init.zeros_(expert[3].bias)
        # 初始化参数
        # nn.init.lecun_normal_(self.mu)
        # nn.init.ones_(self.scale)

    # def compute_capacity(self, num_tokens: int) -> int:
    #     """确保 num_experts * num_slots >= num_tokens"""
    #     num_slots = math.ceil(num_tokens / self.num_experts)
    #     num_slots = num_slots + (-num_slots) % self.multiple_of
    #     return max(num_slots, 1)

    # def compute_capacity(self, num_tokens: int) -> int:
    #     num_slots = (num_tokens + self.num_experts - 1) // self.num_experts
    #     remainder = num_slots % self.multiple_of
    #     return num_slots + (self.multiple_of - remainder) % self.multiple_of
    def compute_capacity(
        self,
        num_tokens: int,
        ceil_or_round: str = "ceil"
    ) -> int:
        """修正后的容量计算"""
        # 确保必要的参数存在
        # assert hasattr(self, 'capacity_factor'), "需要定义capacity_factor"
        # assert hasattr(self, 'multiple_of'), "需要定义multiple_of"

        # 应用capacity_factor
        scaled_tokens = num_tokens * self.capacity_factor
        
        # 计算基础容量
        if ceil_or_round == "ceil":
            capacity = math.ceil(scaled_tokens / self.num_experts)
        elif ceil_or_round == "round":
            capacity = round(scaled_tokens / self.num_experts)
        else:
            raise ValueError(f"Invalid ceil_or_round: {ceil_or_round}")

        # 确保最小容量
        capacity = max(capacity, 1)

        # 对齐到multiple_of
        if self.multiple_of > 1:
            remainder = capacity % self.multiple_of
            if remainder != 0:
                capacity += self.multiple_of - remainder

        # 输出警告（可选）
        actual_factor = capacity * self.num_experts / num_tokens
        if abs(actual_factor - self.capacity_factor) > 1e-6:
            print(f"Warning: Target cap_factor {self.capacity_factor}, Actual {actual_factor:.2f}")

        return capacity

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
    def forward(self, x, mask=None):
        B, N, D = x.shape
        print(f"输入到softmoe的：{x.shape}")
        E = self.num_experts
        S = self.compute_capacity(N)  # 动态计算slot数量
        print(f"计算的槽位数：{S}")
        # print(S)
        # 更新mu参数形状
        # self.mu.data = self.mu.data.expand(D, E, S).contiguous()
        mu = self.mu[:, :, :S]  # [D, E, S]  # [1, D, E, 1]
        # print(f"mu:{mu.shape}") # mu:torch.Size([768, 4, 36])
        # mu = mu.unsqueeze(0).expand(B, -1, -1, -1)  # [B, D, E, S]
        
        # === 输入归一化 ===
        # x = self.norm(x)
        # logits = torch.einsum('bnd,des->bnes', x * self.scale, self.mu[:, :, :S])
        x_norm = torch.nn.functional.normalize(x, p=2, dim=-1)
        mu_norm = torch.nn.functional.normalize(mu, p=2,dim=0)
        
        # print(f"x_norm:{x_norm.shape}") # x_norm:torch.Size([64, 129, 768])
        # print(f"mu_norm:{mu_norm.shape}") # mu_norm:torch.Size([768, 4, 36])
        # === 分发权重计算 ===
        # 计算logits [B, N, E, S]
        # logits = einsum('bnd,bdes->bnes', x_norm * self.scale, mu_norm)
        
        if x.numel() < mu.numel():
            x_norm = x_norm * self.scale
        else:
            mu_norm = mu_norm * self.scale
        logits = torch.einsum('bnd,des->bnes', x_norm, mu_norm)
        # logits = einsum('bnd,denp->bnes', x_norm, mu_norm)              
        # print(f"logits:{logits.shape}")
        # 添加噪声
        if self.training and self.noise_std > 0:
            logits += torch.randn_like(logits) * self.noise_std
        def stable_softmax(logits, dim):
            logits = logits - logits.amax(dim=dim, keepdim=True).detach()
            return torch.softmax(logits, dim=dim)
        # Softmax维度调整
        dispatch_weights = stable_softmax(logits, dim=1)
        # dispatch_weights = F.softmax(logits, dim=1)
        global_max = logits.max().item()   # 全局最大值
        global_min = logits.min().item()   # 全局最小值
        global_mean = logits.mean().item() # 全局平均值

        print(f"Global max: {global_max}")
        print(f"Global min: {global_min}")
        print(f"Global mean: {global_mean}")

        global_max_d = dispatch_weights.max().item()   # 全局最大值
        global_min_d = dispatch_weights.min().item()   # 全局最小值
        global_mean_d = dispatch_weights.mean().item() # 全局平均值

        print(f"d max: {global_max_d}") 
        print(f"d min: {global_min_d}")
        print(f"d mean: {global_mean_d}")


        # print(f"logits.shape:{logits.shape}") # logits.shape:torch.Size([64, 129, 4, 36])
        print(f"B, N, E, S:{B, N, E, S}")
        combine_weights_flat = stable_softmax(logits.view(B, N, E*S), dim=-1)
        # combine_weights_flat = F.softmax(logits.view(B, N, E*S), dim=-1)
        # combine_weights_flat = F.softmax(logits.flatten(2), dim=-1).view_as(logits)


        # combine_weights = combine_weights_flat.view(B, N, E, S)  # [B, N, E, S]
        # print(combine_weights.shape)
        # === Slot输入构造 ===
        slot_inputs = einsum('bnd,bnes->besd', x, dispatch_weights)
        print(f"slot_inputs:{slot_inputs.shape}") # slot_inputs:torch.Size([64, 4, 36, 768])

        # === 专家处理 ===
        slot_outputs = []
        for e in range(E):
            expert_input = slot_inputs[:, e]  # [B, S, D]
            expert_output = self.experts[e](expert_input)
            slot_outputs.append(expert_output)
        slot_outputs = torch.stack(slot_outputs, dim=1)  # [B, E, S, D]
        
        # 前向计算
        # slot_inputs = rearrange(slot_inputs, 'b e s d -> (b e s) d')
        # slot_outputs = torch.vmap(lambda expert, x: expert(x))(self.experts, slot_inputs)
        # slot_outputs = rearrange(slot_outputs, '(b e s) d -> b e s d', b=B, e=E)
        # === 输出聚合 ===
        slot_outputs_flat = rearrange(slot_outputs, 'b e s d -> b (e s) d')
        outputs = einsum('bsd,bns->bnd', slot_outputs_flat, combine_weights_flat)
        # outputs.register_hook(lambda grad: torch.clamp(grad, -self.grad_clip_val, self.grad_clip_val))
        if self.training:  # 仅在训练阶段注册梯度钩子
            outputs.register_hook(lambda grad: torch.clamp(grad, -self.grad_clip_val, self.grad_clip_val))
        # 指标收集
        # if self.compute_metrics:
        #     metrics = self.get_metrics(combine_weights, dispatch_weights, self.mu)
        # else:
        #     metrics = {}        
        metrics = {'auxiliary_loss': torch.tensor(0.0)}   
        return outputs, metrics





class SoftMoE11(nn.Module):
    def __init__(self, dim, num_experts, capacity_factor=1.0, noise_std=0.0, dropout_rate=0.1,num_slots=None,
                 deterministic=False, compute_metrics=True, multiple_of=4,grad_clip_val=1.0,max_seq_len=256,precision=torch.float32):
        super().__init__()
        self.dim = dim
        self.num_experts = num_experts
        self.capacity_factor = capacity_factor
        self.noise_std = noise_std
        self.deterministic = deterministic
        self.compute_metrics = compute_metrics
        self.num_slots = num_slots
        self.multiple_of = multiple_of
        self.grad_clip_val = grad_clip_val
        # 新增参数预定义最大序列长度
        self.max_seq_len = max_seq_len
        self.multiple_of = 4
        self.alpha = 0.4  # 遮挡区域缩放下限
        self.beta = 1.2   # 重要区域缩放上限
        self.compute_similarity_metrics = True
        self.precision = precision
        # 输入归一化层
        # self.norm = nn.LayerNorm(dim)
        
        # 可学习参数（对应论文中的mu和scale）
        fan_in = dim
        std = math.sqrt(1.0 / fan_in)
        max_slots = self._compute_max_slots(max_seq_len, num_experts)
        # self.mu = nn.Parameter(torch.empty(dim, num_experts, 1))  # 初始shape需动态调整
        self.mu = nn.Parameter(torch.empty(dim, num_experts, max_slots))
        # print(f"self.mu:{self.mu.shape}")
        nn.init.normal_(self.mu, mean=0, std=1/math.sqrt(dim))
        # nn.init.normal_(self.mu, mean=0, std=std)  # 手动实现lecun_normal
        self.scale = nn.Parameter(torch.ones(1))
        self.threshold=0.4
        
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
                nn.Linear(dim, 4*dim//self.num_experts),
                nn.GELU(),
                nn.Dropout(dropout_rate),
                nn.Linear(4*dim//self.num_experts, dim),
                nn.Dropout(dropout_rate))   
        nn.init.kaiming_normal_(self.cls_experts[0].weight, nonlinearity='relu')
        nn.init.zeros_(self.cls_experts[0].bias)    
        nn.init.kaiming_normal_(self.cls_experts[3].weight, nonlinearity='relu')
        nn.init.zeros_(self.cls_experts[3].bias)  
        for expert in self.experts:
            nn.init.kaiming_normal_(expert[0].weight, nonlinearity='relu')
            # nn.init.orthogonal_(expert[0].weight)
            # expert[0].weight.data.mul_(math.sqrt(2 / (1 + math.sqrt(2))))  # GELU增益
                # 第二层：零初始化避免初始阶段输出偏移
            # nn.init.zeros_(expert[3].weight)
            nn.init.zeros_(expert[0].bias)
            nn.init.xavier_normal_(expert[3].weight)
            nn.init.zeros_(expert[3].bias)
        # 初始化参数
        # nn.init.lecun_normal_(self.mu)
        # nn.init.ones_(self.scale)

    # def compute_capacity(self, num_tokens: int) -> int:
    #     """确保 num_experts * num_slots >= num_tokens"""
    #     num_slots = math.ceil(num_tokens / self.num_experts)
    #     num_slots = num_slots + (-num_slots) % self.multiple_of
    #     return max(num_slots, 1)

    # def compute_capacity(self, num_tokens: int) -> int:
    #     num_slots = (num_tokens + self.num_experts - 1) // self.num_experts
    #     remainder = num_slots % self.multiple_of
    #     return num_slots + (self.multiple_of - remainder) % self.multiple_of
    def compute_capacity(
        self,
        num_tokens: int,
        ceil_or_round: str = "ceil"
    ) -> int:
        """修正后的容量计算"""
        # 确保必要的参数存在
        # assert hasattr(self, 'capacity_factor'), "需要定义capacity_factor"
        # assert hasattr(self, 'multiple_of'), "需要定义multiple_of"

        # 应用capacity_factor
        scaled_tokens = num_tokens * self.capacity_factor
        
        # 计算基础容量
        if ceil_or_round == "ceil":
            capacity = math.ceil(scaled_tokens / self.num_experts)
        elif ceil_or_round == "round":
            capacity = round(scaled_tokens / self.num_experts)
        else:
            raise ValueError(f"Invalid ceil_or_round: {ceil_or_round}")

        # 确保最小容量
        capacity = max(capacity, 1)

        # 对齐到multiple_of
        if self.multiple_of > 1:
            remainder = capacity % self.multiple_of
            if remainder != 0:
                capacity += self.multiple_of - remainder

        # 输出警告（可选）
        actual_factor = capacity * self.num_experts / num_tokens
        if abs(actual_factor - self.capacity_factor) > 1e-6:
            print(f"Warning: Target cap_factor {self.capacity_factor}, Actual {actual_factor:.2f}")

        return capacity

    def cosine_psim(self, x, contract_dims, batch_dims, eps=1e-9):
        """PyTorch版本的cosine_psim函数"""
        # 沿contract_dims归一化
        norm = torch.rsqrt((x**2).sum(dim=contract_dims, keepdim=True) + eps)
        x_norm = x * norm
        
        # 重新排列维度以进行批处理矩阵乘法
        all_dims = list(range(x_norm.ndim))
        keep_dims = [d for d in all_dims if d not in contract_dims]
        x_flat = x_norm.permute(*keep_dims, *contract_dims).flatten(start_dim=len(keep_dims))
        
        # 计算批处理点积
        similarity = torch.einsum('...i,...j->...ij', x_flat, x_flat)
        return similarity
    def _compute_max_slots(self, max_seq_len, num_experts):
        """预计算最大可能的slot数量"""
        slots = math.ceil(max_seq_len / num_experts)
        return slots + (-slots % self.multiple_of)
    def get_metrics(self, combine_weights, dispatch_weights, mu):
        B, N, E, S = combine_weights.shape
        device = combine_weights.device
        metrics = {}

        if self.compute_similarity_metrics:
            # ===== Combine Weights相似度 =====
            with torch.autocast(device_type='cuda', dtype=self.precision):  # 混合精度支持
                # 计算相似度矩阵 [B, N, N]
                cw_sim = self.cosine_psim(
                    combine_weights,
                    contract_dims=(2, 3),  # 对应E,S维度
                    batch_dims=(0,)        # B维度作为批处理
                )
                
                # 创建对角线掩码
                eye = torch.eye(N, device=device).expand(B, -1, -1)
                
                # 计算均值（排除对角线）
                sum_total = cw_sim.sum()
                sum_diag = torch.diagonal(cw_sim, dim1=1, dim2=2).sum()
                metrics['combine_weights_similarity_mean'] = (sum_total - sum_diag) / (B * N * (N - 1))

            # ===== Dispatch Weights相似度 =====
            with torch.autocast(device_type='cuda', dtype=self.precision):
                # 重新排列维度为[B, E, S, N]
                dw_permuted = dispatch_weights.permute(0, 2, 3, 1)
                
                # 计算相似度矩阵 [B, E*S, E*S]
                dw_sim = self.cosine_psim(
                    dw_permuted,
                    contract_dims=(3,),    # N维度
                    batch_dims=(0,)        # B维度
                )
                
                # 创建对角线掩码
                eye = torch.eye(E*S, device=device).expand(B, -1, -1)
                
                # 计算均值（排除对角线）
                sum_total = dw_sim.sum()
                sum_diag = torch.diagonal(dw_sim, dim1=1, dim2=2).sum()
                metrics['dispatch_weights_similarity_mean'] = (sum_total - sum_diag) / (B * E * S * (E*S - 1))

        return metrics
    def forward(self, x,attn_weight, mode='original'):
        cls_token =x[:, 0, :]
        cls_emd = self.cls_experts(cls_token)
        cls_emd = cls_emd.unsqueeze(1)
        print(f"cls_emd.shape:{cls_emd.shape}")
        x = x[:, 1:, :]
        B, N, D = x.shape
        print(mode)
        if attn_weight is None:
            print("传入softmoe的attn_weight为None")
        else:
            print("attn_weight.max:")
            print(attn_weight.max().item())
            print("attn_weight.min:")
            print(attn_weight.min().item())
            print("attn_weight.mean:")
            print(attn_weight.mean().item())
        # attn_weight_full = torch.cat([torch.ones(B, 1).to(device), attn_weight], dim=1)  
        if mode !='original':
            attn_weight_full = torch.cat([torch.ones(B, 1).to(attn_weight.device), attn_weight], dim=1) 
            scaling_factor = torch.where(
            attn_weight_full < self.threshold,  # 条件：低于阈值
            attn_weight_full,              # 满足条件时：缩放系数 = attn_weight_full
            torch.ones_like(attn_weight_full)  # 不满足条件时：缩放系数 = 1.0
        )
            # attn_weight_scaled = self.alpha + (self.beta - self.alpha) * attn_weight_full
            # attn_weight_expanded = attn_weight_scaled.unsqueeze(-1).unsqueeze(-1)
            attn_weight_expanded = scaling_factor.unsqueeze(-1).unsqueeze(-1)
            # attn_weight_expanded = attn_weight_full.unsqueeze(-1).unsqueeze(-1)
            
            print(f"attn_weight_expanded.max:{attn_weight_expanded.max().item()}")
            print(f"attn_weight_expanded.min:{attn_weight_expanded.min().item()}")
            print(f"attn_weight_expanded.mean:{attn_weight_expanded.mean().item()}")
        B, N, D = x.shape
        print(f"输入到softmoe的：{x.shape}")
        E = self.num_experts
        if self.num_slots is None:
            S = self.compute_capacity(N)  # 动态计算slot数量
        else:
            S = self.num_slots
        print(f"计算的槽位数：{S}")
        # print(S)
        # 更新mu参数形状
        # self.mu.data = self.mu.data.expand(D, E, S).contiguous()
        mu = self.mu[:, :, :S]  # [D, E, S]  # [1, D, E, 1]
        # print(f"mu:{mu.shape}") # mu:torch.Size([768, 4, 36])
        # mu = mu.unsqueeze(0).expand(B, -1, -1, -1)  # [B, D, E, S]
        
        # === 输入归一化 ===
        # x = self.norm(x)
        # logits = torch.einsum('bnd,des->bnes', x * self.scale, self.mu[:, :, :S])
        x_norm = torch.nn.functional.normalize(x, p=2, dim=-1)
        mu_norm = torch.nn.functional.normalize(mu, p=2,dim=0)
        
        # print(f"x_norm:{x_norm.shape}") # x_norm:torch.Size([64, 129, 768])
        # print(f"mu_norm:{mu_norm.shape}") # mu_norm:torch.Size([768, 4, 36])
        # === 分发权重计算 ===
        # 计算logits [B, N, E, S]
        # logits = einsum('bnd,bdes->bnes', x_norm * self.scale, mu_norm)
        
        if x.numel() < mu.numel():
            x_norm = x_norm * self.scale
        else:
            mu_norm = mu_norm * self.scale
        logits = torch.einsum('bnd,des->bnes', x_norm, mu_norm)
        if mode !='original':
            logits = logits * attn_weight_expanded

        # logits = einsum('bnd,denp->bnes', x_norm, mu_norm)              
        # print(f"logits:{logits.shape}")
        # 添加噪声
        if self.training and self.noise_std > 0:
            logits += torch.randn_like(logits) * self.noise_std
        def stable_softmax(logits, dim):
            logits = logits - logits.amax(dim=dim, keepdim=True).detach()
            return torch.softmax(logits, dim=dim)
        # Softmax维度调整
        dispatch_weights = stable_softmax(logits, dim=1)
        print("修改前：---------")
        global_max_d = dispatch_weights.max().item()   # 全局最大值
        global_min_d = dispatch_weights.min().item()   # 全局最小值
        global_mean_d = dispatch_weights.mean().item() # 全局平均值

        print(f"d max: {global_max_d}")
        print(f"d min: {global_min_d}")
        print(f"d mean: {global_mean_d}")
        if mode !='original':
            dispatch_weights_weighted = dispatch_weights * attn_weight_expanded
            dispatch_weights = dispatch_weights_weighted
            print("修改后：---------")
            print(f"d max: {dispatch_weights.max().item()}")
            print(f"d min: {dispatch_weights.min().item()}")
            print(f"d mean: {dispatch_weights.mean().item()}")

        # dispatch_weights = F.softmax(logits, dim=1)



        # print(f"logits.shape:{logits.shape}") # logits.shape:torch.Size([64, 129, 4, 36])
        print(f"B, N, E, S:{B, N, E, S}")
        combine_weights_flat = stable_softmax(logits.view(B, N, E*S), dim=-1)
        combine_weights = combine_weights_flat.view(B, N, E, S)
        # combine_weights_flat = F.softmax(logits.view(B, N, E*S), dim=-1)
        # combine_weights_flat = F.softmax(logits.flatten(2), dim=-1).view_as(logits)


        # combine_weights = combine_weights_flat.view(B, N, E, S)  # [B, N, E, S]
        # print(combine_weights.shape)
        # === Slot输入构造 ===
        slot_inputs = einsum('bnd,bnes->besd', x, dispatch_weights)
        print(f"slot_inputs:{slot_inputs.shape}") # slot_inputs:torch.Size([64, 4, 36, 768])

        # === 专家处理 ===
        slot_outputs = []
        for e in range(E):
            expert_input = slot_inputs[:, e]  # [B, S, D]
            expert_output = self.experts[e](expert_input)
            slot_outputs.append(expert_output)
        slot_outputs = torch.stack(slot_outputs, dim=1)  # [B, E, S, D]
        
        # 前向计算
        # slot_inputs = rearrange(slot_inputs, 'b e s d -> (b e s) d')
        # slot_outputs = torch.vmap(lambda expert, x: expert(x))(self.experts, slot_inputs)
        # slot_outputs = rearrange(slot_outputs, '(b e s) d -> b e s d', b=B, e=E)
        # === 输出聚合 ===
        slot_outputs_flat = rearrange(slot_outputs, 'b e s d -> b (e s) d')
        img_output = einsum('bsd,bns->bnd', slot_outputs_flat, combine_weights_flat)

        outputs = torch.cat([cls_emd, img_output], dim=1)
        # outputs.register_hook(lambda grad: torch.clamp(grad, -self.grad_clip_val, self.grad_clip_val))
        if self.training:  # 仅在训练阶段注册梯度钩子
            outputs.register_hook(lambda grad: torch.clamp(grad, -self.grad_clip_val, self.grad_clip_val))
        # 指标收集
        if self.compute_metrics:
            metrics = self.get_metrics(combine_weights, dispatch_weights, self.mu)
        # else:
        #     metrics = {}        
        # metrics = {'auxiliary_loss': torch.tensor(0.0)}   
        return outputs, metrics



# 5-11新增全局专家专门处理cls的特征，FFNs处理128个小块的特征
class SoftMoE10(nn.Module):
    def __init__(self, dim, num_experts, capacity_factor=1.0, noise_std=0.0, dropout_rate=0.1,
                 deterministic=False, compute_metrics=True, multiple_of=4,grad_clip_val=1.0,max_seq_len=256,
                 compute_similarity_metrics=True, precision=torch.float32):
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
        self.compute_similarity_metrics = True
        self.precision = precision
        # 输入归一化层
        # self.norm = nn.LayerNorm(dim)
        
        # 可学习参数（对应论文中的mu和scale）
        fan_in = dim
        std = math.sqrt(1.0 / fan_in)
        max_slots = self._compute_max_slots(max_seq_len, num_experts)
        # self.mu = nn.Parameter(torch.empty(dim, num_experts, 1))  # 初始shape需动态调整
        self.mu = nn.Parameter(torch.empty(dim, num_experts, max_slots))
        # print(f"self.mu:{self.mu.shape}")
        # nn.init.normal_(self.mu, mean=0, std=1/math.sqrt(dim))
        nn.init.kaiming_normal_(self.mu, mode='fan_in', nonlinearity='linear')
        # nn.init.normal_(self.mu, mean=0, std=std)  # 手动实现lecun_normal
        self.scale = nn.Parameter(torch.ones(1))
        
        self.cls_experts =  nn.Sequential(
                nn.Linear(dim, 4*dim//self.num_experts),
                nn.GELU(),
                nn.Dropout(dropout_rate),
                nn.Linear(4*dim//self.num_experts, dim),
                nn.Dropout(dropout_rate))   
        nn.init.kaiming_normal_(self.cls_experts[0].weight, nonlinearity='relu')
        nn.init.zeros_(self.cls_experts[0].bias)    
        nn.init.kaiming_normal_(self.cls_experts[3].weight, nonlinearity='relu')
        nn.init.zeros_(self.cls_experts[3].bias)          
        # 专家网络
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, 4*dim//self.num_experts),
                nn.GELU(),
                nn.Dropout(dropout_rate),
                nn.Linear(4*dim//self.num_experts, dim),
                nn.Dropout(dropout_rate),)
            for _ in range(num_experts)])
        for expert in self.experts:
            nn.init.kaiming_normal_(expert[0].weight, nonlinearity='relu')
            # nn.init.orthogonal_(expert[0].weight)
            # expert[0].weight.data.mul_(math.sqrt(2 / (1 + math.sqrt(2))))  # GELU增益
                # 第二层：零初始化避免初始阶段输出偏移
            # nn.init.zeros_(expert[3].weight)
            nn.init.zeros_(expert[0].bias)
            nn.init.xavier_normal_(expert[3].weight)
            nn.init.zeros_(expert[3].bias)
        # 初始化参数
        # nn.init.lecun_normal_(self.mu)
        # nn.init.ones_(self.scale)

    def compute_capacity(
        self,
        num_tokens: int,
        ceil_or_round: str = "ceil"
    ) -> int:
        """修正后的容量计算"""
        # 确保必要的参数存在
        # assert hasattr(self, 'capacity_factor'), "需要定义capacity_factor"
        # assert hasattr(self, 'multiple_of'), "需要定义multiple_of"

        # 应用capacity_factor
        scaled_tokens = num_tokens * self.capacity_factor
        
        # 计算基础容量
        if ceil_or_round == "ceil":
            capacity = math.ceil(scaled_tokens / self.num_experts)
        elif ceil_or_round == "round":
            capacity = round(scaled_tokens / self.num_experts)
        else:
            raise ValueError(f"Invalid ceil_or_round: {ceil_or_round}")

        # 确保最小容量
        capacity = max(capacity, 1)

        # 对齐到multiple_of
        if self.multiple_of > 1:
            remainder = capacity % self.multiple_of
            if remainder != 0:
                capacity += self.multiple_of - remainder

        # 输出警告（可选）
        actual_factor = capacity * self.num_experts / num_tokens
        if abs(actual_factor - self.capacity_factor) > 1e-6:
            print(f"Warning: Target cap_factor {self.capacity_factor}, Actual {actual_factor:.2f}")

        return capacity
    def cosine_psim(self, x, contract_dims, batch_dims, eps=1e-9):
        """PyTorch版本的cosine_psim函数"""
        # 沿contract_dims归一化
        norm = torch.rsqrt((x**2).sum(dim=contract_dims, keepdim=True) + eps)
        x_norm = x * norm
        
        # 重新排列维度以进行批处理矩阵乘法
        all_dims = list(range(x_norm.ndim))
        keep_dims = [d for d in all_dims if d not in contract_dims]
        # 之前的
        # x_flat = x_norm.permute(*keep_dims, *contract_dims).flatten(start_dim=len(keep_dims))
        # 修改的
        x_flat = x_norm.reshape(*x_norm.shape[:len(keep_dims)], -1)
        # 计算批处理点积
        similarity = torch.einsum('...i,...j->...ij', x_flat, x_flat)
        return similarity
    def _compute_max_slots(self, max_seq_len, num_experts):
        """预计算最大可能的slot数量"""
        slots = math.ceil(max_seq_len / num_experts)
        return slots + (-slots % self.multiple_of)
    def get_metrics(self, combine_weights, dispatch_weights, mu):
        B, N, E, S = combine_weights.shape
        device = combine_weights.device
        metrics = {}

        if self.compute_similarity_metrics:
            # ===== Combine Weights相似度 =====
            with torch.autocast(device_type='cuda', dtype=self.precision):  # 混合精度支持
                # 计算相似度矩阵 [B, N, N]
                cw_sim = self.cosine_psim(
                    combine_weights,
                    contract_dims=(2, 3),  # 对应E,S维度
                    batch_dims=(0,)        # B维度作为批处理
                )
                
                sum_total = cw_sim.sum()
                sum_diag = torch.diagonal(cw_sim, dim1=1, dim2=2).sum()
                metrics['combine_weights_similarity_mean'] = (sum_total - sum_diag) / (B * N * (N - 1))

            # ===== Dispatch Weights相似度 =====
            with torch.autocast(device_type='cuda', dtype=self.precision):
                # 重新排列维度为[B, E, S, N]
                dw_permuted = dispatch_weights.permute(0, 2, 3, 1)
                
                # 计算相似度矩阵 [B, E*S, E*S]
                dw_sim = self.cosine_psim(
                    dw_permuted,
                    contract_dims=(3,),    # N维度
                    batch_dims=(0,)        # B维度
                )
                
                # 创建对角线掩码
                eye = torch.eye(E*S, device=device).expand(B, -1, -1)
                
                # 计算均值（排除对角线）
                sum_total = dw_sim.sum()
                sum_diag = torch.diagonal(dw_sim, dim1=1, dim2=2).sum()
                metrics['dispatch_weights_similarity_mean'] = (sum_total - sum_diag) / (B * E * S * (E*S - 1))

        return metrics

    def forward(self, x, mask=None):
        cls_token =x[:, 0, :]
        cls_emd = self.cls_experts(cls_token)
        cls_emd = cls_emd.unsqueeze(1)
        print(f"cls_emd.shape:{cls_emd.shape}")
        x = x[:, 1:, :]
        B, N, D = x.shape
        print(f"输入到softmoe的：{x.shape}")
        E = self.num_experts
        S = self.compute_capacity(N)  # 动态计算slot数量
        print(f"计算的槽位数：{S}")

        mu = self.mu[:, :, :S]  # [D, E, S]  # [1, D, E, 1]

        x_norm = torch.nn.functional.normalize(x, p=2, dim=-1)
        mu_norm = torch.nn.functional.normalize(mu, p=2,dim=0)
        
        if x.numel() < mu.numel():
            x_norm = x_norm * self.scale
        else:
            mu_norm = mu_norm * self.scale
        logits = torch.einsum('bnd,des->bnes', x_norm, mu_norm)
        # logits = einsum('bnd,denp->bnes', x_norm, mu_norm)              
        # print(f"logits:{logits.shape}")
        # 添加噪声
        if self.training and self.noise_std > 0:
            logits += torch.randn_like(logits) * self.noise_std
        def stable_softmax(logits, dim):
            logits = logits - logits.amax(dim=dim, keepdim=True).detach()
            return torch.softmax(logits, dim=dim)
        # Softmax维度调整
        dispatch_weights = stable_softmax(logits, dim=1)
        # dispatch_weights = F.softmax(logits, dim=1)
        global_max = logits.max().item()   # 全局最大值
        global_min = logits.min().item()   # 全局最小值
        global_mean = logits.mean().item() # 全局平均值

        print(f"Global max: {global_max}")
        print(f"Global min: {global_min}")
        print(f"Global mean: {global_mean}")

        global_max_d = dispatch_weights.max().item()   # 全局最大值
        global_min_d = dispatch_weights.min().item()   # 全局最小值
        global_mean_d = dispatch_weights.mean().item() # 全局平均值

        print(f"d max: {global_max_d}")
        print(f"d min: {global_min_d}")
        print(f"d mean: {global_mean_d}")
        print(f"B, N, E, S:{B, N, E, S}")
        combine_weights_flat = stable_softmax(logits.view(B, N, E*S), dim=-1)


        combine_weights = combine_weights_flat.view(B, N, E, S)  # [B, N, E, S]
        # print(combine_weights.shape)
        # === Slot输入构造 ===
        slot_inputs = einsum('bnd,bnes->besd', x, dispatch_weights)
        print(f"slot_inputs:{slot_inputs.shape}") # slot_inputs:torch.Size([64, 4, 36, 768])

        # === 专家处理 ===
        slot_outputs = []
        for e in range(E):
            expert_input = slot_inputs[:, e]  # [B, S, D]
            expert_output = self.experts[e](expert_input)
            slot_outputs.append(expert_output)
        slot_outputs = torch.stack(slot_outputs, dim=1)  # [B, E, S, D]
        # === 输出聚合 ===
        slot_outputs_flat = rearrange(slot_outputs, 'b e s d -> b (e s) d')
        img_output = einsum('bsd,bns->bnd', slot_outputs_flat, combine_weights_flat)

        outputs = torch.cat([cls_emd, img_output], dim=1)
        # outputs.register_hook(lambda grad: torch.clamp(grad, -self.grad_clip_val, self.grad_clip_val))
        if self.training:  # 仅在训练阶段注册梯度钩子
            outputs.register_hook(lambda grad: torch.clamp(grad, -self.grad_clip_val, self.grad_clip_val))
        # 指标收集
        if self.compute_metrics:
            metrics = self.get_metrics(combine_weights, dispatch_weights, mu_norm)
      
  
        return outputs, metrics


        
# 分组，遮挡区域分组,注意FFN隐藏维度 是之前的1.5倍，是6*，而不是4*
class SoftMoE11(nn.Module):
    def __init__(self, dim, num_experts=8, capacity_factor=1.0, noise_std=0.0, dropout_rate=0.1,num_slots=None,
                 deterministic=False, compute_metrics=True, multiple_of=4,grad_clip_val=1.0,max_seq_len=256,precision=torch.float32,
                num_clean_experts=2,   # 新增：专注处理干净区域的专家数
                 num_occlusion_experts=2,  # 新增：专注处理遮挡的专家数
                 clean_threshold=0.7,   # 高置信阈值
                 occlusion_threshold=0.3,  # 遮挡阈值
):
        super().__init__()
        self.dim = dim
        self.num_experts = num_experts
        self.capacity_factor = capacity_factor
        self.noise_std = noise_std
        self.deterministic = deterministic
        self.compute_metrics = compute_metrics
        self.num_slots = num_slots
        self.multiple_of = multiple_of
        self.grad_clip_val = grad_clip_val
        # 新增参数预定义最大序列长度
        self.max_seq_len = max_seq_len
        self.multiple_of = 4
        self.alpha = 0.4  # 遮挡区域缩放下限
        self.beta = 1.2   # 重要区域缩放上限
        self.compute_similarity_metrics = True
        self.precision = precision
        self.num_clean = num_clean_experts
        self.num_occlusion = num_occlusion_experts
        self.num_general = num_experts - num_clean_experts - num_occlusion_experts
        
        # 定义专家索引范围
        self.clean_idx = slice(0, num_clean_experts)  # 前N个为Clean Experts
        self.occlusion_idx = slice(-num_occlusion_experts, None)  # 最后M个为Occlusion
        self.general_idx = slice(num_clean_experts, -num_occlusion_experts)
        
        # 阈值参数
        self.clean_thresh = clean_threshold
        self.occlusion_thresh = occlusion_threshold
        # === 新增参数校验 ===
        assert num_clean_experts + num_occlusion_experts <= num_experts, "专家总数不足"
        
        
        # 定义专家索引列表
        self.clean_experts = list(range(num_clean_experts))  # [0,1]
        self.occlusion_experts = list(range(num_experts - num_occlusion_experts, num_experts))  # [6,7]（假设num_experts=8）
        
        # 阈值参数
        self.clean_thresh = clean_threshold
        self.occlusion_thresh = occlusion_threshold        
        # 输入归一化层
        # self.norm = nn.LayerNorm(dim)
        
        # 可学习参数（对应论文中的mu和scale）
        fan_in = dim
        std = math.sqrt(1.0 / fan_in)
        max_slots = self._compute_max_slots(max_seq_len, num_experts)
        # self.mu = nn.Parameter(torch.empty(dim, num_experts, 1))  # 初始shape需动态调整
        self.mu = nn.Parameter(torch.empty(dim, num_experts, max_slots))
        # print(f"self.mu:{self.mu.shape}")
        nn.init.normal_(self.mu, mean=0, std=1/math.sqrt(dim))
        # nn.init.normal_(self.mu, mean=0, std=std)  # 手动实现lecun_normal
        self.scale = nn.Parameter(torch.ones(1))
        self.threshold=0.4
        
        # 专家网络
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, 6*dim//self.num_experts),
                nn.GELU(),
                nn.Dropout(dropout_rate),
                nn.Linear(6*dim//self.num_experts, dim),
                nn.Dropout(dropout_rate),)
            for _ in range(num_experts)])
        for expert in self.experts:
            nn.init.kaiming_normal_(expert[0].weight, nonlinearity='relu')
            nn.init.zeros_(expert[0].bias)
            nn.init.xavier_normal_(expert[3].weight)
            nn.init.zeros_(expert[3].bias)
        # 初始化参数
        # nn.init.lecun_normal_(self.mu)
        # nn.init.ones_(self.scale)

    # def compute_capacity(self, num_tokens: int) -> int:
    #     """确保 num_experts * num_slots >= num_tokens"""
    #     num_slots = math.ceil(num_tokens / self.num_experts)
    #     num_slots = num_slots + (-num_slots) % self.multiple_of
    #     return max(num_slots, 1)

    # def compute_capacity(self, num_tokens: int) -> int:
    #     num_slots = (num_tokens + self.num_experts - 1) // self.num_experts
    #     remainder = num_slots % self.multiple_of
    #     return num_slots + (self.multiple_of - remainder) % self.multiple_of
    def compute_capacity(
        self,
        num_tokens: int,
        ceil_or_round: str = "ceil"
    ) -> int:
        """修正后的容量计算"""
        scaled_tokens = num_tokens * self.capacity_factor
        
        # 计算基础容量
        if ceil_or_round == "ceil":
            capacity = math.ceil(scaled_tokens / self.num_experts)
        elif ceil_or_round == "round":
            capacity = round(scaled_tokens / self.num_experts)
        else:
            raise ValueError(f"Invalid ceil_or_round: {ceil_or_round}")

        # 确保最小容量
        capacity = max(capacity, 1)

        # 对齐到multiple_of
        if self.multiple_of > 1:
            remainder = capacity % self.multiple_of
            if remainder != 0:
                capacity += self.multiple_of - remainder

        # 输出警告（可选）
        actual_factor = capacity * self.num_experts / num_tokens
        if abs(actual_factor - self.capacity_factor) > 1e-6:
            print(f"Warning: Target cap_factor {self.capacity_factor}, Actual {actual_factor:.2f}")

        return capacity

    def cosine_psim(self, x, contract_dims, batch_dims, eps=1e-9):
        """PyTorch版本的cosine_psim函数"""
        # 沿contract_dims归一化
        norm = torch.rsqrt((x**2).sum(dim=contract_dims, keepdim=True) + eps)
        x_norm = x * norm
        
        # 重新排列维度以进行批处理矩阵乘法
        all_dims = list(range(x_norm.ndim))
        keep_dims = [d for d in all_dims if d not in contract_dims]
        x_flat = x_norm.permute(*keep_dims, *contract_dims).flatten(start_dim=len(keep_dims))
        
        # 计算批处理点积
        similarity = torch.einsum('...i,...j->...ij', x_flat, x_flat)
        return similarity
    def _compute_max_slots(self, max_seq_len, num_experts):
        """预计算最大可能的slot数量"""
        slots = math.ceil(max_seq_len / num_experts)
        return slots + (-slots % self.multiple_of)
    def get_metrics(self, combine_weights, dispatch_weights, mu):
        B, N, E, S = combine_weights.shape
        device = combine_weights.device
        metrics = {}

        if self.compute_similarity_metrics:
            # ===== Combine Weights相似度 =====
            with torch.autocast(device_type='cuda', dtype=self.precision):  # 混合精度支持
                # 计算相似度矩阵 [B, N, N]
                cw_sim = self.cosine_psim(
                    combine_weights,
                    contract_dims=(2, 3),  # 对应E,S维度
                    batch_dims=(0,)        # B维度作为批处理
                )
                
                # 创建对角线掩码
                eye = torch.eye(N, device=device).expand(B, -1, -1)
                
                # 计算均值（排除对角线）
                sum_total = cw_sim.sum()
                sum_diag = torch.diagonal(cw_sim, dim1=1, dim2=2).sum()
                metrics['combine_weights_similarity_mean'] = (sum_total - sum_diag) / (B * N * (N - 1))

            # ===== Dispatch Weights相似度 =====
            with torch.autocast(device_type='cuda', dtype=self.precision):
                # 重新排列维度为[B, E, S, N]
                dw_permuted = dispatch_weights.permute(0, 2, 3, 1)
                
                # 计算相似度矩阵 [B, E*S, E*S]
                dw_sim = self.cosine_psim(
                    dw_permuted,
                    contract_dims=(3,),    # N维度
                    batch_dims=(0,)        # B维度
                )
                
                # 创建对角线掩码
                eye = torch.eye(E*S, device=device).expand(B, -1, -1)
                
                # 计算均值（排除对角线）
                sum_total = dw_sim.sum()
                sum_diag = torch.diagonal(dw_sim, dim1=1, dim2=2).sum()
                metrics['dispatch_weights_similarity_mean'] = (sum_total - sum_diag) / (B * E * S * (E*S - 1))

        return metrics
    def forward(self, x,attn_weight, mode='original'):
        B, N, D = x.shape
        print(mode)
        if attn_weight is None:
            print("传入softmoe的attn_weight为None")
        else:
            print("attn_weight.max:")
            print(attn_weight.max().item())
            print("attn_weight.min:")
            print(attn_weight.min().item())
            print("attn_weight.mean:")
            print(attn_weight.mean().item())
        # attn_weight_full = torch.cat([torch.ones(B, 1).to(device), attn_weight], dim=1)  
        if mode !='original':
            attn_weight_full = torch.cat([torch.ones(B, 1).to(attn_weight.device), attn_weight], dim=1) 
            
                    # === 构建区域掩码 ===
            # 高置信区域掩码 (CLS token自动包含在内)
            mask_high = (attn_weight_full > self.clean_thresh).float().unsqueeze(-1).unsqueeze(-1)  # [B,129,1,1]
            # 遮挡区域掩码
            mask_low = (attn_weight_full <= self.occlusion_thresh).float().unsqueeze(-1).unsqueeze(-1)
            # 中间区域掩码
            mask_mid = 1.0 - mask_high - mask_low

        #     scaling_factor = torch.where(
        #     attn_weight_full < self.threshold,  # 条件：低于阈值
        #     attn_weight_full,              # 满足条件时：缩放系数 = attn_weight_full
        #     torch.ones_like(attn_weight_full)  # 不满足条件时：缩放系数 = 1.0
        # )
            # attn_weight_scaled = self.alpha + (self.beta - self.alpha) * attn_weight_full
            # attn_weight_expanded = attn_weight_scaled.unsqueeze(-1).unsqueeze(-1)
            # attn_weight_expanded = scaling_factor.unsqueeze(-1).unsqueeze(-1)
            # attn_weight_expanded = attn_weight_full.unsqueeze(-1).unsqueeze(-1)
            
            # print(f"attn_weight_expanded.max:{attn_weight_expanded.max().item()}")
            # print(f"attn_weight_expanded.min:{attn_weight_expanded.min().item()}")
            # print(f"attn_weight_expanded.mean:{attn_weight_expanded.mean().item()}")
        E = self.num_experts
        if self.num_slots is None:
            S = self.compute_capacity(N)  # 动态计算slot数量
        else:
            S = self.num_slots
        print(f"计算的槽位数：{S}")
        # print(S)
        # 更新mu参数形状
        # self.mu.data = self.mu.data.expand(D, E, S).contiguous()
        mu = self.mu[:, :, :S]  # [D, E, S]  # [1, D, E, 1]
        # print(f"mu:{mu.shape}") # mu:torch.Size([768, 4, 36])
        # mu = mu.unsqueeze(0).expand(B, -1, -1, -1)  # [B, D, E, S]
        
        # === 输入归一化 ===
        # x = self.norm(x)
        # logits = torch.einsum('bnd,des->bnes', x * self.scale, self.mu[:, :, :S])
        x_norm = torch.nn.functional.normalize(x, p=2, dim=-1)
        mu_norm = torch.nn.functional.normalize(mu, p=2,dim=0)
        
        # print(f"x_norm:{x_norm.shape}") # x_norm:torch.Size([64, 129, 768])
        # print(f"mu_norm:{mu_norm.shape}") # mu_norm:torch.Size([768, 4, 36])
        # === 分发权重计算 ===
        # 计算logits [B, N, E, S]
        # logits = einsum('bnd,bdes->bnes', x_norm * self.scale, mu_norm)
        
        if x.numel() < mu.numel():
            x_norm = x_norm * self.scale
        else:
            mu_norm = mu_norm * self.scale
        logits = torch.einsum('bnd,des->bnes', x_norm, mu_norm)
        if mode !='original':
            # 克隆logits用于不同区域
            logits_high = logits.clone()
            logits_low = logits.clone()
            logits_mid = logits.clone()
            # --- 高置信区域约束到Clean Experts ---
            # 生成Clean专家掩码（布尔矩阵）
            clean_mask = torch.zeros(self.num_experts, dtype=torch.bool, device=x.device)
            clean_mask[self.clean_experts] = True
            logits_high[:, :, ~clean_mask, :] = -1e4
            logits_high *= mask_high
            # --- 遮挡区域约束到Occlusion Experts ---
            occlusion_mask = torch.zeros(self.num_experts, dtype=torch.bool, device=x.device)
            occlusion_mask[self.occlusion_experts] = True
            logits_low[:, :, ~occlusion_mask, :] = -1e4
            logits_low *= mask_low
            
            # --- 中间区域自由路由 ---
            logits_mid *= mask_mid
            
            # 合并logits
            logits = logits_high + logits_low + logits_mid

        # logits = einsum('bnd,denp->bnes', x_norm, mu_norm)              
        # print(f"logits:{logits.shape}")
        # 添加噪声
        if self.training and self.noise_std > 0:
            logits += torch.randn_like(logits) * self.noise_std
        def stable_softmax(logits, dim):
            logits = logits - logits.amax(dim=dim, keepdim=True).detach()
            return torch.softmax(logits, dim=dim)
        # Softmax维度调整
        dispatch_weights = stable_softmax(logits, dim=1)
        print("修改前：---------")
        global_max_d = dispatch_weights.max().item()   # 全局最大值
        global_min_d = dispatch_weights.min().item()   # 全局最小值
        global_mean_d = dispatch_weights.mean().item() # 全局平均值

        print(f"d max: {global_max_d}")
        print(f"d min: {global_min_d}")
        print(f"d mean: {global_mean_d}")
        # if mode !='original':
        #     dispatch_weights_weighted = dispatch_weights * attn_weight_expanded
        #     dispatch_weights = dispatch_weights_weighted

        #     print("修改后：---------")
        #     print(f"d max: {dispatch_weights.max().item()}")
        #     print(f"d min: {dispatch_weights.min().item()}")
        #     print(f"d mean: {dispatch_weights.mean().item()}")

        # dispatch_weights = F.softmax(logits, dim=1)
        # print(f"logits.shape:{logits.shape}") # logits.shape:torch.Size([64, 129, 4, 36])
        print(f"B, N, E, S:{B, N, E, S}")
        combine_weights_flat = stable_softmax(logits.view(B, N, E*S), dim=-1)
        combine_weights = combine_weights_flat.view(B, N, E, S)
        # combine_weights_flat = F.softmax(logits.view(B, N, E*S), dim=-1)
        # combine_weights_flat = F.softmax(logits.flatten(2), dim=-1).view_as(logits)


        # combine_weights = combine_weights_flat.view(B, N, E, S)  # [B, N, E, S]
        # print(combine_weights.shape)
        # === Slot输入构造 ===
        slot_inputs = einsum('bnd,bnes->besd', x, dispatch_weights)
        print(f"slot_inputs:{slot_inputs.shape}") # slot_inputs:torch.Size([64, 4, 36, 768])

        # === 专家处理 ===
        slot_outputs = []
        for e in range(E):
            expert_input = slot_inputs[:, e]  # [B, S, D]
            expert_output = self.experts[e](expert_input)
            slot_outputs.append(expert_output)
        slot_outputs = torch.stack(slot_outputs, dim=1)  # [B, E, S, D]
        
        # 前向计算
        # slot_inputs = rearrange(slot_inputs, 'b e s d -> (b e s) d')
        # slot_outputs = torch.vmap(lambda expert, x: expert(x))(self.experts, slot_inputs)
        # slot_outputs = rearrange(slot_outputs, '(b e s) d -> b e s d', b=B, e=E)
        # === 输出聚合 ===
        slot_outputs_flat = rearrange(slot_outputs, 'b e s d -> b (e s) d')
        outputs = einsum('bsd,bns->bnd', slot_outputs_flat, combine_weights_flat)
        # outputs.register_hook(lambda grad: torch.clamp(grad, -self.grad_clip_val, self.grad_clip_val))
        if self.training:  # 仅在训练阶段注册梯度钩子
            outputs.register_hook(lambda grad: torch.clamp(grad, -self.grad_clip_val, self.grad_clip_val))
        # 指标收集
        if self.compute_metrics:
            metrics = self.get_metrics(combine_weights, dispatch_weights, self.mu)
        # else:
        #     metrics = {}        
        # metrics = {'auxiliary_loss': torch.tensor(0.0)}   
        return outputs, metrics        