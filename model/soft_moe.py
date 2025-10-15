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

class Mlp(nn.Module):
    def __init__(self, mlp_dim, hidden_features=3072, out_features=None, act_layer=nn.GELU, dropout_rate=0.):
        super().__init__()
        out_features = out_features or mlp_dim
        hidden_features = hidden_features or mlp_dim
        self.fc1 = nn.Linear(mlp_dim, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x    

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

# experts

class Experts(nn.Module):
    def __init__(
        self,
        experts,
        is_distributed = None,
        offload_unused_experts_to_cpu = True
    ):
        super().__init__()
        self.num_experts = len(experts)
        self.experts = nn.ModuleList(experts)

        self.is_distributed = is_distributed
        if not exists(self.is_distributed):
            self.is_distributed = dist.is_initialized() and dist.get_world_size() > 1

        # whether to offload unused experts to cpu, will require optimizer handles conversion of gradients to right device when accumulating
        self.offload_unused_experts_to_cpu = offload_unused_experts_to_cpu

        self.all_gather = AllGather()
        self.register_buffer('dummy', torch.ones(1), persistent = False)

    @property
    def device(self):
        return self.dummy.device

    def all_experts_to_cpu_besides(self, selection):
        if not self.offload_unused_experts_to_cpu:
            return

        if isinstance(selection, int):
            experts = [self.experts[selection]]
        if isinstance(selection, slice):
            experts = self.experts[selection]
        else:
            experts = selection

        experts_set = set(experts)

        for expert in self.experts:
            device = self.device if expert in experts_set else 'cpu'
            expert.to(device)

    def forward(
        self,
        x,
        is_distributed = None
    ):
        """
        einops notation:
        b - batch
        r - rank (device / machines)
        e - experts
        n - sequence (number of tokens per expert)
        d - feature dimension
        """

        is_distributed = default(is_distributed, self.is_distributed)
        shape, num_experts = x.shape, self.num_experts

        # for now naively all gather across batch dimension if distributed, optimize later

        if is_distributed:
            seq_sizes = gather_sizes(x, dim = -2)
            assert has_only_one_value(seq_sizes), 'number of tokens per expert must be the same'

            x, batch_sizes = self.all_gather(x)
            total_batch_size = x.shape[0]

            world_size = dist.get_world_size()
            rank = dist.get_rank()
        else:
            world_size = 1
            rank = 0

        # the experts in use on the rank

        if is_distributed:
            if world_size <= num_experts:
                num_experts_across_ranks = chunk_num(num_experts, world_size)
                start_indices = cumsum_exclusive(torch.tensor(num_experts_across_ranks), dim = -1)

                num_experts_per_rank = num_experts_across_ranks[rank]
                num_experts_batches_across_ranks = tuple(i * total_batch_size for i in num_experts_across_ranks)

                expert_start_index = start_indices[rank].item()
            else:
                num_batch_chunks = world_size // num_experts
                total_ranks_in_use = num_batch_chunks * num_experts

                expert_start_index = rank // num_batch_chunks

                batch_splits = chunk_num(total_batch_size, num_batch_chunks)
                num_experts_batches_across_ranks = batch_splits * num_experts

                # for now, remaining machines just process nothing

                remain_ranks = world_size % num_experts
                num_experts_batches_across_ranks += (0,) * remain_ranks

                num_experts_per_rank = int(rank < total_ranks_in_use)

            assert len(num_experts_batches_across_ranks) == world_size

            expert_slice = slice(expert_start_index, expert_start_index + num_experts_per_rank)
        else:
            num_experts_per_rank = num_experts
            expert_slice = slice(0, num_experts)

        # if distributed, each machine only handles subset of experts and batch

        x = rearrange(x, 'b e n d -> e b n d')

        if is_distributed:
            x, expert_batch_packed_shape = pack_one(x, '* n d')
            x = x.split(num_experts_batches_across_ranks, dim = 0)
            x = split_by_rank(x)

            if num_experts_per_rank > 0:
                x = rearrange(x, '(e b) n d -> e b n d', e = num_experts_per_rank)
            else:
                x = x.reshape(num_experts, *x.shape)

        # get the experts in use

        self.all_experts_to_cpu_besides(expert_slice)

        experts = self.experts[expert_slice]

        # route tokens to appropriate experts

        outs = []
        for expert, expert_input in zip(experts, x):
            out = expert(expert_input)
            outs.append(out)

        if len(outs) > 0:
            outs = torch.stack(outs)
        else:
            outs = torch.empty_like(x).requires_grad_()

        # all gather across merged expert batches dimensions
        # then split the batch dimension back

        if is_distributed:
            outs = rearrange(outs, 'e b n d -> (e b) n d')
            outs, _ = self.all_gather(outs)
            outs = unpack_one(outs, expert_batch_packed_shape, '* n d')

        outs = rearrange(outs, 'e b n d -> b e n d')

        if is_distributed:
            outs = outs.split(batch_sizes.tolist())
            outs = split_by_rank(outs)

        assert outs.shape == shape
        return outs

# main class

class SoftMoE(Module):
    def __init__(
        self,
        dim,
        *,
        seq_len = 129,
        num_experts = 3,
        num_slots = 129,
        expert_mult = 4,
        dropout = 0.,
        geglu = False,
        is_distributed = None,
        offload_unused_experts_to_cpu = True,
        use_layernorm = False
    ):
        super().__init__()
        assert exists(seq_len) ^ exists(num_slots), 'either seq_len, or num_slots must be passed into SoftMoE'

        if exists(seq_len):
            num_slots = default(num_slots, seq_len // num_experts)
        elif exists(num_slots):
            seq_len = num_slots * num_experts

        norm_klass = LayerNorm if use_layernorm else RMSNorm
        self.norm = norm_klass(dim)

        self.slot_norm = norm_klass(dim)
        self.slot_embeds = nn.Parameter(torch.randn(num_experts, num_slots, dim))
  
        expert_klass = GLUFeedForward if geglu else FeedForward

        self.experts = Experts(
            experts = [expert_klass(dim = dim, num_experts=num_experts, dropout = dropout) for _ in range(num_experts)],
            is_distributed = is_distributed,
            offload_unused_experts_to_cpu = offload_unused_experts_to_cpu
        )

    def forward(self, x, mask = None, add_noise = False, noise_mult = 1.):
        """
        einstein notation
        b - batch
        n - sequence length
        e - number of experts
        s - number of slots per expert
        d - feature dimension
        """

        is_single_token = x.ndim == 2
        is_image = x.ndim == 4

        if is_image:
            x = rearrange(x, 'b d h w -> b h w d')
            x, ps = pack([x], 'b * d')
        elif is_single_token:
            x = rearrange(x, 'b d -> b 1 d')

        # following Algorithm 1, with the normalization they proposed, but with scaling of both (the now popular rmsnorm + gamma)

        x = self.norm(x)
        slot_embeds = self.slot_norm(self.slot_embeds)

        logits = einsum('b n d, e s d -> b n e s', x, slot_embeds)

        # noised dispatch and combine gate logits, with annealing if needed

        if add_noise:
            noise = gumbel_noise(logits) * noise_mult
            logits = logits + noise

        # account for key padding mask

        if exists(mask):
            mask = rearrange(mask, 'b n -> b n 1 1')
            logits = logits.masked_fill(~mask, -torch.finfo(logits.dtype).max)

        # get dispatch and combine weights (softmax across right dimensions)

        dispatch_weights = logits.softmax(dim = 1)

        combine_weights = rearrange(logits, 'b n e s -> b n (e s)')
        combine_weights = combine_weights.softmax(dim = -1)

        # derive slots by weighted average of input tokens using the dispatch weights from above

        slots = einsum('b n d, b n e s -> b e s d', x, dispatch_weights)

        # route the slots per expert to each expert

        out = self.experts(slots)

        # combine back out

        out = rearrange(out, ' b e s d -> b (e s) d')
        out = einsum('b s d, b n s -> b n d', out, combine_weights)

        if is_image:
            out, = unpack(out, ps, 'b * d')
            out = rearrange(out, 'b h w d -> b d h w')
        elif is_single_token:
            out = rearrange(out, 'b 1 d -> b d')

        return out
    

class SharedExperts(nn.Module):
    def __init__(self, dim, num_experts, mult=4, dropout=0., geglu=False):
        super().__init__()
        self.num_experts = num_experts
        self.geglu = geglu
        
        # 共享参数矩阵（与单MLP参数量相同）
        hidden_dim = int(dim * mult)
        if geglu:
            hidden_dim = int(2 * hidden_dim / 3)  # GEGLU调整
            
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        
        self.dropout = nn.Dropout(dropout)
        self.gelu = nn.GELU()
        
    def forward(self, slots):
        """
        输入：slots [batch, num_experts, num_slots, dim]
        输出：processed_slots [batch, num_experts, num_slots, dim]
        """
        batch, num_experts, num_slots, dim = slots.shape
        
        # 合并专家维度
        slots = rearrange(slots, 'b e s d -> (b e) s d')
        
        # 第一阶段：所有专家共享W1
        h = self.w1(slots)  # [(b e) s h]
        
        if self.geglu:
            # GEGLU处理
            h, gate = h.chunk(2, dim=-1)
            h = h * self.gelu(gate)
        else:
            h = self.gelu(h)
        
        h = self.dropout(h)
        
        # 第二阶段：所有专家共享W2
        out = self.w2(h)  # [(b e) s d]
        
        # 恢复专家维度
        return rearrange(out, '(b e) s d -> b e s d', e=num_experts)
class SoftMoE2(nn.Module):
    def __init__(
        self,
        dim,
        *,
        seq_len=None,
        num_experts=4,
        num_slots=None,
        expert_mult=4,
        dropout=0.,
        geglu=False,
        use_layernorm=False
    ):
        super().__init__()
        assert exists(seq_len) ^ exists(num_slots), "需指定seq_len或num_slots"

        if seq_len is not None:
            num_slots = num_slots or seq_len // num_experts
        else:
            seq_len = num_slots * num_experts

        norm_klass = nn.LayerNorm if use_layernorm else RMSNorm
        self.norm = norm_klass(dim)
        self.slot_norm = norm_klass(dim)
        
        # 可学习的槽位嵌入（自动适配专家数量）
        self.slot_embeds = nn.Parameter(torch.randn(num_experts, num_slots, dim))
        
        # 共享参数的专家模块（总参数量=单MLP）
        self.experts = SharedExperts(
            dim=dim,
            num_experts=num_experts,
            mult=expert_mult,
            dropout=dropout,
            geglu=geglu
        )

    def forward(self, x, mask=None):
        # 输入处理 [batch, seq_len, dim]
        x = self.norm(x)
        slot_embeds = self.slot_norm(self.slot_embeds)
        
        # 计算路由权重 [batch, seq, experts, slots]
        logits = einsum('b s d, e t d -> b s e t', x, slot_embeds)
        
        if mask is not None:
            logits = logits.masked_fill(~mask.unsqueeze(-1).unsqueeze(-1), -torch.inf)
        
        # 分配权重 [batch, seq, experts, slots]
        dispatch_weights = logits.softmax(dim=1)
        
        # 生成槽位 [batch, experts, slots, dim]
        slots = einsum('b s d, b s e t -> b e t d', x, dispatch_weights)
        
        # 专家处理（参数共享）
        processed_slots = self.experts(slots)
        
        # 合并结果 [batch, seq, dim]
        combine_weights = rearrange(logits, 'b s e t -> b s (e t)').softmax(dim=-1)
        out = einsum('b e t d, b s t -> b s d', 
                    rearrange(processed_slots, 'b e t d -> b (e t) d'),
                    combine_weights)
        
        return out

# class SoftMoE3(Module):
#     def __init__(self, dim, seq_len, num_experts,
#         num_slots = 43,
#         expert_mult = 4,
#         dropout = 0.,
#         geglu = False,
#         is_distributed = None,
#         offload_unused_experts_to_cpu = True,
#         use_layernorm = False):
#         super().__init__()
#         self.seq_len = seq_len
#         self.num_experts = num_experts
#         self.norm = nn.LayerNorm(dim)
        
#         # Dispatch和Combine的独立线性层
#         self.dispatch_linear = nn.Linear(dim, num_experts * seq_len)
#         self.combine_linear = nn.Linear(dim, num_experts * seq_len)
        
#         # 专家MLP（总参数量与原MLP一致）
#         hidden_dim_total = 4 * dim
#         hidden_dim_per_expert = hidden_dim_total // num_experts
#         self.experts = nn.ModuleList([
#             nn.Sequential(
#                 nn.Linear(dim, hidden_dim_per_expert),
#                 nn.GELU(),
#                 nn.Linear(hidden_dim_per_expert, dim)
#             ) for _ in range(num_experts)
#         ])
    
#     def forward(self, x):
#         B, N, D = x.shape
#         E, S = self.num_experts, self.seq_len
        
#         # 1. Dispatch权重生成
#         dispatch_logits = self.dispatch_linear(x)  # [B, N, E*S]
#         dispatch_weights = torch.softmax(dispatch_logits.view(B, N, E, S), dim=1)
        
#         # 2. 计算Slot输入
#         slot_inputs = torch.einsum('b n d, b n e s -> b e s d', x, dispatch_weights)
        
#         # 3. 专家处理
#         slot_outputs = []
#         for i, expert in enumerate(self.experts):
#             slot_outputs.append(expert(slot_inputs[:, i]))  # [B, S, D]
#         slot_outputs = torch.stack(slot_outputs, dim=1)      # [B, E, S, D]
        
#         # 4. Combine权重生成
#         combine_logits = self.combine_linear(x)             # [B, N, E*S]
#         combine_weights = torch.softmax(combine_logits.view(B, N, E, S), dim=-1)
        
#         # 5. 组合输出
#         output = torch.einsum('b e s d, b n e s -> b n d', slot_outputs, combine_weights)
#         return output

class SoftMoE3(Module):
    def __init__(self, dim, num_experts, num_slots, geglu=False):
        super().__init__()
        self.num_experts = num_experts
        self.num_slots = num_slots

        self.norm = nn.LayerNorm(dim)
        self.proj = nn.Linear(dim, dim)  # 输入投影
        self.slot_keys = nn.Parameter(torch.randn(num_experts, num_slots, dim))
        self.query_proj = nn.Linear(dim, dim)  # 查询投影

        # 每个专家独立定义（隐藏层不分割）
        expert_klass = GEGLU if geglu else FeedForward
        self.experts = nn.ModuleList([expert_klass(dim, 4*dim) for _ in range(num_experts)])

    def forward(self, x, mask=None):
        B, N, D = x.shape
        E, S = self.num_experts, self.num_slots

        x = self.norm(x)
        x_proj = self.proj(x)  # (B, N, D)
        x_query = self.query_proj(x)  # (B, N, D)

        # 分发权重
        logits = einsum('b n d, e s d -> b n e s', x_proj, self.slot_keys)
        if mask is not None:
            logits = logits.masked_fill(~mask[:, :, None, None], -torch.finfo(logits.dtype).max)
        dispatch_weights = rearrange(logits, 'b n e s -> b n (e s)').softmax(dim=-1)
        dispatch_weights = rearrange(dispatch_weights, 'b n (e s) -> b n e s', e=E, s=S)

        # Slot输入构造
        slot_inputs = einsum('b n d, b n e s -> b e s d', x_proj, dispatch_weights)  # (B, E, S, D)

        # 专家处理（关键修正）
        slot_outputs = []
        for e in range(E):
            expert_input = slot_inputs[:, e]  # (B, S, D)
            expert_output = self.experts[e](expert_input)  # (B, S, D)
            slot_outputs.append(expert_output)
        slot_outputs = torch.stack(slot_outputs, dim=1)  # (B, E, S, D)

        slot_outputs_flat = rearrange(slot_outputs, 'b e s d -> b (e s) d')  # (B, E*S, D)

        # 合并权重
        combine_logits = einsum('b n d, b s d -> b n s', x_query, slot_outputs_flat)
        combine_weights = combine_logits.softmax(dim=-1)  # (B, N, E*S)

        # 输出聚合
        output = einsum('b s d, b n s -> b n d', slot_outputs_flat, combine_weights)
        return output


import math
import logging
import torch
from torch import nn, einsum
from einops import rearrange

class SoftMoE4(nn.Module):
    def __init__(self, dim, num_experts, capacity_factor=1.0, noise_std=0.0, 
                 deterministic=False, compute_metrics=True, multiple_of=4):
        super().__init__()
        self.dim = dim
        self.num_experts = num_experts
        self.capacity_factor = capacity_factor
        self.noise_std = noise_std
        self.deterministic = deterministic
        self.compute_metrics = compute_metrics
        self.multiple_of = multiple_of

        # 输入归一化层
        self.norm = nn.LayerNorm(dim)
        
        # 可学习参数（对应论文中的mu和scale）
        fan_in = dim
        std = math.sqrt(1.0 / fan_in)
        self.mu = nn.Parameter(torch.empty(dim, num_experts, 1))  # 初始shape需动态调整
        nn.init.normal_(self.mu, mean=0, std=std)  # 手动实现lecun_normal
        self.scale = nn.Parameter(torch.ones(1))
        self.mu = nn.Parameter(torch.Tensor(dim, num_experts, 1))  # 初始shape需动态调整
        self.scale = nn.Parameter(torch.ones(1))
        
        # 专家网络
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, 4*dim//self.num_experts),
                nn.GELU(),
                nn.Linear(4*dim//self.num_experts, dim))
            for _ in range(num_experts)])

        # 初始化参数
        # nn.init.lecun_normal_(self.mu)
        # nn.init.ones_(self.scale)

    def compute_capacity(self, num_tokens: int) -> int:
        """动态计算每个专家的slot数量"""
        capacity = math.ceil(num_tokens * self.capacity_factor / self.num_experts)
        capacity = max(capacity, 1)
        # 对齐到multiple_of
        capacity = capacity + (-capacity) % self.multiple_of
        actual_cf = capacity * self.num_experts / num_tokens
        if abs(actual_cf - self.capacity_factor) > 1e-4:
            logging.warning(f"Actual capacity factor {actual_cf:.2f} vs target {self.capacity_factor:.2f}")
        return capacity

    def forward(self, x, mask=None):
        B, N, D = x.shape
        E = self.num_experts
        S = self.compute_capacity(N)  # 动态计算slot数量
        
        # 更新mu参数形状
        self.mu.data = self.mu.data.expand(D, E, S).contiguous()
        
        # === 输入归一化 ===
        x = self.norm(x)
        x_norm = torch.nn.functional.normalize(x, p=2, dim=-1)
        
        # === 分发权重计算 ===
        # 计算logits [B, N, E, S]
        logits = einsum('bnd,des->bnes', x_norm * self.scale, 
                       torch.nn.functional.normalize(self.mu, dim=0))
        
        # 添加噪声
        if self.training and self.noise_std > 0:
            logits += torch.randn_like(logits) * self.noise_std
        
        # Softmax维度调整
        dispatch_weights = torch.softmax(logits, dim=1)  # 按输入序列维度归一化
        combine_weights = torch.softmax(logits.view(B, N, E*S), dim=-1)  # 合并维度
        
        # === Slot输入构造 ===
        slot_inputs = einsum('bnd,bnes->besd', x, dispatch_weights)
        
        # === 专家处理 ===
        slot_outputs = []
        for e in range(E):
            expert_input = slot_inputs[:, e]  # [B, S, D]
            expert_output = self.experts[e](expert_input)
            slot_outputs.append(expert_output)
        slot_outputs = torch.stack(slot_outputs, dim=1)  # [B, E, S, D]
        
        # === 输出聚合 ===
        slot_outputs_flat = rearrange(slot_outputs, 'b e s d -> b (e s) d')
        outputs = einsum('bsd,bns->bnd', slot_outputs_flat, combine_weights)
        
        # === 指标收集 ===
        metrics = {}
        if self.compute_metrics:
            metrics.update({
                'auxiliary_loss': torch.tensor(0.0),
                'combine_weights_min': combine_weights.min(),
                'combine_weights_max': combine_weights.max(),
                'dispatch_weights_min': dispatch_weights.min(),
                'dispatch_weights_max': dispatch_weights.max(),
            })
            
            # 计算余弦相似性（简化实现）
            with torch.no_grad():
                cw_sim = torch.cosine_similarity(
                    combine_weights[:, :, None], combine_weights[:, None, :], dim=-1)
                metrics['combine_sim'] = cw_sim.mean()
                
                mu_sim = torch.cosine_similarity(
                    self.mu.view(D, E*S)[:, None], self.mu.view(D, E*S)[None, :], dim=0)
                metrics['mu_sim'] = mu_sim.mean()
        
        return outputs, metrics

# def cosine_psim(x: torch.Tensor, contract_axes: tuple, eps: float = 1e-9) -> torch.Tensor:
#     """优化的余弦相似度计算函数"""
#     # 步骤1: 沿contract_axes归一化
#     norm = torch.linalg.norm(x, dim=contract_axes, keepdim=True)
#     x_norm = x / (norm + eps)
    
#     # 步骤2: 重组张量为二维矩阵
#     original_shape = x_norm.shape
#     preserved_dims = [d for d in range(x_norm.dim()) if d not in contract_axes]
#     flattened_size = torch.prod(torch.tensor([original_shape[d] for d in preserved_dims])) # 保持的维度总大小
    
#     # 展平非收缩维度
#     x_flat = x_norm.permute(*contract_axes, *preserved_dims).contiguous()
#     x_flat = x_flat.view(-1, flattened_size)
    
#     # 步骤3: 计算相似度矩阵
#     similarity = torch.matmul(x_flat, x_flat.T)
#     return similarity
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
    def __init__(self, dim, num_experts, capacity_factor=1.0, noise_std=0.0, 
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
    def forward(self, x, mask=None):
        B, N, D = x.shape
        print(f"输入到softmoe的：{x.shape}")
        E = self.num_experts
        S = self.compute_capacity(N)  # 动态计算slot数量
        # print(S)
        # 更新mu参数形状
        # self.mu.data = self.mu.data.expand(D, E, S).contiguous()
        mu = self.mu[:, :, :S]  # [D, E, S]  # [1, D, E, 1]
        mu = mu.unsqueeze(0).expand(B, -1, -1, -1)  # [B, D, E, S]
        # === 输入归一化 ===
        x = self.norm(x)
        x_norm = torch.nn.functional.normalize(x, p=2, dim=-1)
        mu_norm = torch.nn.functional.normalize(mu, dim=1)
        # === 分发权重计算 ===
        # 计算logits [B, N, E, S]
        logits = einsum('bnd,bdes->bnes', x_norm * self.scale, mu_norm)
                       
        
        # 添加噪声
        if self.training and self.noise_std > 0:
            logits += torch.randn_like(logits) * self.noise_std
        def stable_softmax(logits, dim):
            logits = logits - logits.amax(dim=dim, keepdim=True).detach()
            return torch.softmax(logits, dim=dim)
        # Softmax维度调整
        dispatch_weights = stable_softmax(logits, dim=1)
        print(f"logits.shape:{logits.shape}")
        print(f"B, N, E, S:{B, N, E, S}")
        combine_weights_flat = stable_softmax(logits.view(B, N, E*S), dim=-1)
        combine_weights = combine_weights_flat.view(B, N, E, S)  # [B, N, E, S]
        # print(combine_weights.shape)
        # === Slot输入构造 ===
        slot_inputs = einsum('bnd,bnes->besd', x, dispatch_weights)
        
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
        # 指标收集
        # if self.compute_metrics:
        #     metrics = self.get_metrics(combine_weights, dispatch_weights, self.mu)
        # else:
        #     metrics = {}        
        metrics = {'auxiliary_loss': torch.tensor(0.0)}   
        return outputs, metrics

# 在5的基础上，让slot键应该是可学习的参数
class SoftMoE6(nn.Module):
    def __init__(self, dim, num_experts, capacity_factor=1.0, noise_std=0.0, 
                 compute_metrics=True, multiple_of=4, grad_clip_val=1.0, max_seq_len=256):
        super().__init__()
        self.dim = dim
        self.num_experts = num_experts
        self.capacity_factor = capacity_factor
        self.noise_std = noise_std
        self.compute_metrics = compute_metrics
        self.multiple_of = multiple_of
        self.grad_clip_val = grad_clip_val
        self.max_seq_len = max_seq_len
        self.norm = nn.LayerNorm(dim)
        # ✅ 关键修复：正确初始化slot键
        # 可学习参数（对应论文中的mu和scale）
                # ✅ 关键修复：正确初始化slot键的维度顺序
        max_slots = self._compute_max_slots(max_seq_len, num_experts)
        self.mu = nn.Parameter(torch.empty(num_experts, max_slots, dim))  # [E, S, D]
        nn.init.normal_(self.mu, mean=0, std=1/math.sqrt(dim))
        print(f"self.mu.shape: {self.mu.shape}")  # 应输出 [4,64,768]
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

    def compute_capacity(self, num_tokens: int) -> int:
        """确保 num_experts * num_slots >= num_tokens"""
        num_slots = math.ceil(num_tokens / self.num_experts)
        num_slots = num_slots + (-num_slots) % self.multiple_of
        return max(num_slots, 1)
    
    def _compute_max_slots(self, max_seq_len, num_experts):
        """预计算最大可能的slot数量"""
        slots_per_expert = math.ceil(max_seq_len / num_experts)
        return slots_per_expert + (-slots_per_expert % self.multiple_of)
    
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
        E= self.num_experts
        S = self.compute_capacity(N)
        # ✅ 关键修复：正确使用slot键
        mu = self.mu[:, :S, :]  # [E, S, D]
        mu_norm = torch.nn.functional.normalize(mu, dim=-1)  # 沿特征维度归一化
        
        # 输入归一化
        x = self.norm(x)
        x_norm = torch.nn.functional.normalize(x, p=2, dim=-1)
        
        # ✅ 关键修复：正确的点积计算
        logits = einsum('bnd,esd->bnes', x_norm * self.scale, mu_norm)
        
        
        # 添加噪声
        if self.training and self.noise_std > 0:
            logits += torch.randn_like(logits) * self.noise_std
        def stable_softmax(logits, dim):
            logits = logits - logits.amax(dim=dim, keepdim=True).detach()
            return torch.softmax(logits, dim=dim)
        # Softmax维度调整
        dispatch_weights = stable_softmax(logits, dim=1)
        print(f"logits.shape:{logits.shape}")
        print(f"B, N, E, S:{B, N, E, S}")
        combine_weights_flat = stable_softmax(logits.view(B, N, E*S), dim=-1)
        combine_weights = combine_weights_flat.view(B, N, E, S)  # [B, N, E, S]
        # print(combine_weights.shape)
        # === Slot输入构造 ===
        slot_inputs = einsum('bnd,bnes->besd', x, dispatch_weights)
        
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
        if self.training:
            outputs.register_hook(lambda grad: torch.clamp(grad, -self.grad_clip_val, self.grad_clip_val))
        # 指标收集
        # if self.compute_metrics:
        #     metrics = self.get_metrics(combine_weights, dispatch_weights, self.mu)
        # else:
        #     metrics = {}        
        metrics = {'auxiliary_loss': torch.tensor(0.0)}   
        return outputs, metrics

# import torch
# from torch import nn
  # 假设已将原始代码保存为 soft_moe_pytorch.py

# def test_image_input():
#     batch_size = 64
#     num_patches = 129
#     dim = 768
#     x = torch.randn(batch_size, num_patches, dim)
    # # 方案一：精确匹配
    # model = SoftMoE(dim=dim, num_experts=3,seq_len=129, num_slots=43)  # 5*26=130   [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96]
    # # 确保num_slots=129//num_experts
    # x = torch.randn(batch_size, num_patches, dim)
    # print(model)
    # output = model(x)
    # assert output.shape == x.shape, f"Shape mismatch: {output.shape} vs {x.shape}"
    # print("方案一测试通过 (精确匹配)")



    # model3 = SoftMoE3(dim=dim, num_slots=43, num_experts=3)
    # output3 = model3(x)
    # print(model3)
    # print(f"output3.shape:{output3.shape}")
    # assert output3.shape == x.shape  # 应通过


#     model4 = SoftMoE4(
#     dim=dim,
#     num_experts=3,
#     capacity_factor=1.2,
#     noise_std=0.1
# )
#     print(model4)
#     output, metrics = model4(x)

#     print(output.shape)  # (64, 129, 768)
#     print(metrics.keys()) # 输出统计指标]

#     model5 = SoftMoE6(
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
#     print(metrics.keys()) # 输出统计指标
    # # 方案二：动态填充
    # model = SoftMoE(dim=dim, num_experts=4, num_slots=33)
    # x = torch.randn(batch_size, num_patches, dim)
    # mask = torch.ones(batch_size, num_patches, dtype=torch.bool)
    
    # # 填充至132
    # x_padded = torch.nn.functional.pad(x, (0, 0, 0, 2))
    # mask_padded = torch.nn.functional.pad(mask, (0, 2), value=False)
    
    # output = model(x_padded, mask=mask_padded)
    # output = output[:, :num_patches, :]  # 截取原始长度
    # assert output.shape == x.shape, f"Shape mismatch: {output.shape} vs {x.shape}"
    # print("方案二测试通过 (动态填充)")

# if __name__ == "__main__":
#     test_image_input()
