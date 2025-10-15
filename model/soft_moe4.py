import math
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from .fast_softmoe_layer import MultiExpertLayer
from einops import rearrange
from typing import Callable, Optional, Tuple, Union
from timm.layers import DropPath
import torch.nn.functional as F
from timm.layers import trunc_normal_, lecun_normal_
from entmax import sparsemax, entmax15


# def l2norm(t, dim=-1):
#     return F.normalize(t, dim = dim)

# def l2norm(x):
#     return x / (torch.norm(x, dim=-1, keepdim=True) + 1e-8)

def l2norm(x, dim=-1, eps=1e-6):
    norm = torch.sqrt(torch.sum(x**2, dim=dim, keepdim=True))
    return x * (1 / (norm + eps))

def stable_softmax(x, dim):
    z = x - x.max(dim=dim, keepdim=True)[0]
    return torch.exp(z) / torch.exp(z).sum(dim=dim, keepdim=True)

class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        # self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x, norm_dim=-1):
        # return l2norm(x, norm_dim) * self.scale * self.gamma
        return l2norm(x, norm_dim) * self.gamma

def log(t, eps = 1e-20):
    return torch.log(t.clamp(min = eps))

def gumbel_noise(t):
    noise = torch.zeros_like(t).uniform_(0, 1)
    return -log(-log(noise))

def normal_noise(t):
    # noise = torch.normal(mean=0, std=1, size=t.shape[-2:],requires_grad=False).to(t.device)
    noise = torch.normal(mean=0, std=1, size=t.shape, device=t.device)
    return noise

class NormalNoiseGenerator(nn.Module):
    """Generates a random noisy mask for logits tensor.

    All noise is generated from a normal distribution :math:`(0, 1 / E^2)`, where
    `E = the number of experts`.

    Args:
        num_experts (int): The number of experts.
    """

    def __init__(self, num_experts: int):
        super(NormalNoiseGenerator, self).__init__()
        self.num_experts = num_experts
        self.loc = 0.0
        self.scale = 1.0 / num_experts**2

    def forward(self, inputs: torch.Tensor):
        device = inputs.device
        normal_dist = torch.distributions.normal.Normal(
            loc=torch.tensor(self.loc, device=device),
            scale=torch.tensor(self.scale, device=device)
        )
        noisy = normal_dist.rsample(inputs.shape)
        return inputs + noisy


class LearnableNormalNoiseGenerator(nn.Module):
    """Generates a random noisy mask for logits tensor.

    All noise is generated from a normal distribution :math:`(0, 1 / E^2)`, where
    `E = the number of experts`.

    Args:
        num_experts (int): The number of experts.
    """

    def __init__(self, num_experts: int):
        super(LearnableNormalNoiseGenerator, self).__init__()
        self.num_experts = num_experts
        self.loc_min, self.loc_max = -0.1, 0.1
        self.scale_min, self.scale_max = 1e-12, 0.1

        # Initialize parameters
        initial_loc_value = 0.0
        initial_scale_value = 1.0 / num_experts**2
        self.loc = nn.Parameter(self._initialize_param(initial_loc_value, self.loc_min, self.loc_max))
        self.scale = nn.Parameter(self._initialize_param(initial_scale_value, self.scale_min, self.scale_max))

    def forward(self, inputs: torch.Tensor):
        loc = self._apply_constraints(self.loc, self.loc_min, self.loc_max)
        scale = self._apply_constraints(self.scale, self.scale_min, self.scale_max)
        normal_dist = torch.distributions.normal.Normal(loc=loc, scale=scale)
        noisy = normal_dist.rsample(inputs.shape)
        return inputs + noisy

    def _apply_constraints(self, param, min_val, max_val):
        # Apply constraints using a sigmoid function to keep values within [min_val, max_val]
        return min_val + (max_val - min_val) * torch.sigmoid(param)

    def _initialize_param(self, target_value, min_val, max_val):
        # Calculate the initial parameter value that will result in target_value after applying constraints
        target_scaled = (target_value - min_val) / (max_val - min_val)
        initial_param = torch.log(torch.tensor(target_scaled / (1.0 - target_scaled)))
        return initial_param


class UniformNoiseGenerator(nn.Module):
    """Generates Uniform noise for logits tensor."""

    def __init__(self, num_experts: int):
        super(UniformNoiseGenerator, self).__init__()
        self.num_experts = num_experts
        self.eps = 1e-2

    def forward(self, inputs: torch.Tensor):
        device = inputs.device
        uniform_dist = torch.distributions.uniform.Uniform(
            low=torch.tensor(1.0 + self.eps, device=device),
            high=torch.tensor(1.0 - self.eps, device=device)
        )
        noisy = uniform_dist.rsample(inputs.shape)
        return inputs * noisy

class OffsetScale(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.beta = nn.Parameter(torch.zeros(dim))
        # nn.init.normal_(self.gamma, std = 0.02)

    def forward(self, x):
        out = x * self.gamma + self.beta
        return out


def softmax(x: torch.Tensor, dim) -> torch.Tensor:
    """
    Compute the softmax along the specified dimensions.
    This function adds the option to specify multiple dimensions

    Args:
        x (torch.Tensor): Input tensor.
        dims (int or tuple[int]): The dimension or list of dimensions along which the softmax probabilities are computed.

    Returns:
        torch.Tensor: Output tensor containing softmax probabilities along the specified dimensions.
    """
    max_vals = torch.amax(x, dim=dim, keepdim=True)
    e_x = torch.exp(x - max_vals)
    sum_exp = e_x.sum(dim=dim, keepdim=True)
    return e_x / sum_exp


def adjust_query_length(queries, new_length):
    """
    Adjust the length of queries to new_length using interpolation.
    
    :param queries: Input queries tensor of shape (N, query_len, embed_size)
    :param new_length: The desired length of the queries
    :return: New queries of shape (N, new_length, embed_size)
    """
    N, query_len, embed_size = queries.shape
    # Change the shape to (N, embed_size, query_len) for interpolation
    queries = queries.transpose(1, 2)
    # Interpolate along the last dimension (original query_len)
    queries_interpolated = F.interpolate(queries, size=new_length, mode='linear', align_corners=False)
    # Transpose back to (N, new_length, embed_size)
    queries_interpolated = queries_interpolated.transpose(1, 2)
    return queries_interpolated


class SoftMoELayerWrapper(nn.Module):
    """
    A wrapper class to create a Soft Mixture of Experts layer.

    From "From Sparse to Soft Mixtures of Experts"
    https://arxiv.org/pdf/2308.00951.pdf
    """

    def __init__(
        self,
        dim: int,
        num_experts: int,
        slots_per_expert: int,
        add_noise: bool = False,
        noise_mult: float = 1.0,
        moe_droprate: float = 0.0,
        moe_drop_path_rate: float = 0.0,
        moe_logits_drop: float = 0.0,
        compress_ratio: float = 1.0,
        phi: nn.Parameter = None,
        key_proj: bool = True,
        query_proj: bool = True,
        slot_layernorm: bool = True,
        precision=torch.float32,
        **kwargs,
    ) -> None:
        """
        Args:
            dim (int): Dimensionality of input features.
            num_experts (int): Number of experts.
            slots_per_expert (int): Number of token slots per expert.
            layer (Callable): Network layer of the experts.
            normalize (bool): Normalize input and phi (sec. 2.3 from paper)
            **layer_kwargs: Additional keyword arguments for the layer class.
        """
        super().__init__()

        self.dim = dim
        self.num_experts = num_experts
        self.slots_per_expert = slots_per_expert
        self.add_noise = add_noise
        self.noise_mult = noise_mult
        self.compute_similarity_metrics = True
        self.precision = precision


        # Initialize phi and normalization scaling factor
        # self.register_parameter('phi' , nn.Parameter(torch.zeros(num_experts, slots_per_expert, dim)))
        if phi is None:
            print(f"传入的num_experts：{num_experts}")
            print(f"传入的slots_per_expert：{slots_per_expert}")
            self.phi = nn.Parameter(torch.zeros(num_experts, slots_per_expert, dim))
            # Initialize phi using LeCun normal initialization
            # https://github.com/google-research/vmoe/blob/662341d007650d5bbb7c6a2bef7f3c759a20cc7e/vmoe/projects/soft_moe/router.py#L49C1-L49C1
            nn.init.normal_(self.phi, mean=0, std=1 / dim**0.5)
        else:
            self.phi = phi

        if slot_layernorm:
            self.norm = nn.Identity()
            self.slot_norm = nn.LayerNorm(dim)
        else:
            self.norm = l2norm
            self.slot_norm = l2norm
        self.scale = nn.Parameter(torch.tensor(1.0))
        
        self.key_proj = key_proj
        if self.key_proj:
            self.phi_key_proj = OffsetScale(dim)
        else:
            self.phi_key_proj = nn.Identity()

        self.query_proj = query_proj
        if self.query_proj:
            self.phi_query_proj = OffsetScale(dim)
        else:
            self.phi_query_proj = nn.Identity()

        # Create a list of expert networks
        # self.experts = nn.ModuleList(
        #     [layer(**layer_kwargs) for _ in range(num_experts)]
        # )
        self.experts = MultiExpertLayer(
            in_dim=dim, 
            hidden_dim=int(4* compress_ratio * dim), 
            num_experts=num_experts, 
            moe_droprate=moe_droprate,
            layer_scale=kwargs.get('layer_scale', False),
            freeze_moe=kwargs.get('freeze_moe', False),
            )
        
        self.moe_logits_drop = moe_logits_drop
        self.expert_drop = DropPath(moe_drop_path_rate) if moe_drop_path_rate > 0. else nn.Identity()

    def cosine_psim(self, x, contract_dims, batch_dims=None, eps=1e-9):
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

    def get_metrics(self, combine_weights, dispatch_weights, mu=None):
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
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the Soft-MoE layer (algorithm 1 from paper).

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, seq_len, input_dim].

        Returns:
            torch.Tensor: Output tensor of shape [batch_size, seq_len, input_dim].
        """
        assert (
            x.shape[-1] == self.dim
        ), f"Input feature dim of {x.shape[-1]} does not match layer dim of {self.dim}"

        if len(x.shape) == 4:
            is_conv = True
            x = rearrange(x, 'b h w d -> b (h w) d')
        else:
            is_conv = False
            assert (len(x.shape) == 3), f"Input expected to have 3 or 4 dimensions but has {len(x.shape)}"
        B, N, D = x.shape
        query = self.phi

        # Normalize input and phi
        key = self.norm(self.phi_key_proj(x))
        query = self.slot_norm(self.phi_query_proj(query))
        query = query * self.scale
            # if x_norm.numel() < phi.numel():
            #     x_norm = x_norm * self.scale
            # else:
            #     phi = phi * self.scale

        # Compute dispatch and combine weights
        logits = torch.einsum("b n d, e s d->b n e s", key, query)
        B,N,E,S = logits.shape
        # noised dispatch and combine gate logits, with annealing if needed

        if self.add_noise:
            noise = normal_noise(logits) * self.noise_mult
            logits = logits + noise

        if self.moe_logits_drop > 0.0:
            mask = torch.ones_like(logits) * self.moe_logits_drop
            mask = torch.bernoulli(mask) * -1e12
            logits = logits + mask

        dispatch_weights = logits.softmax(dim = 1)

        combine_weights = rearrange(logits, 'b n e s -> b n (e s)')
        combine_weights = combine_weights.softmax(dim = -1)
        combine_weights_flat = combine_weights.view(B, N, E, S)
        # print(f"dispatch_weights.shape:{dispatch_weights.shape}")
        # print(f"combine_weights_flat.shape:{combine_weights_flat.shape}")

        # Compute input slots as weighted average of input tokens using dispatch weights
        slots = torch.einsum('b n d, b n e s -> b e s d', x, dispatch_weights)

        # Apply expert to corresponding slots
        # out = torch.stack(
        #     [f_i(slots[:, i, :, :]) for i, f_i in enumerate(self.experts)], dim=1
        # )
        out = self.experts(slots)
        # Compute output tokens as weighted average of output slots using combine weights
        out = rearrange(out, ' b e s d -> b (e s) d')
        out = self.expert_drop(out)
        out = torch.einsum('b s d, b n s -> b n d', out, combine_weights)
        if is_conv:
            out = rearrange(out, 'b (h w) d -> b h w d', h=int(out.shape[1]**0.5))
        
        metrics = self.get_metrics(combine_weights=combine_weights_flat, dispatch_weights=dispatch_weights)
        return out,metrics


class DualPathSoftMoELayerWrapper(nn.Module):
    """
    A wrapper class to create a Soft Mixture of Experts layer.

    From "From Sparse to Soft Mixtures of Experts"
    https://arxiv.org/pdf/2308.00951.pdf
    """

    def __init__(
        self,
        dim: int,
        num_experts: int,
        slots_per_expert: int,
        add_noise: bool = False,
        noise_mult: float = 1.0,
        moe_droprate: float = 0.0,
        moe_droprate_act: Optional[float] = None,
        moe_logits_drop: float = 0.0,
        moe_drop_path_rate: float = 0.0,
        compress_ratio: float = 1.0,
        phi = None,
        key_proj: bool = True,
        query_proj: bool = True,
        input_as_phi: bool = False,
        slot_layernorm: bool = True,
        multi_head: bool = True,
        **kwargs,
    ) -> None:
        """
        Args:
            dim (int): Dimensionality of input features.
            num_experts (int): Number of experts.
            slots_per_expert (int): Number of token slots per expert.
            layer (Callable): Network layer of the experts.
            normalize (bool): Normalize input and phi (sec. 2.3 from paper)
            **layer_kwargs: Additional keyword arguments for the layer class.
        """
        super().__init__()

        self.dim = dim
        self.num_experts = num_experts
        self.slots_per_expert = slots_per_expert
        self.add_noise = add_noise
        self.noise_mult = noise_mult

        self.occ_factor = kwargs.get('occ_factor', 1/4)
        self.N_core = kwargs.get('core_experts', int(self.num_experts / 2))
        self.N_univ = kwargs.get('occ_experts', int(self.N_core / self.occ_factor))
        logit_scale = kwargs.get('logit_scale', 1.0)

        self.multi_head = multi_head
        if self.multi_head:
            self.num_heads = kwargs.get('num_heads', 12)
            self.head_dim = dim // self.num_heads

        # Initialize phi and normalization scaling factor
        # self.register_parameter('phi' , nn.Parameter(torch.zeros(num_experts, slots_per_expert, dim)))
        self.input_as_phi = input_as_phi
        self.num_experts = self.N_core + self.N_univ
        self.slots_per_expert = slots_per_expert
        if phi is None and not input_as_phi:
            
            # self.phi = nn.Parameter(torch.zeros((self.N_core + self.N_univ), slots_per_expert, dim))
            # trunc_normal_(self.phi, std=0.02)
            query_init = self.initialize_query((self.N_core + self.N_univ), slots_per_expert, dim, pattern_type='orthogonal')
            self.phi = nn.Parameter(query_init)

        elif not input_as_phi:
            self.phi = phi

        if slot_layernorm:
            # self.norm = nn.Identity()
            self.norm = l2norm
            self.slot_norm = nn.LayerNorm(dim) if not self.multi_head else nn.LayerNorm(self.head_dim)
        else:
            self.norm = l2norm
            self.slot_norm = nn.Identity()
        # self.scale = nn.Parameter(torch.tensor(1.0) * logit_scale)
        self.scales = nn.ParameterList([
            nn.Parameter(torch.tensor(0.5) * logit_scale),
            nn.Parameter(torch.tensor(0.5) * logit_scale)
        ])
        
        self.key_proj = key_proj
        if self.key_proj:
            self.phi_key_proj = OffsetScale(dim) if not self.multi_head else OffsetScale(self.head_dim)
            # self.phi_key_proj = nn.Linear(dim, dim)

        else:
            self.phi_key_proj = nn.Identity()

        self.query_proj = query_proj
        if self.query_proj:
            self.phi_query_proj = OffsetScale(dim) if not self.multi_head else OffsetScale(self.head_dim)
            # self.phi_query_proj = nn.Linear(dim, dim)
        else:
            self.phi_query_proj = nn.Identity()

        self.moe_logits_drop = moe_logits_drop
        # self.moe_logits_drop = nn.Dropout(moe_logits_drop) if moe_logits_drop > 0.0 else nn.Identity()
        self.expert_drop = DropPath(moe_drop_path_rate) if moe_drop_path_rate > 0. else nn.Identity()
        self.core_experts = MultiExpertLayer(in_dim=dim, hidden_dim=int(4 * compress_ratio * dim), num_experts=self.N_core, moe_droprate=moe_droprate, moe_droprate_act=moe_droprate_act,glu=False)
        self.occ_experts = MultiExpertLayer(in_dim=dim, hidden_dim=int(4 * self.occ_factor * compress_ratio * dim), num_experts=self.N_univ, moe_droprate=moe_droprate, moe_droprate_act=moe_droprate_act,glu=False)

        # self.noise_generator = LearnableNormalNoiseGenerator(self.num_experts)

    def initialize_query(self, num_experts, slots_per_expert, d_model, pattern_type='orthogonal'):
        query_init = torch.randn(num_experts, slots_per_expert, d_model)
        if pattern_type == 'orthogonal':
            nn.init.orthogonal_(query_init)
        elif pattern_type == 'trunc_normal':
            trunc_normal_(query_init, mean=0.0, std=0.02)  # 截断正态分布初始化
        else:
            raise ValueError(f"Unsupported pattern_type: {pattern_type}")

        # pos_encoding = torch.arange(0, d_model).unsqueeze(0).repeat(num_experts, 1).float()
        # query_init += 0.01 * pos_encoding

        return query_init
    def aggregate_expert_attention(self, dispatch_weights, combine_weights):
        """
        聚合每个专家的注意力分布
        """
        # 确保张量维度一致
        if self.multi_head:
            # 处理多头情况
            # 聚合分发权重：平均所有槽的权重
            # expert_dispatch = dispatch_weights.mean(dim=-1)  # [B, H, N, E]
            expert_dispatch = dispatch_weights.permute(0, 2, 1, 3)  # [B, N, H, E]
            
            # 聚合合并权重：按专家分组
            # [B, H, N, E*S] -> [B, H, N, E, S]
            expert_combine = combine_weights.unflatten(-1, (self.num_experts, self.slots_per_expert))
            expert_combine = expert_combine.sum(dim=-1)  # [B, H, N, E]
            expert_combine = expert_combine.permute(0, 2, 1, 3)  # [B, N, H, E]
            
            # 组合两种权重
            # expert_attentions = (expert_dispatch + expert_combine) / 2
            expert_attentions = expert_dispatch
            expert_attentions = expert_attentions.mean(dim=2)  # [B, N, E]
        else:
            # 处理单头情况
            # 聚合分发权重：平均所有槽的权重
            expert_dispatch = dispatch_weights.mean(dim=-1)  # [B, N, E]
            
            # 聚合合并权重：按专家分组
            # [B, N, E*S] -> [B, N, E, S]
            expert_combine = combine_weights.unflatten(-1, (self.num_experts, self.slots_per_expert))
            expert_combine = expert_combine.sum(dim=-1)  # [B, N, E]
            
            # 组合两种权重
            expert_attentions = (expert_dispatch + expert_combine) / 2
        
        return expert_attentions.permute(0, 2, 1)  # [B, E, N]

    def calculate_expert_contributions(self, combine_weights):
        """
        计算每个专家的贡献比例
        """
        try:
            if combine_weights is None:
                return None
                
            # 确保是张量
            if not isinstance(combine_weights, torch.Tensor):
                return None
                
            # 聚合合并权重：按专家分组
            if self.multi_head:
                # [B, H, N, E*S] -> [B, H, N, E, S]
                expert_combine = combine_weights.unflatten(-1, (self.num_experts, self.slots_per_expert))
                expert_combine = expert_combine.sum(dim=-1)  # [B, H, N, E]
                expert_combine = expert_combine.mean(dim=1)  # [B, N, E]
            else:
                # [B, N, E*S] -> [B, N, E, S]
                expert_combine = combine_weights.unflatten(-1, (self.num_experts, self.slots_per_expert))
                expert_combine = expert_combine.sum(dim=-1)  # [B, N, E]
            
            # 按专家和批次平均
            # [num_experts]
            avg_contrib = expert_combine.mean(dim=(0, 1))
            
            return avg_contrib
            
        except Exception as e:
            print(f"计算专家贡献时出错: {str(e)}")
            return None
    def forward(self, x: torch.Tensor) -> torch.Tensor:

        assert (
            x.shape[-1] == self.dim
        ), f"Input feature dim of {x.shape[-1]} does not match layer dim of {self.dim}"

        if len(x.shape) == 4:
            is_conv = True
            x = rearrange(x, 'b h w d -> b (h w) d')
        else:
            is_conv = False
            assert (len(x.shape) == 3), f"Input expected to have 3 or 4 dimensions but has {len(x.shape)}"

        if not self.input_as_phi:
            query = self.phi
        else:
            query = adjust_query_length(x, self.num_experts*self.slots_per_expert)
            query = rearrange(query, 'b (e s) d -> b e s d', e=int(self.num_experts))

        # Normalize input and phi
        if self.multi_head:
            x = rearrange(x, 'b n (h d) -> b h n d', h=self.num_heads)
            query = rearrange(query, 'e s (h d) -> h e s d', h=self.num_heads)

        key = self.norm(self.phi_key_proj(x))
        query = self.slot_norm(self.phi_query_proj(query))
        query = l2norm(query)
        # query = query * self.scale

        # Compute dispatch and combine weights
        if self.multi_head:
            logits = torch.einsum("b h n d, h e s d -> b h n e s", key, query)
        else: 
            if not self.input_as_phi:
                logits = torch.einsum("b n d, e s d->b n e s", key, query)
            else:
                logits = torch.einsum("b n d, b e s d->b n e s", key, query)
                logits = logits / (self.dim ** 0.5)

        # noised dispatch and combine gate logits, with annealing if needed
        dispatch_logits = logits / self.scales[0]
        combine_logits = logits / self.scales[1]
        
        if self.add_noise:
            noise = normal_noise(logits) * self.noise_mult
            # logits = logits + noise
            dispatch_logits = dispatch_logits + noise
            combine_logits = combine_logits + noise
            # logits = self.noise_generator(logits)

        if self.moe_logits_drop > 0.0:
            mask = torch.ones_like(logits) * self.moe_logits_drop
            mask = torch.bernoulli(mask) * -1e12
            logits = logits + mask
        
        # dispatch_weights = dispatch_logits.softmax(dim = -3)
        # combine_weights = combine_logits.flatten(start_dim=-2).softmax(dim = -1)

        dispatch_weights = softmax(dispatch_logits, dim=-3)
        combine_weights = entmax15(
                combine_logits.flatten(start_dim=-2), dim=-1)

        # dispatch_weights = self.moe_logits_drop(dispatch_weights)
        # combine_weights = self.moe_logits_drop(combine_weights)

        slots = torch.einsum('... n d, ... n e s -> ... e s d', x, dispatch_weights)

        if self.multi_head:
            slots = rearrange(slots, 'b h e s d -> b e s (h d)')

        core_slots = slots[:, :self.N_core, :, :]
        occ_slots = slots[:, self.N_core:, :, :]

        core_out = self.core_experts(core_slots)
        univ_out = self.occ_experts(occ_slots)

        out = torch.cat((core_out, univ_out), dim=1)

        # Compute output tokens as weighted average of output slots using combine weights
        if self.multi_head:
            out = rearrange(out, ' b e s (h d) -> b h (e s) d', h=self.num_heads)
        else:
            out = rearrange(out, ' b e s d -> b (e s) d')

        out = self.expert_drop(out)
        out = torch.einsum('... s d, ... n s -> ... n d', out, combine_weights)

        if self.multi_head:
            out = rearrange(out, 'b h n d -> b n (h d)', h=self.num_heads)

        if is_conv:
            out = rearrange(out, 'b (h w) d -> b h w d', h=int(out.shape[1]**0.5))
        
        metrics = {'auxiliary_loss': torch.tensor(0.0),
        'combine_weights_similarity_mean': torch.tensor(0.0),
        'dispatch_weights_similarity_mean': torch.tensor(0.0)}
        # print(f"dispatch_weights:{dispatch_weights.shape}")
        # print(f"combine_weights:{combine_weights.shape}") # combine_weights:torch.Size([256, 12, 129, 64])
        dispatch_weights = dispatch_weights.squeeze()
        # print(f"dispatch_weights:{dispatch_weights.shape}") # dispatch_weights:torch.Size([256, 12, 129, 64])
        expert_attentions = self.aggregate_expert_attention(dispatch_weights, combine_weights)
        expert_contributions = self.calculate_expert_contributions(combine_weights)
        # return out,metrics,dispatch_weights
        return out, metrics, dispatch_weights, combine_weights, expert_attentions, expert_contributions
    









class DualPathSoftMoE2(nn.Module):
    """
    A wrapper class to create a Soft Mixture of Experts layer.

    From "From Sparse to Soft Mixtures of Experts"
    https://arxiv.org/pdf/2308.00951.pdf
    """

    def __init__(
        self,
        dim: int,
        num_experts: int,
        slots_per_expert: int,
        add_noise: bool = False,
        noise_mult: float = 1.0,
        moe_droprate: float = 0.0,
        moe_droprate_act: Optional[float] = None,
        moe_logits_drop: float = 0.0,
        moe_drop_path_rate: float = 0.0,
        compress_ratio: float = 1.0,
        phi = None,
        key_proj: bool = True,
        query_proj: bool = True,
        input_as_phi: bool = False,
        slot_layernorm: bool = True,
        multi_head: bool = False,
        tau = 1.0,
        **kwargs,
    ) -> None:
        """
        Args:
            dim (int): Dimensionality of input features.
            num_experts (int): Number of experts.
            slots_per_expert (int): Number of token slots per expert.
            layer (Callable): Network layer of the experts.
            normalize (bool): Normalize input and phi (sec. 2.3 from paper)
            **layer_kwargs: Additional keyword arguments for the layer class.
        """
        super().__init__()

        self.dim = dim
        self.num_experts = num_experts
        self.slots_per_expert = slots_per_expert
        self.add_noise = add_noise
        self.noise_mult = noise_mult

        self.occ_factor = kwargs.get('occ_factor', 1/4)
        self.N_core = kwargs.get('core_experts', int(self.num_experts / 2))
        self.N_univ = kwargs.get('occ_experts', int(self.N_core / self.occ_factor))
        logit_scale = kwargs.get('logit_scale', 1.0)

        self.multi_head = multi_head
        if self.multi_head:
            self.num_heads = kwargs.get('num_heads', 3)
            self.head_dim = dim // self.num_heads

        # Initialize phi and normalization scaling factor
        # self.register_parameter('phi' , nn.Parameter(torch.zeros(num_experts, slots_per_expert, dim)))
        self.input_as_phi = input_as_phi
        self.num_experts = self.N_core + self.N_univ
        self.slots_per_expert = slots_per_expert
        query_init = self.initialize_query((self.N_core + self.N_univ), slots_per_expert, dim, pattern_type='orthogonal')
        self.phi = nn.Parameter(query_init)
        # if phi is None and not input_as_phi:
            
        #     # self.phi = nn.Parameter(torch.zeros((self.N_core + self.N_univ), slots_per_expert, dim))
        #     # trunc_normal_(self.phi, std=0.02)
        #     query_init = self.initialize_query((self.N_core + self.N_univ), slots_per_expert, dim, pattern_type='orthogonal')
        #     self.phi = nn.Parameter(query_init)

        # elif not input_as_phi:
        #     self.phi = phi

        if slot_layernorm:
            # self.norm = nn.Identity()
            self.norm = l2norm
            self.slot_norm = nn.LayerNorm(dim) if not self.multi_head else nn.LayerNorm(self.head_dim)
        else:
            self.norm = l2norm
            self.slot_norm = nn.Identity()
        # self.scale = nn.Parameter(torch.tensor(1.0) * logit_scale)
        self.scales = nn.ParameterList([
            nn.Parameter(torch.tensor(0.5) * logit_scale),
            nn.Parameter(torch.tensor(0.5) * logit_scale)
        ])
        
        self.key_proj = key_proj
        if self.key_proj:
            self.phi_key_proj = OffsetScale(dim) if not self.multi_head else OffsetScale(self.head_dim)
            # self.phi_key_proj = nn.Linear(dim, dim)

        else:
            self.phi_key_proj = nn.Identity()

        self.query_proj = query_proj
        if self.query_proj:
            self.phi_query_proj = OffsetScale(dim)
            # self.phi_query_proj = nn.Linear(dim, dim)
        else:
            self.phi_query_proj = nn.Identity()

        self.moe_logits_drop = moe_logits_drop
        # self.moe_logits_drop = nn.Dropout(moe_logits_drop) if moe_logits_drop > 0.0 else nn.Identity()
        self.expert_drop = DropPath(moe_drop_path_rate) if moe_drop_path_rate > 0. else nn.Identity()
        self.core_experts = MultiExpertLayer(in_dim=dim, hidden_dim=int(4 * compress_ratio * dim), num_experts=self.N_core, moe_droprate=moe_droprate, moe_droprate_act=moe_droprate_act)
        self.occ_experts = MultiExpertLayer(in_dim=dim, hidden_dim=int(4 * self.occ_factor * compress_ratio * dim), num_experts=self.N_univ, moe_droprate=moe_droprate, moe_droprate_act=moe_droprate_act)
        self.tau = nn.Parameter(torch.tensor(tau))  # 可学习温度系数
        self.register_buffer("core_expert_mask", torch.zeros(1, 1, num_experts, 1))
        self.register_buffer("univ_expert_mask", 1 - self.core_expert_mask)
        self.core_expert_mask = torch.zeros(1, 1, self.num_experts, 1)  # [1,1,32,1]
        self.core_expert_mask[:, :, :self.num_experts//2, :] = 1.0
        self.univ_expert_mask = 1 - self.core_expert_mask
        # self.noise_generator = LearnableNormalNoiseGenerator(self.num_experts)

    def initialize_query(self, num_experts, slots_per_expert, d_model, pattern_type='orthogonal'):
        query_init = torch.randn(num_experts, slots_per_expert, d_model)
        if pattern_type == 'orthogonal':
            nn.init.orthogonal_(query_init)
        elif pattern_type == 'trunc_normal':
            trunc_normal_(query_init, mean=0.0, std=0.02)  # 截断正态分布初始化
        else:
            raise ValueError(f"Unsupported pattern_type: {pattern_type}")

        # pos_encoding = torch.arange(0, d_model).unsqueeze(0).repeat(num_experts, 1).float()
        # query_init += 0.01 * pos_encoding

        return query_init

    def forward(self, x: torch.Tensor,attn_weight, mode='original') -> torch.Tensor:

        assert (
            x.shape[-1] == self.dim
        ), f"Input feature dim of {x.shape[-1]} does not match layer dim of {self.dim}"

        if len(x.shape) == 4:
            is_conv = True
            x = rearrange(x, 'b h w d -> b (h w) d')
        else:
            is_conv = False
            assert (len(x.shape) == 3), f"Input expected to have 3 or 4 dimensions but has {len(x.shape)}"
        # if attn_weight is None:
        #     print("传入softmoe的attn_weight为None")
        # else:
        #     print("attn_weight.max:")
        #     print(attn_weight.max().item())
        #     print("attn_weight.min:")
        #     print(attn_weight.min().item())
        #     print("attn_weight.mean:")
        #     print(attn_weight.mean().item())
        if not self.input_as_phi:
            query = self.phi
        else:
            query = adjust_query_length(x, self.num_experts*self.slots_per_expert)
            query = rearrange(query, 'b (e s) d -> b e s d', e=int(self.num_experts))

        # Normalize input and phi
        if self.multi_head:
            x = rearrange(x, 'b n (h d) -> b h n d', h=self.num_heads)
            query = rearrange(query, 'e s (h d) -> h e s d', h=self.num_heads)

        key = self.norm(self.phi_key_proj(x))
        query = self.slot_norm(self.phi_query_proj(query))
        query = l2norm(query)
        # query = query * self.scale

        # Compute dispatch and combine weights
        if self.multi_head:
            logits = torch.einsum("b h n d, h e s d -> b h n e s", key, query)
        else: 
            if not self.input_as_phi:
                logits = torch.einsum("b n d, e s d->b n e s", key, query)
            else:
                logits = torch.einsum("b n d, b e s d->b n e s", key, query)
                logits = logits / (self.dim ** 0.5)

        # ========== 模式区分 ==========
        if mode == 'original':
            # 强制所有输入路由到 core_experts（前16专家）
            core_expert_mask = torch.zeros_like(logits)  # [B,129,E,S]
            core_expert_mask[..., :self.N_core, :] = 1.0
            adjusted_logits = logits * core_expert_mask + (1 - core_expert_mask) * (-1e4)
            # 验证调整后的 logits 值
            print(f"Original Mode - Adjusted Logits (non-core): {adjusted_logits[0, 0, self.N_core:, 0]}")  # 应全为 -1e9
            print(f"Original Mode - Adjusted Logits (core): {adjusted_logits[0, 0, :self.N_core, 0]}") 
            dispatch_logits = adjusted_logits / self.scales[0]
        else:
            assert attn_weight.shape == (x.shape[0], 128), "attn_weight 必须为 [B,128]"
            # 动态路由逻辑（基于 attn_weight）
            cls_logits = logits[:, :1, :, :]          # [B,1,E,S]
            patch_logits = logits[:, 1:, :, :]        # [B,128,E,S]

            attn_mask = attn_weight.unsqueeze(-1).unsqueeze(-1)  # [B,128,1,1]
            core_bias = attn_mask * self.core_expert_mask * self.tau
            univ_bias = (1 - attn_mask) * self.univ_expert_mask * self.tau
            adjusted_patch_logits = patch_logits + core_bias + univ_bias

            adjusted_logits = torch.cat([cls_logits, adjusted_patch_logits], dim=1)
            print(f"occ Mode - Adjusted Logits (non-core): {adjusted_logits[0, 0, self.N_core:, 0]}")  # 应全为 -1e9
            print(f"occ Mode - Adjusted Logits (core): {adjusted_logits[0, 0, :self.N_core, 0]}") 
            dispatch_logits = adjusted_logits / self.scales[0]
        # # 分离CLS和小块（CLS保持原始逻辑）
        # cls_logits = logits[:, :1, :, :]          # [B,1,E,S]
        # patch_logits = logits[:, 1:, :, :]        # [B,128,E,S]
        # # 获取 attn_weight [B,128] 并扩展维度
        # attn_mask = attn_weight.unsqueeze(-1).unsqueeze(-1)  # [B,128,1,1]
        # # 动态调整 logits
        # # ----------------------------------------------------------------
        # # 定义专家分组掩码
        # E = logits.shape[-3]  # 总专家数（假设E=32）
        # # core_expert_mask = torch.zeros(1, 1, E, 1, device=logits.device)  # [1,1,32,1]
        # # core_expert_mask[:, :, :E//2, :] = 1.0   # 前16专家为core_experts
        # # univ_expert_mask = 1 - core_expert_mask  # 后16专家为univ_experts
        # # 根据 attn_weight 加权调整 logits
        # # 重要小块增强 core_expert 的 logits
        # core_bias = attn_mask * self.core_expert_mask * self.tau  # [B,128,32,1], tau为温度系数（可学习或固定）
        # # 非重要小块增强 univ_expert 的 logits
        # univ_bias = (1 - attn_mask) * self.univ_expert_mask * self.tau

        # # 合并调整后的 patch_logits
        # adjusted_patch_logits = patch_logits + core_bias + univ_bias  # [B,128,E,S]

        # # 重新组合CLS与调整后的小块logits
        # adjusted_logits = torch.cat([cls_logits, adjusted_patch_logits], dim=1)  # [B,129,E,S]

        # noised dispatch and combine gate logits, with annealing if needed
        # dispatch_logits = logits / self.scales[0]
        # dispatch_logits = adjusted_logits / self.scales[0]
        
        combine_logits = adjusted_logits / self.scales[1]
        
        if self.add_noise:
            noise = normal_noise(logits) * self.noise_mult
            # logits = logits + noise
            dispatch_logits = dispatch_logits + noise
            combine_logits = combine_logits + noise
            # logits = self.noise_generator(logits)

        if self.moe_logits_drop > 0.0:
            mask = torch.ones_like(logits) * self.moe_logits_drop
            mask = torch.bernoulli(mask) * -1e12
            logits = logits + mask
        
        # dispatch_weights = dispatch_logits.softmax(dim = -3)
        # combine_weights = combine_logits.flatten(start_dim=-2).softmax(dim = -1)

        # dispatch_weights = softmax(dispatch_logits, dim=-3)

        dispatch_weights = torch.softmax(dispatch_logits, dim=2)
        print(f"dispatch_logits:{dispatch_logits.shape}")
        print(f"dispatch_weights:{dispatch_weights.shape}")
        # print(f"dispatch_weights:{dispatch_weights.shape}")
        # print(f"self.num_experts:{self.num_experts}")
        # assert dispatch_weights.shape[-3] == self.num_experts
        combine_weights = entmax15(
                combine_logits.flatten(start_dim=-2), dim=-1)
        core_expert_weights = dispatch_weights[..., :self.N_core, :].mean().item()
        univ_expert_weights = dispatch_weights[..., self.N_core:, :].mean().item()

        # 打印模式信息和权重分布
        print(f"\n[Mode: {mode}]")
        print(f"Core Experts Weight Avg: {core_expert_weights:.4f}")
        print(f"Univ Experts Weight Avg: {univ_expert_weights:.4f}")
        # dispatch_weights = self.moe_logits_drop(dispatch_weights)
        # combine_weights = self.moe_logits_drop(combine_weights)
        # print(f"dispatch_weights:{dispatch_weights.shape}")
        # print(f"x:{x.shape}")
        slots = torch.einsum('... n d, ... n e s -> ... e s d', x, dispatch_weights)

        if self.multi_head:
            slots = rearrange(slots, 'b h e s d -> b e s (h d)')

        core_slots = slots[:, :self.N_core, :, :]
        occ_slots = slots[:, self.N_core:, :, :]

        core_out = self.core_experts(core_slots)
        univ_out = self.occ_experts(occ_slots)

        out = torch.cat((core_out, univ_out), dim=1)

        # Compute output tokens as weighted average of output slots using combine weights
        if self.multi_head:
            out = rearrange(out, ' b e s (h d) -> b h (e s) d', h=self.num_heads)
        else:
            out = rearrange(out, ' b e s d -> b (e s) d')

        out = self.expert_drop(out)
        out = torch.einsum('... s d, ... n s -> ... n d', out, combine_weights)

        if self.multi_head:
            out = rearrange(out, 'b h n d -> b n (h d)', h=self.num_heads)

        if is_conv:
            out = rearrange(out, 'b (h w) d -> b h w d', h=int(out.shape[1]**0.5))
        
        metrics = {'auxiliary_loss': torch.tensor(0.0),
        'combine_weights_similarity_mean': torch.tensor(0.0),
        'dispatch_weights_similarity_mean': torch.tensor(0.0)}

        return out,metrics
    

# 直接加
class DualPathSoftMoE3(nn.Module):
    """
    A wrapper class to create a Soft Mixture of Experts layer.

    From "From Sparse to Soft Mixtures of Experts"
    https://arxiv.org/pdf/2308.00951.pdf
    """

    def __init__(
        self,
        dim: int,
        num_experts: int,
        slots_per_expert: int,
        add_noise: bool = False,
        noise_mult: float = 1.0,
        moe_droprate: float = 0.0,
        moe_droprate_act: Optional[float] = None,
        moe_logits_drop: float = 0.0,
        moe_drop_path_rate: float = 0.0,
        compress_ratio: float = 1.0,
        phi = None,
        key_proj: bool = True,
        query_proj: bool = True,
        input_as_phi: bool = False,
        slot_layernorm: bool = True,
        multi_head: bool = False,
        tau = 1.0,
        **kwargs,
    ) -> None:
        """
        Args:
            dim (int): Dimensionality of input features.
            num_experts (int): Number of experts.
            slots_per_expert (int): Number of token slots per expert.
            layer (Callable): Network layer of the experts.
            normalize (bool): Normalize input and phi (sec. 2.3 from paper)
            **layer_kwargs: Additional keyword arguments for the layer class.
        """
        super().__init__()

        self.dim = dim
        self.num_experts = num_experts
        self.slots_per_expert = slots_per_expert
        self.add_noise = add_noise
        self.noise_mult = noise_mult

        self.occ_factor = kwargs.get('occ_factor', 1/4)
        self.N_core = kwargs.get('core_experts', int(self.num_experts / 2))
        self.N_univ = kwargs.get('occ_experts', int(self.N_core / self.occ_factor))
        logit_scale = kwargs.get('logit_scale', 1.0)

        self.multi_head = multi_head
        if self.multi_head:
            self.num_heads = kwargs.get('num_heads', 3)
            self.head_dim = dim // self.num_heads

        # Initialize phi and normalization scaling factor
        # self.register_parameter('phi' , nn.Parameter(torch.zeros(num_experts, slots_per_expert, dim)))
        self.input_as_phi = input_as_phi
        self.num_experts = self.N_core + self.N_univ
        self.slots_per_expert = slots_per_expert
        if phi is None and not input_as_phi:
            
            # self.phi = nn.Parameter(torch.zeros((self.N_core + self.N_univ), slots_per_expert, dim))
            # trunc_normal_(self.phi, std=0.02)
            query_init = self.initialize_query((self.N_core + self.N_univ), slots_per_expert, dim, pattern_type='orthogonal')
            self.phi = nn.Parameter(query_init)

        elif not input_as_phi:
            self.phi = phi

        if slot_layernorm:
            # self.norm = nn.Identity()
            self.norm = l2norm
            self.slot_norm = nn.LayerNorm(dim) if not self.multi_head else nn.LayerNorm(self.head_dim)
        else:
            self.norm = l2norm
            self.slot_norm = nn.Identity()
        # self.scale = nn.Parameter(torch.tensor(1.0) * logit_scale)
        self.scales = nn.ParameterList([
            nn.Parameter(torch.tensor(0.5) * logit_scale),
            nn.Parameter(torch.tensor(0.5) * logit_scale)
        ])
        
        self.key_proj = key_proj
        if self.key_proj:
            self.phi_key_proj = OffsetScale(dim) if not self.multi_head else OffsetScale(self.head_dim)
            # self.phi_key_proj = nn.Linear(dim, dim)

        else:
            self.phi_key_proj = nn.Identity()

        self.query_proj = query_proj
        if self.query_proj:
            self.phi_query_proj = OffsetScale(dim)
            # self.phi_query_proj = nn.Linear(dim, dim)
        else:
            self.phi_query_proj = nn.Identity()

        self.moe_logits_drop = moe_logits_drop
        # self.moe_logits_drop = nn.Dropout(moe_logits_drop) if moe_logits_drop > 0.0 else nn.Identity()
        self.expert_drop = DropPath(moe_drop_path_rate) if moe_drop_path_rate > 0. else nn.Identity()
        self.core_experts = MultiExpertLayer(in_dim=dim, hidden_dim=int(4 * compress_ratio * dim), num_experts=self.N_core, moe_droprate=moe_droprate, moe_droprate_act=moe_droprate_act)
        self.occ_experts = MultiExpertLayer(in_dim=dim, hidden_dim=int(4 * self.occ_factor * compress_ratio * dim), num_experts=self.N_univ, moe_droprate=moe_droprate, moe_droprate_act=moe_droprate_act)
        self.tau = nn.Parameter(torch.tensor(tau))  # 可学习温度系数
        self.register_buffer("core_expert_mask", torch.zeros(1, 1, num_experts, 1))
        self.register_buffer("univ_expert_mask", 1 - self.core_expert_mask)
        self.core_expert_mask = torch.zeros(1, 1, self.num_experts, 1)  # [1,1,32,1]
        self.core_expert_mask[:, :, :self.num_experts//2, :] = 1.0
        self.univ_expert_mask = 1 - self.core_expert_mask
        # self.noise_generator = LearnableNormalNoiseGenerator(self.num_experts)

    def initialize_query(self, num_experts, slots_per_expert, d_model, pattern_type='orthogonal'):
        query_init = torch.randn(num_experts, slots_per_expert, d_model)
        if pattern_type == 'orthogonal':
            nn.init.orthogonal_(query_init)
        elif pattern_type == 'trunc_normal':
            trunc_normal_(query_init, mean=0.0, std=0.02)  # 截断正态分布初始化
        else:
            raise ValueError(f"Unsupported pattern_type: {pattern_type}")

        # pos_encoding = torch.arange(0, d_model).unsqueeze(0).repeat(num_experts, 1).float()
        # query_init += 0.01 * pos_encoding

        return query_init

    def forward(self, x: torch.Tensor,attn_weight, mode='original') -> torch.Tensor:
        assert (
            x.shape[-1] == self.dim
        ), f"Input feature dim of {x.shape[-1]} does not match layer dim of {self.dim}"

        if len(x.shape) == 4:
            is_conv = True
            x = rearrange(x, 'b h w d -> b (h w) d')
        else:
            is_conv = False
            assert (len(x.shape) == 3), f"Input expected to have 3 or 4 dimensions but has {len(x.shape)}"

        if not self.input_as_phi:
            query = self.phi
        else:
            query = adjust_query_length(x, self.num_experts*self.slots_per_expert)
            query = rearrange(query, 'b (e s) d -> b e s d', e=int(self.num_experts))

        # Normalize input and phi
        if self.multi_head:
            x = rearrange(x, 'b n (h d) -> b h n d', h=self.num_heads)
            query = rearrange(query, 'e s (h d) -> h e s d', h=self.num_heads)

        key = self.norm(self.phi_key_proj(x))
        query = self.slot_norm(self.phi_query_proj(query))
        query = l2norm(query)
        # query = query * self.scale

        # Compute dispatch and combine weights
        if self.multi_head:
            logits = torch.einsum("b h n d, h e s d -> b h n e s", key, query)
        else: 
            if not self.input_as_phi:
                logits = torch.einsum("b n d, e s d->b n e s", key, query)
            else:
                logits = torch.einsum("b n d, b e s d->b n e s", key, query)
                logits = logits / (self.dim ** 0.5)
        # noised dispatch and combine gate logits, with annealing if needed
        dispatch_logits = logits / self.scales[0]
        combine_logits = logits / self.scales[1]

        if mode != 'original':
            # 分离CLS和小块（CLS保持原始逻辑）
            cls_logits = logits[:, :1, :, :]          # [B,1,E,S]
            patch_logits = logits[:, 1:, :, :]        # [B,128,E,S]
            # 获取 attn_weight [B,128] 并扩展维度
            attn_mask = attn_weight.unsqueeze(-1).unsqueeze(-1)  # [B,128,1,1]
            E = self.num_experts
            core_expert_mask = torch.zeros(1, 1, E, 1, device=logits.device)  # [1,1,32,1]
            core_expert_mask[:, :, :E//2, :] = 1.0   # 前16专家为core_experts
            univ_expert_mask = 1 - core_expert_mask  # 后16专家为univ_experts
            # 根据 attn_weight 加权调整 logits
            # 重要小块增强 core_expert 的 logits
            core_bias = attn_mask * core_expert_mask * self.tau  # [B,128,32,1], tau为温度系数（可学习或固定）
            # 非重要小块增强 univ_expert 的 logits
            univ_bias = (1 - attn_mask) * univ_expert_mask * self.tau
            # 合并调整后的 patch_logits
            adjusted_patch_logits = patch_logits + core_bias + univ_bias  # [B,128,E,S]
            # 重新组合CLS与调整后的小块logits
            adjusted_logits = torch.cat([cls_logits, adjusted_patch_logits], dim=1)  # [B,129,E,S]       
                    # 后续保持原逻辑
            dispatch_logits = adjusted_logits / self.scales[0]        
        
        if self.add_noise:
            noise = normal_noise(logits) * self.noise_mult
            # logits = logits + noise
            dispatch_logits = dispatch_logits + noise
            combine_logits = combine_logits + noise
            # logits = self.noise_generator(logits)

        if self.moe_logits_drop > 0.0:
            mask = torch.ones_like(logits) * self.moe_logits_drop
            mask = torch.bernoulli(mask) * -1e12
            logits = logits + mask
        
        # dispatch_weights = dispatch_logits.softmax(dim = -3)
        # combine_weights = combine_logits.flatten(start_dim=-2).softmax(dim = -1)

        dispatch_weights = softmax(dispatch_logits, dim=-3)
        combine_weights = entmax15(
                combine_logits.flatten(start_dim=-2), dim=-1)

        # dispatch_weights = self.moe_logits_drop(dispatch_weights)
        # combine_weights = self.moe_logits_drop(combine_weights)

        slots = torch.einsum('... n d, ... n e s -> ... e s d', x, dispatch_weights)

        if self.multi_head:
            slots = rearrange(slots, 'b h e s d -> b e s (h d)')

        core_slots = slots[:, :self.N_core, :, :]
        occ_slots = slots[:, self.N_core:, :, :]

        core_out = self.core_experts(core_slots)
        univ_out = self.occ_experts(occ_slots)

        out = torch.cat((core_out, univ_out), dim=1)

        # Compute output tokens as weighted average of output slots using combine weights
        if self.multi_head:
            out = rearrange(out, ' b e s (h d) -> b h (e s) d', h=self.num_heads)
        else:
            out = rearrange(out, ' b e s d -> b (e s) d')

        out = self.expert_drop(out)
        out = torch.einsum('... s d, ... n s -> ... n d', out, combine_weights)

        if self.multi_head:
            out = rearrange(out, 'b h n d -> b n (h d)', h=self.num_heads)

        if is_conv:
            out = rearrange(out, 'b (h w) d -> b h w d', h=int(out.shape[1]**0.5))
        
        metrics = {'auxiliary_loss': torch.tensor(0.0),
        'combine_weights_similarity_mean': torch.tensor(0.0),
        'dispatch_weights_similarity_mean': torch.tensor(0.0)}

        return out,metrics
    
# 使用通道抑制
class DualPathSoftMoE4(nn.Module):
    """
    A wrapper class to create a Soft Mixture of Experts layer.

    From "From Sparse to Soft Mixtures of Experts"
    https://arxiv.org/pdf/2308.00951.pdf
    """

    def __init__(
        self,
        dim: int,
        num_experts: int,
        slots_per_expert: int,
        add_noise: bool = False,
        noise_mult: float = 1.0,
        moe_droprate: float = 0.0,
        moe_droprate_act: Optional[float] = None,
        moe_logits_drop: float = 0.0,
        moe_drop_path_rate: float = 0.0,
        compress_ratio: float = 1.0,
        phi = None,
        key_proj: bool = True,
        query_proj: bool = True,
        input_as_phi: bool = False,
        slot_layernorm: bool = True,
        multi_head: bool = False,
        channel_groups: int = 4,
        tau = 1.0,
        **kwargs,
    ) -> None:
        """
        Args:
            dim (int): Dimensionality of input features.
            num_experts (int): Number of experts.
            slots_per_expert (int): Number of token slots per expert.
            layer (Callable): Network layer of the experts.
            normalize (bool): Normalize input and phi (sec. 2.3 from paper)
            **layer_kwargs: Additional keyword arguments for the layer class.
        """
        super().__init__()

        self.dim = dim
        self.num_experts = num_experts
        self.slots_per_expert = slots_per_expert
        self.add_noise = add_noise
        self.noise_mult = noise_mult

        self.occ_factor = kwargs.get('occ_factor', 1/4)
        self.N_core = kwargs.get('core_experts', int(self.num_experts / 2))
        self.N_univ = kwargs.get('occ_experts', int(self.N_core / self.occ_factor))
        logit_scale = kwargs.get('logit_scale', 1.0)

        self.multi_head = multi_head
        if self.multi_head:
            self.num_heads = kwargs.get('num_heads', 3)
            self.head_dim = dim // self.num_heads
        self.channel_groups = channel_groups
        self.dim_per_group = dim // channel_groups
        # 通道感知器：预测每组通道的遮挡相关性分数
        self.channel_scorer = nn.Sequential(
            nn.Linear(self.dim_per_group, 1),
            nn.Sigmoid()  # 输出分数在[0,1]之间
        )        
        # Initialize phi and normalization scaling factor
        # self.register_parameter('phi' , nn.Parameter(torch.zeros(num_experts, slots_per_expert, dim)))
        self.input_as_phi = input_as_phi
        # self.num_experts = self.N_core + self.N_univ
        self.slots_per_expert = slots_per_expert
        if phi is None and not input_as_phi:
            
            # self.phi = nn.Parameter(torch.zeros((self.N_core + self.N_univ), slots_per_expert, dim))
            # trunc_normal_(self.phi, std=0.02)
            query_init = self.initialize_query((self.N_core + self.N_univ), slots_per_expert, dim, pattern_type='orthogonal')
            self.phi = nn.Parameter(query_init)

        elif not input_as_phi:
            self.phi = phi

        if slot_layernorm:
            # self.norm = nn.Identity()
            self.norm = l2norm
            self.slot_norm = nn.LayerNorm(dim) if not self.multi_head else nn.LayerNorm(self.head_dim)
        else:
            self.norm = l2norm
            self.slot_norm = nn.Identity()
        # self.scale = nn.Parameter(torch.tensor(1.0) * logit_scale)
        self.scales = nn.ParameterList([
            nn.Parameter(torch.tensor(0.5) * logit_scale),
            nn.Parameter(torch.tensor(0.5) * logit_scale)
        ])
        
        self.key_proj = key_proj
        if self.key_proj:
            self.phi_key_proj = OffsetScale(dim) if not self.multi_head else OffsetScale(self.head_dim)
            # self.phi_key_proj = nn.Linear(dim, dim)

        else:
            self.phi_key_proj = nn.Identity()

        self.query_proj = query_proj
        if self.query_proj:
            self.phi_query_proj = OffsetScale(dim)
            # self.phi_query_proj = nn.Linear(dim, dim)
        else:
            self.phi_query_proj = nn.Identity()
        self.tau = nn.Parameter(torch.tensor(tau))
        self.moe_logits_drop = moe_logits_drop
        # self.moe_logits_drop = nn.Dropout(moe_logits_drop) if moe_logits_drop > 0.0 else nn.Identity()
        self.expert_drop = DropPath(moe_drop_path_rate) if moe_drop_path_rate > 0. else nn.Identity()
        self.core_experts = MultiExpertLayer(in_dim=dim, hidden_dim=int(4 * compress_ratio * dim), num_experts=self.N_core, moe_droprate=moe_droprate, moe_droprate_act=moe_droprate_act)
        self.occ_experts = MultiExpertLayer(in_dim=dim, hidden_dim=int(4 * self.occ_factor * compress_ratio * dim), num_experts=self.N_univ, moe_droprate=moe_droprate, moe_droprate_act=moe_droprate_act)

        # self.noise_generator = LearnableNormalNoiseGenerator(self.num_experts)

    def initialize_query(self, num_experts, slots_per_expert, d_model, pattern_type='orthogonal'):
        query_init = torch.randn(num_experts, slots_per_expert, d_model)
        if pattern_type == 'orthogonal':
            nn.init.orthogonal_(query_init)
        elif pattern_type == 'trunc_normal':
            trunc_normal_(query_init, mean=0.0, std=0.02)  # 截断正态分布初始化
        else:
            raise ValueError(f"Unsupported pattern_type: {pattern_type}")

        # pos_encoding = torch.arange(0, d_model).unsqueeze(0).repeat(num_experts, 1).float()
        # query_init += 0.01 * pos_encoding

        return query_init

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        assert (
            x.shape[-1] == self.dim
        ), f"Input feature dim of {x.shape[-1]} does not match layer dim of {self.dim}"

        if len(x.shape) == 4:
            is_conv = True
            x = rearrange(x, 'b h w d -> b (h w) d')
        else:
            is_conv = False
            assert (len(x.shape) == 3), f"Input expected to have 3 or 4 dimensions but has {len(x.shape)}"

        if not self.input_as_phi:
            query = self.phi
        else:
            query = adjust_query_length(x, self.num_experts*self.slots_per_expert)
            query = rearrange(query, 'b (e s) d -> b e s d', e=int(self.num_experts))

        # Normalize input and phi
        if self.multi_head:
            x = rearrange(x, 'b n (h d) -> b h n d', h=self.num_heads)
            query = rearrange(query, 'e s (h d) -> h e s d', h=self.num_heads)
        B, N, D = x.shape
        x_group = x.view(B, N, self.channel_groups, self.dim_per_group)
        print(f"x_group:{x_group.shape}")
        # 计算每组通道的遮挡相关性分数 [B, N, Groups]
        channel_scores = self.channel_scorer(x_group).squeeze(-1)
        key = self.norm(self.phi_key_proj(x))
        query = self.slot_norm(self.phi_query_proj(query))
        query = l2norm(query)
        # query = query * self.scale

        # Compute dispatch and combine weights
        if self.multi_head:
            logits = torch.einsum("b h n d, h e s d -> b h n e s", key, query)
        else: 
            if not self.input_as_phi:
                logits = torch.einsum("b n d, e s d->b n e s", key, query)
            else:
                logits = torch.einsum("b n d, b e s d->b n e s", key, query)
                logits = logits / (self.dim ** 0.5)
        experts_per_group = self.num_experts // self.channel_groups
        print(f"experts_per_group:{experts_per_group}")
            # 将logits从 [B, N, E, S] 重组为 [B, N, G, EG, S]
        # === 通道感知路由调整 ===
        if self.multi_head:
            # 多头模式：扩展维度匹配
            channel_bias = channel_scores.unsqueeze(1) * self.tau  # [B,1,N,G] * tau
            logits = logits + channel_bias.unsqueeze(-1)  # [B,H,N,E,S]
        else:
            # 单头模式：专家掩码应用
            core_mask = (channel_scores < 0.5).float()
            core_mask = core_mask.unsqueeze(-1).repeat(1, 1, 1, experts_per_group)  # [B,N,G,16]
            core_mask = core_mask.view(B, N, self.num_experts, 1)  # [B,N,E=64,1]
            print("core_mask:", core_mask.shape)
            univ_mask = (channel_scores >= 0.5).float()
            univ_mask = univ_mask.unsqueeze(-1).repeat(1, 1, 1, experts_per_group)
            univ_mask = univ_mask.view(B, N, self.num_experts, 1)

            channel_bias = channel_scores.unsqueeze(-1).repeat(1, 1, 1, experts_per_group)
            channel_bias = channel_bias.view(B, N, self.num_experts, 1) * self.tau
            
            # 创建专家掩码
            # experts_per_group = self.num_experts // self.channel_groups
            core_expert_mask = torch.zeros(1, 1, self.channel_groups, experts_per_group, 1, device=x.device)
            core_expert_mask[..., :self.N_core//self.channel_groups, :] = 1.0
            core_expert_mask = core_expert_mask.view(1, 1, self.num_experts, 1)
            print("core_expert_mask:", core_expert_mask.shape) 
            univ_expert_mask = 1 - core_expert_mask
            print(f"logits:{logits.shape}")

            # 偏置注入
            logits = logits + (core_mask * core_expert_mask * channel_bias) \
                            + (univ_mask * univ_expert_mask * channel_bias)
        logits = logits.view(B, N, self.channel_groups, experts_per_group, -1)  # -1表示S维度
        print(f"Reshaped logits: {logits.shape}")  # 应输出 [64, 129, 4, 16, 1]            
        # noised dispatch and combine gate logits, with annealing if needed
        dispatch_logits = logits / self.scales[0]
        combine_logits = logits / self.scales[1]
        
        if self.add_noise:
            noise = normal_noise(logits) * self.noise_mult
            # logits = logits + noise
            dispatch_logits = dispatch_logits + noise
            combine_logits = combine_logits + noise
            # logits = self.noise_generator(logits)

        if self.moe_logits_drop > 0.0:
            mask = torch.ones_like(logits) * self.moe_logits_drop
            mask = torch.bernoulli(mask) * -1e12
            logits = logits + mask
        
        # dispatch_weights = dispatch_logits.softmax(dim = -3)
        # combine_weights = combine_logits.flatten(start_dim=-2).softmax(dim = -1)
        
        dispatch_weights = softmax(dispatch_logits, dim=-3)
        print(f"dispatch_weights:{dispatch_weights.shape}") # dispatch_weights:torch.Size([64, 129, 4, 16, 1])
        print(f"combine_logits:{combine_logits.shape}")
        # combine_logits_merged = combine_logits.flatten(start_dim=-3)
        # print(f"combine_logits_merged:{combine_logits_merged.shape}")
        # combine_weights = entmax15(combine_logits_merged, dim=-1)
        # print(f"combine_weights :{combine_weights.shape}")
        # combine_weights = entmax15(
        #         combine_logits.flatten(start_dim=-2), dim=-1)

        # dispatch_weights = self.moe_logits_drop(dispatch_weights)
        # combine_weights = self.moe_logits_drop(combine_weights)
        # 分通道组生成slots
        slots = []
        for g in range(self.channel_groups):
            print(f"g:{g}")
            x_group_g = x_group[..., g, :]
            print(f"x_group_g:{x_group_g.shape}")
            dispatch_weights_g = dispatch_weights[:, :, g, :, :]
            slots_g = torch.einsum('bnd,bnes->besd', x_group_g, dispatch_weights_g)
            print(f"当前slots_g：{slots_g.shape}")
            slots.append(slots_g)
     
        # === 合并通道组和专家维度 ===
        slots = torch.stack(slots, dim=3)          # [B=64, EG=16, S=1, G=4, D/G=192]
        print(f"slots:{slots.shape}")
        slots = slots.permute(0, 3, 1, 2, 4)       # [B=64, G=4, EG=16, S=1, D/G=192]
        slots = slots.reshape(B, self.num_experts, -1, self.dim_per_group)  # [B=64, E=64, S=1, D/G=192]

        # === 显式添加通道组维度（无需改变元素总数） ===
        # 直接定义最终形状，无需额外 view 操作
        slots = slots.unsqueeze(3)                 # [B=64, E=64, S=1, G=1, D/G=192]
        slots = slots.expand(-1, -1, -1, self.channel_groups, -1)  # [B=64, E=64, S=1, G=4, D/G=192]
        print(f"Final slots shape: {slots.shape}")  # 输出 [64, 64, 1, 4, 192]
        # print(f"Final slots shape: {slots.shape}")  

        # slots = torch.einsum('... n d, ... n e s -> ... e s d', x, dispatch_weights)

        # if self.multi_head:
        #     slots = rearrange(slots, 'b h e s d -> b e s (h d)')
        # 分拆核心/通用专家
        core_slots = slots[:, :self.N_core, ...]   # [B,Core,S,G,D/G]
        print(f"core_slots:{core_slots.shape}")
        occ_slots = slots[:, self.N_core:, ...]   # [B,Univ,S,G,D/G]
        # core_slots = slots[:, :self.N_core, :, :]
        # occ_slots = slots[:, self.N_core:, :, :]
        # 通道组维度合并 [B,E,S,G,D/G] -> [B,E,S,G*D/G]
        core_slots = core_slots.reshape(*core_slots.shape[:-2], -1)
        occ_slots = occ_slots.reshape(*occ_slots.shape[:-2], -1)
        core_out = self.core_experts(core_slots)
        print(f"core_out shape: {core_out.shape}") 
        univ_out = self.occ_experts(occ_slots)
        print(f"univ_out shape: {univ_out.shape}")
        # 动态通道融合
        # 生成通道权重（与专家维度对齐）
        # channel_weights = channel_scores.unsqueeze(-1).repeat(1, 1, 1, experts_per_group)  # [B, N, G, 16]
        # channel_weights = channel_weights.view(B, N, self.num_experts, 1)  # [B, N, E=64, 1]

        # # 分割核心/通用专家权重
        # # 动态通道融合部分修正
        # core_weights = 1 - channel_scores.unsqueeze(-1).repeat(1, 1, 1, self.N_core // self.channel_groups)
        # core_weights = core_weights.view(B, 1, self.N_core, self.slots_per_expert)  # [B, 1, Core, S]
        # univ_weights = channel_scores.unsqueeze(-1).repeat(1, 1, 1, self.N_univ // self.channel_groups)
        # univ_weights = univ_weights.view(B, 1, self.N_univ, self.slots_per_expert)  # [B, 1, Univ, S]
        group_scores = channel_scores.mean(dim=1)  # [B, 4]

        # 扩展核心专家权重：每个通道组对应 N_core//G 个专家
        experts_per_group = self.N_core // self.channel_groups  # 32//4=8
        core_group_scores = group_scores.unsqueeze(2).repeat(1, 1, experts_per_group)  # [B,4,8]
        core_group_scores = core_group_scores.view(B, self.N_core)  # [B,32]
        core_weights = (1 - core_group_scores).unsqueeze(1).unsqueeze(3)  # [B,1,32,1]

        # 扩展通用专家权重同理
        univ_group_scores = group_scores.unsqueeze(2).repeat(1, 1, self.N_univ//self.channel_groups)
        univ_group_scores = univ_group_scores.view(B, self.N_univ)
        univ_weights = univ_group_scores.unsqueeze(1).unsqueeze(3)  # [B,1,32,1]
        print(f"core_weights shape: {core_weights.shape}")
        print(f"univ_weights shape: {univ_weights.shape}")
        # 修改后（正确）
        # 加权融合
        out = torch.cat([
            core_out * core_weights.permute(0,2,1,3),  # [B, Core=32, N, 192]
            univ_out * univ_weights.permute(0,2,1,3)   # [B, Univ=32, N, 192]
        ], dim=1)  
        print(f"out shape: {out.shape}")  # torch.Size([64, 64, 32, 768])
        # out = torch.cat((core_out, univ_out), dim=1)

        # Compute output tokens as weighted average of output slots using combine weights
        if self.multi_head:
            out = rearrange(out, ' b e s (h d) -> b h (e s) d', h=self.num_heads)
        else:
            out = rearrange(out, ' b e s d -> b (e s) d')
            # out = rearrange(out, 'b e s d -> b e d')
        # ==== combine_weights 生成修正 ====
        combine_logits_merged = combine_logits.flatten(start_dim=-3)  # [64,129,4*16*1=64]
        combine_weights = entmax15(combine_logits_merged, dim=-1)     # [64,129,64]
        out = self.expert_drop(out)
        print(f"out:{out.shape}")
        print(f"combine_weights:{combine_weights.shape}")

        # out = torch.einsum('... s d, ... n s -> ... n d', out, combine_weights)
        out = torch.einsum('b s d, b n s -> b n d', out, combine_weights)  # 显式指定维度

        if self.multi_head:
            out = rearrange(out, 'b h n d -> b n (h d)', h=self.num_heads)

        if is_conv:
            out = rearrange(out, 'b (h w) d -> b h w d', h=int(out.shape[1]**0.5))
        
        metrics = {'auxiliary_loss': torch.tensor(0.0),
        'combine_weights_similarity_mean': torch.tensor(0.0),
        'dispatch_weights_similarity_mean': torch.tensor(0.0)}

        return out,metrics
    # 通道分数热力图
    # plt.figure(figsize=(12,6))
    # plt.subplot(121)
    # plt.title("Channel Scores (Group 0)")
    # plt.imshow(channel_scores[0,:,0].detach().cpu().numpy())
    # plt.subplot(122)
    # plt.title("Channel Scores (Group 3)")
    # plt.imshow(channel_scores[0,:,3].detach().cpu().numpy())