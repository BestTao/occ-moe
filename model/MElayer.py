import math
from typing import Optional, Union

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class LayerScale(nn.Module):
    """层缩放组件 - 功能与OffsetScale相同但实现不同"""
    def __init__(self, dim):
        super().__init__()
        self.scaling_param = nn.Parameter(torch.ones(dim))
        self.shifting_param = nn.Parameter(torch.zeros(dim))
        
    def forward(self, x):
        return x * self.scaling_param + self.shifting_param


class ExpertSpecificNorm(nn.Module):
    """专家专用归一化层 - 功能与BlockExpert_LayerNorm相同"""
    def __init__(self, num_experts, dim):
        super().__init__()
        self.scale_params = nn.Parameter(torch.ones(num_experts, dim))
        self.offset_params = nn.Parameter(torch.zeros(num_experts, dim))
        self.epsilon = 1e-6
    
    def forward(self, x):
        # 计算统计量
        mu = x.mean(dim=-1, keepdim=True)
        sigma = x.std(dim=-1, keepdim=True, unbiased=False)
        
        # 归一化并应用专家特定参数
        normalized = (x - mu) / (sigma + self.epsilon)
        scale = self.scale_params.view(1, -1, 1, self.scale_params.size(-1))
        offset = self.offset_params.view(1, -1, 1, self.offset_params.size(-1))
        
        return normalized * scale + offset


class BatchedExpertLinear(nn.Module):
    """批量专家线性层 - 功能与MultiExpertLinear相同"""
    def __init__(self, num_experts, input_dim, output_dim, use_bias=True):
        super().__init__()
        self.num_experts = num_experts
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # 权重参数: (experts, output_dim, input_dim)
        self.weight_matrix = nn.Parameter(
            torch.Tensor(num_experts, output_dim, input_dim)
        )
        
        self.has_bias = use_bias
        if self.has_bias:
            self.bias_vector = nn.Parameter(torch.Tensor(num_experts, output_dim))
            torch.nn.init.constant_(self.bias_vector, 0.0)
        else:
            self.register_parameter('bias_vector', None)
            
        # 初始化权重
        torch.nn.init.kaiming_uniform_(self.weight_matrix, a=math.sqrt(5))
        
    def forward(self, x):
        # 使用矩阵乘法实现线性变换
        # x: (batch, experts, seq, input_dim)
        # weight: (experts, output_dim, input_dim)
        # 输出: (batch, experts, seq, output_dim)
        output = torch.einsum('besi,eo i->beso', x, self.weight_matrix)
        
        if self.has_bias:
            bias = self.bias_vector.unsqueeze(0).unsqueeze(2)  # (1, experts, 1, output_dim)
            output = output + bias
            
        return output


class MultiExpertLayer(nn.Module):
    """混合专家层 - 保持原有接口但内部实现不同"""
    
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        num_experts: int,
        moe_droprate: float = 0.0,
        moe_droprate_act: Optional[float] = None,
        act_fn: nn.Module = nn.GELU,
        bias: bool = True,
        device: Optional[Union[torch.device, str]] = None,
        dtype: Optional[torch.dtype] = None,
        norm_layer=None,
        layer_scale: bool = False,
        glu: bool = True,  # 保留参数但忽略GLU模式
        freeze_moe: bool = False,
    ):
        super().__init__()
        
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        
        # 归一化层配置
        self.norm_layer = norm_layer(hidden_dim) if norm_layer is not None else nn.Identity()
        
        # 专家MLP组件 - 只使用标准MLP路径
        self.expert_fc1 = BatchedExpertLinear(num_experts, in_dim, hidden_dim, bias)
        self.activation = act_fn()
        self.expert_fc2 = BatchedExpertLinear(num_experts, hidden_dim, in_dim, bias)
        
        # 层缩放配置
        self.layer_scale_enabled = layer_scale
        self.input_scale = LayerScale(in_dim) if layer_scale else nn.Identity()
        self.output_scale = LayerScale(in_dim) if layer_scale else nn.Identity()
        
        # Dropout配置
        self.dropout1 = nn.Dropout(moe_droprate_act if moe_droprate_act is not None else moe_droprate)
        self.dropout2 = nn.Dropout(moe_droprate)
        
        # 冻结配置
        self.freeze_moe = freeze_moe
        if freeze_moe:
            self._disable_expert_gradients()
    
    def _disable_expert_gradients(self):
        """冻结专家参数"""
        self.expert_fc1.weight_matrix.requires_grad = False
        self.expert_fc2.weight_matrix.requires_grad = False
        
        if self.expert_fc1.has_bias:
            self.expert_fc1.bias_vector.requires_grad = False
            self.expert_fc2.bias_vector.requires_grad = False
    
    def forward(self, x: Tensor) -> Tensor:
        # 输入验证
        if x.shape[-1] != self.in_dim:
            raise ValueError(
                f"输入维度应为{self.in_dim}，但实际为{x.shape[-1]}"
            )
        
        if x.shape[1] != self.num_experts:
            raise ValueError(
                f"专家数量应为{self.num_experts}，但实际为{x.shape[1]}"
            )
        
        # 输入缩放
        x_scaled = self.input_scale(x)
        
        # 第一层线性变换
        x_transformed = self.expert_fc1(x_scaled)
        
        # 激活函数
        x_activated = self.activation(x_transformed)
        
        # 第一次dropout
        x_dropped1 = self.dropout1(x_activated)
        
        # 归一化
        x_normalized = self.norm_layer(x_dropped1)
        
        # 第二层线性变换
        x_projected = self.expert_fc2(x_normalized)
        
        # 第二次dropout
        x_dropped2 = self.dropout2(x_projected)
        
        # 输出缩放
        output = self.output_scale(x_dropped2)
        
        return output
