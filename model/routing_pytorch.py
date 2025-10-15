import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Any, Mapping, Optional, Tuple
KwArgs = Mapping[str, Any]
Metrics = Mapping[str, torch.Tensor]
from .moe_pytorch import BaseDispatcher,TopExpertsPerItemDispatcher,get_top_experts_per_item_dispatcher
import math
class NoisyTopExpertsPerItemRouter(nn.Module):
    """Noisy TopExpertsPerItem router used in https://arxiv.org/abs/2106.05974."""

    def __init__(self, num_experts: int, num_selected_experts: int = 1, noise_std: float = 1.0,
                 gshard_loss_weight: float = 0.0, importance_loss_weight: float = 1.0,
                 load_loss_weight: float = 1.0, dispatcher: Optional[KwArgs] = None,
                 deterministic: bool = False, dtype: Optional[torch.dtype] = None):
        super().__init__()
        # print(f"传入NoisyTopExpertsPerItemRouter的num_experts：{num_experts}")
        # print(f"传入router的noise_std：{noise_std}")
        self.num_experts = num_experts
        self.num_selected_experts = num_selected_experts
        self.noise_std = noise_std
        self.gshard_loss_weight = gshard_loss_weight
        self.importance_loss_weight = importance_loss_weight
        self.load_loss_weight = load_loss_weight
        self.dispatcher = dispatcher
        self.deterministic = deterministic
        self.dtype = dtype


    def forward(self, inputs: torch.Tensor) -> Tuple[BaseDispatcher, Metrics]:
        gates_softmax, metrics = self._compute_gates_softmax_and_metrics(inputs, self.num_experts)
        dispatcher = self._create_dispatcher(gates_softmax)
        return dispatcher, metrics



    def _compute_gates_softmax_and_metrics(self, inputs: torch.Tensor, num_experts: int) -> Tuple[torch.Tensor, Metrics]:
        if inputs.dim() != 3:
            raise ValueError(f"inputs.dim() must be 3, but it is {inputs.dim()}")
        if not num_experts >= self.num_selected_experts >= 1:
            raise ValueError(f"num_experts >= num_selected_experts >= 1, but got "
                             f"num_experts = {num_experts} and num_selected_experts = {self.num_selected_experts}.")
        assert not torch.isnan(inputs).any(), "Inputs contain NaN!"
        assert not torch.isinf(inputs).any(), "Inputs contain Inf!"
        device = inputs.device
        inputs = nn.LayerNorm(inputs.shape[-1]).to(inputs.device)(inputs)
        # Compute the gating logits for each pair of (item, expert).
        gate_layer = nn.Linear(inputs.shape[-1], num_experts, bias=False).to(device)
        nn.init.xavier_normal_(gate_layer.weight, gain=0.01)
        gates_logits = gate_layer(inputs)
        print(f"gates_logits: max={gates_logits.max().item():.2f}, min={gates_logits.min().item():.2f}")
        print(f"[Debug] gates_logits: mean={gates_logits.mean().item():.4f}, std={gates_logits.std().item():.4f}, min={gates_logits.min().item():.4f}, max={gates_logits.max().item():.4f}")

        # Compute the softmax and auxiliary losses.
        gates_logits = gates_logits - gates_logits.max(dim=-1, keepdim=True).values  # 防止指数溢出
        gates_softmax = F.softmax(gates_logits, dim=-1)
        # importance_loss = torch.mean(self._importance_auxiliary_loss(gates_softmax))
        importance_loss = torch.stack([self._importance_auxiliary_loss(g) for g in gates_softmax.unbind(dim=0)]).mean()
        if self.deterministic or self.noise_std == 0.0:
            gshard_loss = self._gshard_auxiliary_loss(gates_softmax)
            metrics = {
                "auxiliary_loss": self._weighted_sum(
                    (self.gshard_loss_weight, gshard_loss),
                    (self.importance_loss_weight, importance_loss)),
                "gshard_loss": gshard_loss,
                "importance_loss": importance_loss,
            }
            return gates_softmax, metrics
        else:
            # noise_std = (1.0 / num_experts) * self.noise_std
            noise_std = max((1.0 / num_experts) * self.noise_std, 1e-6)
            logits_noise = noise_std * torch.randn_like(gates_logits)
            print(f"[Debug] logits_noise: mean={logits_noise.mean().item():.4f}, std={logits_noise.std().item():.4f}")
            gates_logits_noisy = gates_logits + logits_noise
            gates_softmax_noisy = F.softmax(gates_logits_noisy, dim=-1)

            load_loss = self._load_auxiliary_loss(gates_logits, gates_logits_noisy, noise_std)
            gshard_loss = self._gshard_auxiliary_loss(gates_softmax_noisy)

            metrics = {
                "auxiliary_loss": self._weighted_sum(
                    (self.gshard_loss_weight, gshard_loss),
                    (self.importance_loss_weight, importance_loss),
                    (self.load_loss_weight, load_loss)),
                "gshard_loss": gshard_loss,
                "importance_loss": importance_loss,
                "load_loss": load_loss,
            }
            return gates_softmax_noisy, metrics
  
    def _create_dispatcher(self, gates_dispatch: torch.Tensor) -> BaseDispatcher:
        # 使用门控概率创建具体的 Dispatcher
        return TopExpertsPerItemDispatcher(
            gates_dispatch,
            num_selected_experts=self.num_selected_experts
        )
 
    def _gshard_auxiliary_loss(self, gates: torch.Tensor) -> torch.Tensor:
        """
        计算 GShard 辅助损失，确保输入 gates 的维度为 (num_groups, group_size, num_experts)
        """
        # 合并前两个维度 (num_groups * group_size, num_experts)
        gates_flat = gates.reshape(-1, gates.size(-1))  # (394*4, 8)
        print(f"gates_flat shape: {gates_flat.shape}")  # 应输出 (1576, 8)

        # 计算每个专家的平均门控概率 (num_experts,)
        # mean_gates_per_expert = gates_flat.mean(dim=0)  # shape: (8,)
        num_experts = gates_flat.size(-1)
        mean_gates_per_expert = gates_flat.mean(dim=0)
        # 计算每个样本的 top-1 专家索引 (num_groups * group_size,)
        top1_indices = gates_flat.argmax(dim=1)  # shape: (1576,)

        # 生成 one-hot 编码 (num_groups * group_size, num_experts)
        one_hot = F.one_hot(top1_indices, num_classes=num_experts).float()  # (1576, 8)

        # 计算每个专家的平均 top-1 分配概率 (num_experts,)
        mean_top1_per_expert = one_hot.mean(dim=0)  # shape: (8,)

        # 计算损失
        auxiliary_loss = torch.mean(mean_top1_per_expert * mean_gates_per_expert)
        auxiliary_loss *= self.num_experts ** 2
        # print(f"mean_gates_per_expert shape: {mean_gates_per_expert.shape}")  # (8,)
        # print(f"mean_top1_per_expert shape: {mean_top1_per_expert.shape}")    # (8,)
        return auxiliary_loss

    def _weighted_sum(self, *weighted_values: Tuple[float, torch.Tensor]) -> torch.Tensor:
        """计算加权损失和。"""
        total = 0.0
        for weight, value in weighted_values:
            total += weight * value
        return total
 
    def _importance_auxiliary_loss(self, gates: torch.Tensor) -> torch.Tensor:
        # 合并所有样本维度
        gates_flat = gates.reshape(-1, gates.size(-1))  # (total_samples, num_experts)
        # 计算每个专家的总重要性
        importance_per_expert = gates_flat.sum(dim=0)    # (num_experts,)
        # 计算标准差和均值
        std_importance = importance_per_expert.std()
        mean_importance = importance_per_expert.mean()
        return (std_importance / mean_importance) ** 2

    def _load_auxiliary_loss(self, logits: torch.Tensor, logits_noisy: torch.Tensor, noise_std: torch.Tensor) -> torch.Tensor:
        num_experts = logits_noisy.shape[-1]

        # 计算 top-k 中的最小值索引，即第 num_selected_experts 大的值的索引
        topk_values, topk_indices = logits_noisy.topk(self.num_selected_experts, dim=-1)
        threshold_per_item_index = topk_indices[..., -1].unsqueeze(-1)  # 形状匹配

        # 通过 one-hot 方式取出对应的 threshold_per_item
        one_hot = torch.zeros_like(logits_noisy).scatter_(-1, threshold_per_item_index, 1)
        threshold_per_item = (one_hot * logits_noisy).sum(dim=-1, keepdim=True)

        # 计算 "需要超过的值"
        noise_required_to_win = (threshold_per_item - logits) / noise_std
        noise_required_to_win = torch.clamp(noise_required_to_win, min=-10, max=10)
        print(f"noise_required_to_win: max={noise_required_to_win.max().item():.2f}, min={noise_required_to_win.min().item():.2f}")
        # 计算概率
        # p = 1 - torch.distributions.Normal(0, 1).cdf(noise_required_to_win)
        # 数值稳定的 CDF 计算
        p = 0.5 * (1 + torch.erf(noise_required_to_win / math.sqrt(2)))
        p = torch.where(noise_required_to_win > 10, 1.0, p)
        p = torch.where(noise_required_to_win < -10, 0.0, p)
        # 计算每个专家的平均值
        p_mean = p.mean(dim=0)
        # 新增两行
        std = p_mean.std(unbiased=False)
        mean = p_mean.mean()
        # 计算 variation coefficient squared
        # return (p_mean.std() / p_mean.mean()) ** 2
        return (std / mean) ** 2

class Bfloat16Dispatcher(BaseDispatcher):
    def __init__(self, dispatcher: BaseDispatcher):
        super().__init__()
        self.dispatcher = dispatcher

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        return self.dispatcher(x.to(torch.bfloat16))  

class NoisyTopItemsPerExpertRouter(nn.Module):
    """Noisy TopItemsPerExpert路由器。

    与其为每个项选择Top-K专家，并忽略超过任何给定专家容量(C)的选择，我们在此选择每个专家的Top-C项，按得分排序。

    这使得各个专家之间的负载自动平衡，然而分配给每个项的专家数目没有界限并且可能变化。有些项可能不会路由到任何专家。尽管如此，在实践中，这种方法效果非常好。

    该方法在论文 https://arxiv.org/abs/2202.09368 中被称为“专家选择路由”（Experts Choice Routing）。
    """
    def __init__(self, num_experts: int, noise_std: float = 1.0, deterministic: bool = False, dtype: torch.dtype = None):
        super(NoisyTopItemsPerExpertRouter, self).__init__()
        self.num_experts = num_experts
        self.noise_std = noise_std
        self.deterministic = deterministic
        self.dtype = dtype

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        gates_softmax = self._compute_gates_softmax(inputs)
        dispatcher, metrics = self._create_dispatcher_and_metrics(gates_softmax)
        metrics["auxiliary_loss"] = 0.0  # 设置辅助损失为0
        return dispatcher, metrics  # 返回分配器和指标（metrics）
    def _weighted_sum(self, *weighted_values: Tuple[float, torch.Tensor]) -> torch.Tensor:
        """计算加权损失和。"""
        total = 0.0
        for weight, value in weighted_values:
            total += weight * value
        return total
    def _compute_gates_softmax(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndimension() != 3:
            raise ValueError(f"inputs.ndimension() 必须是 3，但实际是 {inputs.ndimension()}")
        dtype = self.dtype or inputs.dtype
        device = inputs.device
        # 计算每一对（item，expert）的门控logits
        gate_layer = nn.Linear(inputs.shape[-1], self.num_experts, bias=False).to(device)
        gates_logits = gate_layer(inputs)

        if self.deterministic or self.noise_std == 0.0:
            gates_softmax = F.softmax(gates_logits, dim=-1)  # 对logits进行softmax
            return gates_softmax
        else:
            noise_std = (1.0 / self.num_experts) * self.noise_std  # 噪声标准差
            logits_noise = noise_std * torch.randn_like(gates_logits)  # 生成噪声
            gates_logits_noisy = gates_logits + logits_noise  # 将噪声加入到logits中
            gates_softmax_noisy = F.softmax(gates_logits_noisy, dim=-1)  # 对带噪声的logits进行softmax
            return gates_softmax_noisy

    def _create_dispatcher_and_metrics(self, gates_dispatch):
        # 由于我们没有完整的 `get_top_items_per_expert_dispatcher`，我们假设该函数返回两个值
        # 此处为示例代码，实际应根据项目需求进一步实现具体的分配器
        dispatcher = gates_dispatch.argmax(dim=-1)  # 假设获取最大得分的分配
        metrics = {"dispatcher_output": dispatcher}
        return dispatcher, metrics

# def _weighted_sum(*args):
#     """返回加权和 [（weight, element），...]，仅对权重大于0的元素进行计算。"""
#     # 注意：某些损失函数在某些情境下可能是未定义的（例如，可能会有inf/NaN的梯度），
#     # 在这种情况下，我们通过将它们的权重设置为0来避免它们影响总辅助损失。
#     return sum(x * w for w, x in args if w > 0)  # 计算加权和

