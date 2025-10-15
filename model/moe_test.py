import abc
import torch
from typing import Any, Callable, Dict, Literal, Mapping, Optional, Tuple,Union, Dict
import math
import logging
import torch.nn as nn
import torch.nn.functional as F
# 替代原来的 Array = jnp.ndarray
Array = torch.Tensor
from typing import Callable, Dict, Optional, List
# 替代原来的 CeilOrRound = Literal["ceil", "round"]
CeilOrRound = Literal["ceil", "round"]

# 替代 PartitionSpec 和相关功能
# 在 PyTorch 中，我们可以使用 torch.distributed 或 DataParallel 来实现并行计算
# 例如，torch.distributed需要特定的设置进行分布式训练。
import torch.distributed as dist
# 或者
from torch.nn import DataParallel
def _add_axis_to_metadata(
    module: nn.Module,
    tensor_name: str,
    axis_pos: int,
    axis_name: str
):
    """为 PyTorch 模型的某个张量添加 axis metadata（相当于 Flax 的 _add_axis_to_metadata）

    Args:
        module (nn.Module): 目标 PyTorch 模型
        tensor_name (str): 需要标注的张量名称
        axis_pos (int): 需要标注的轴位置
        axis_name (str): 该轴的名称
    """
    if not hasattr(module, "_axis_metadata"):
        module._axis_metadata = {}

    module._axis_metadata[tensor_name] = (axis_pos, axis_name)

# 如果你有分区约束的需求，可以考虑使用 PyTorch 提供的多GPU训练策略
import torch
import torch.distributed as dist

def with_sharding_constraint(tensor: torch.Tensor, partition_spec: Optional[Tuple[str, ...]]) -> torch.Tensor:
    """
    在 PyTorch 中模拟 JAX 的 `with_sharding_constraint`，确保张量在不同设备之间正确分区。

    Args:
        tensor: 需要 sharding 约束的张量。
        partition_spec: 分区规范（例如 'dp' 表示数据并行，'tp' 表示张量并行）。

    Returns:
        处理后的张量（如果在分布式环境中，可以进行 sharding）。
    """
    if not dist.is_initialized():
        return tensor  # 如果没有初始化分布式，就直接返回
    
    world_size = dist.get_world_size()
    rank = dist.get_rank()

    # 这里假设 partition_spec 是 ('tp',) 代表张量并行 (Tensor Parallelism)
    if partition_spec and 'tp' in partition_spec:
        # 按照 world_size 对数据进行 sharding
        split_size = tensor.shape[0] // world_size
        if tensor.shape[0] % world_size != 0:
            raise ValueError("Tensor shape 不符合张量并行 sharding 规则！")
        
        tensor = tensor[rank * split_size : (rank + 1) * split_size]

    return tensor

class BaseDispatcher(abc.ABC):
    """不同调度器实现的基类。

    调度器负责准备数据，并将其分派给不同的专家，然后将每个专家的输出结果进行合并。在实际硬件上运行时，不同的分派和合并方法会有不同的内存、计算量和运行时影响。

    在所有情况下，分派数据时，它们会接收形状为 (G, S, ...) 的张量。
    每个组（G）会独立地被分派。每组中的项（S）将进入一个专家（E）处理的容量（C）缓冲区。输出是形状为 (E, G * C, ...) 的张量，包含需要由每个专家处理的数据。

    合并数据时，它们会接收形状为 (E, G * C, ...) 的张量，输出形状为 (G, S, ...) 的张量。请注意，合并时的尾部维度（...）可能与分派时的维度不同（例如，如果专家改变了数据的形状）。
    """

    @abc.abstractmethod
    def dispatch(self, data: torch.Tensor) -> torch.Tensor:
        """将数据分派给专家。

        参数：
            data: (G, S, ...) 张量，包含要分派给专家的数据。

        返回：
            (E, G * C, ...) 张量，包含每个专家需要处理的数据。
        """
        pass

    @abc.abstractmethod
    def combine(self, data: torch.Tensor) -> torch.Tensor:
        """将多个专家的输出合并。

        参数：
            data: (E, G * C, ...) 张量，包含来自每个专家的输出数据。

        返回：
            (G, S, ...) 张量，包含每个专家合并后的输出结果。
        """
        pass

class TopExpertsPerItemDispatcher(BaseDispatcher):
    """实现基于门控概率的 Top-k 专家分配逻辑。"""
    def __init__(self, gates: torch.Tensor, num_selected_experts: int = 1):
        super().__init__()
        # print(f"TopExpertsPerItemDispatcher的gates.shape:{gates.shape}")
        self.gates = gates
        self.num_selected_experts = num_selected_experts
        self.num_experts = gates.size(-1)
        # 获取每个token选择的前k个专家索引 [num_groups, group_size, num_selected]
        _, self.expert_indices = torch.topk(gates, k=num_selected_experts, dim=-1)
        self.expert_indices = self._compute_expert_indices(gates)
        # print(f"self.expert_indices:{self.expert_indices.shape}")
        # 收集每个专家对应的token索引
        self.batch_indices = []
        for expert_idx in range(gates.shape[-1]):  # num_experts
            # 找到所有选择当前专家的token位置
            mask = (self.expert_indices == expert_idx)
            self.batch_indices.append(torch.nonzero(mask, as_tuple=True)[0])

    def _compute_expert_indices(self, gates: torch.Tensor) -> torch.Tensor:
        """计算每个样本的前 k 个专家索引。"""
        # gates 形状: (num_groups, group_size, num_experts)
        # 选择每个样本的前 k 个专家
        _, topk_indices = torch.topk(gates, k=self.num_selected_experts, dim=-1)
        return topk_indices  # 形状: (num_groups, group_size, num_selected_experts)

    def dispatch(self, inputs: torch.Tensor) -> torch.Tensor:
        """将输入分发给选定的专家。"""
        # inputs 形状: (num_groups, group_size, hidden_dim)
        # 展开为 (num_groups * group_size, hidden_dim)
        inputs_flat = inputs.reshape(-1, inputs.size(-1))  # (num_groups * group_size, hidden_dim)
        expert_inputs = [inputs_flat[indices] for indices in self.batch_indices]

        return expert_inputs

    def combine(self, expert_outputs: List[torch.Tensor]):
        total_tokens = self.gates.shape[0] * self.gates.shape[1]
        output_dim = expert_outputs[0].shape[-1]
        print(f"Total tokens: {total_tokens}, Output dim: {output_dim}")
        combined = torch.zeros(total_tokens, output_dim,dtype=expert_outputs[0].dtype,device=expert_outputs[0].device)

        # 将每个专家的输出填充到对应位置
        for expert_idx, outputs in enumerate(expert_outputs):
            if outputs is None:
                continue
            # 获取当前专家对应的token索引
            indices = self.batch_indices[expert_idx]
            if indices.numel() > 0:
                combined[indices] = outputs
        # 恢复为原始组结构 [num_groups, group_size, output_dim]
        # original_shape = (*self.gates.shape[:2], -1)
        # combined = combined.reshape(original_shape)
        # print(f"TopExpertsPerItemDispatcher返回的形状：{combined.shape}") # torch.Size([394, 4, 192])
        combined = combined.reshape(self.gates.shape[0], self.gates.shape[1], output_dim)
        print(f"Combined shape: {combined.shape}")  # 应为 [394, 4, 768]
        return combined

from typing import Optional, Tuple

class DenseEinsumDispatcher(BaseDispatcher):
    """Dispatcher using Einsum, dispatching data to all experts.
    
    This is similar to EinsumDispatcher, but with the assumption that C = S.
    
    Attributes:
        combine_weights: (G, S, E) tensor with the combine weights for each item
        (G, S) for each expert (E).
        partition_spec: Optional. Specifies the partitioning strategy for distributed training.
        einsum_precision: Optional. Precision used in all the einsums (e.g.
        combining the outputs of different experts).
    """
    
    def __init__(self, combine_weights: torch.Tensor, partition_spec: Optional[Tuple] = None,
                 einsum_precision: Optional[str] = 'default'):
        self.combine_weights = combine_weights
        self.partition_spec = partition_spec
        self.einsum_precision = einsum_precision  # PyTorch does not have direct precision control like JAX
    
    def dispatch(self, data: torch.Tensor) -> torch.Tensor:
        # Dispatch the data to the experts. Dispatch weights are always 1 (i.e., every item goes to each expert).
        dispatch_weights = torch.ones_like(self.combine_weights, dtype=torch.bool)
        # einsum operation to dispatch
        data = torch.einsum("GSE,GS...->GES...", dispatch_weights, data)
        
        # Handle partitioning if applicable (in PyTorch, partition_spec handling is user-defined).
        return self._dispatch(data)
    
    def combine(self, data: torch.Tensor) -> torch.Tensor:
        """Combines data from experts according to combine_weights."""
        num_groups, _, _ = self.combine_weights.shape
        data = self._receive(data, num_groups)
        
        # Use einsum to combine the data
        return torch.einsum("GSE,GES...->GS...", self.combine_weights, data)

    def _dispatch(self, data: torch.Tensor) -> torch.Tensor:
        """Handles partitioning (if any)."""
        # If partition_spec is used, partition the data accordingly (this is a placeholder for handling partitioning logic)
        if self.partition_spec:
            # Implement custom partitioning logic if needed
            pass
        return data

    def _receive(self, data: torch.Tensor, num_groups: int) -> torch.Tensor:
        """Handles receiving data from the experts (this is a placeholder function)."""
        # You might need custom logic for receiving and restructuring data depending on partition_spec
        return data



class EinsumDispatcher(BaseDispatcher):
    """Dispatcher using Einsum.
    
    Attributes:
        combine_weights: (G, S, E, C) tensor with the combine weights for each item
        (G, S) for each expert (E) and buffer position (C).
        dispatch_weights: Optional. (G, S, E, C) tensor with the dispatch weights of
        each item (G, S) for each expert (E) and buffer position (C).
        partition_spec: Optional. Specifies the partitioning strategy for distributed training.
        einsum_precision: Optional. Precision used in all the einsums (e.g.
        combining the outputs of different experts).
    """
    
    def __init__(self, combine_weights: torch.Tensor, dispatch_weights: Optional[torch.Tensor] = None,
                 partition_spec: Optional[Tuple] = None, einsum_precision: Optional[str] = 'default'):
        self.combine_weights = combine_weights
        self.dispatch_weights = dispatch_weights
        self.partition_spec = partition_spec
        self.einsum_precision = einsum_precision  # PyTorch does not have direct precision control like JAX
    
    def dispatch(self, data: torch.Tensor) -> torch.Tensor:
        # If dispatch_weights is not provided, default to combine_weights > 0
        if self.dispatch_weights is None:
            dispatch_weights = self.combine_weights > 0
        else:
            dispatch_weights = self.dispatch_weights
        
        # einsum operation to dispatch
        data = torch.einsum("GSEC,GS...->GEC...", dispatch_weights, data)
        
        # Handle partitioning if applicable (in PyTorch, partition_spec handling is user-defined).
        return self._dispatch(data)
    
    def combine(self, data: torch.Tensor) -> torch.Tensor:
        """Combines data from experts according to combine_weights."""
        num_groups, _, _, _ = self.combine_weights.shape
        data = self._receive(data, num_groups)
        
        # Use einsum to combine the data
        return torch.einsum("GSEC,GEC...->GS...", self.combine_weights, data)

    def _dispatch(self, data: torch.Tensor) -> torch.Tensor:
        """Handles partitioning (if any)."""
        # If partition_spec is used, partition the data accordingly (this is a placeholder for handling partitioning logic)
        if self.partition_spec:
            # Implement custom partitioning logic if needed
            pass
        return data

    def _receive(self, data: torch.Tensor, num_groups: int) -> torch.Tensor:
        """Handles receiving data from the experts (this is a placeholder function)."""
        # You might need custom logic for receiving and restructuring data depending on partition_spec
        return data

class ExpertIndicesDispatcher(BaseDispatcher):
    """Dispatcher using scatter/gather with (expert, buffer) indices.

    Attributes:
        indices: (G, S, K, 2) tensor with the (expert, buffer) indices of
        each item (G, S) and their K-selected experts. The tuple (expert, buffer)
        for each item is represented in the last dimension (of size 2).
        combine_weights: (G, S, K) tensor with the combine weights of each item
        (G, S) and their K-selected experts.
        num_experts: Number of experts.
        capacity: Capacity of each expert's buffer per group.
        partition_spec: Optional. PartitionSpec used to constrain the sharding of
        the data arrays. By default (None), no sharding constraint is specified.
        einsum_precision: Optional. Precision used in all the einsums (e.g.
        combining the outputs of different experts).
    """
    
    def __init__(self, indices: torch.Tensor, combine_weights: torch.Tensor, num_experts: int, capacity: int,
                 partition_spec: Optional[Tuple] = None, einsum_precision: Optional[str] = 'default'):
        self.indices = indices
        self.combine_weights = combine_weights
        self.num_experts = num_experts
        self.capacity = capacity
        self.partition_spec = partition_spec
        self.einsum_precision = einsum_precision  # PyTorch does not have direct precision control like JAX
    
    def dispatch(self, data: torch.Tensor) -> torch.Tensor:
        num_groups, _, num_selected_experts, _ = self.indices.shape
        _, _, *item_shape = data.shape
        
        # Repeat data to match the number of selected experts
        data = data.repeat(1, num_selected_experts, *[1 for _ in item_shape])
        
        # Reshape indices
        indices = self.indices.view(num_groups, -1, 2)
        
        # Create the shape for the scatter operation
        shape = (self.num_experts, self.capacity, *item_shape)
        
        # Scatter the data based on the indices
        data = self._scatter_nd(indices, data, shape)
        
        return self._dispatch(data)

    def combine(self, data: torch.Tensor) -> torch.Tensor:
        num_groups, _, _ = self.combine_weights.shape
        
        data = self._receive(data, num_groups)
        
        # Gather the data based on the indices
        gathered_data = self._gather_indices(data, self.indices)
        
        # Mask invalid gathered data
        mask = (self.indices[..., 0] < self.num_experts) & (self.indices[..., 1] < self.capacity)
        gathered_data = gathered_data * mask.unsqueeze(-1)
        
        # Weighted sum of the outputs of the K-selected experts for each item
        return torch.einsum("GSK...,GSK->GS...", gathered_data, self.combine_weights)

    def _scatter_nd(self, indices: torch.Tensor, data: torch.Tensor, shape: Tuple[int]) -> torch.Tensor:
        """Scatter operation based on expert and buffer indices."""
        # Initialize a tensor of zeros with the given shape
        output = torch.zeros(shape, dtype=data.dtype, device=data.device)
        
        # Convert indices to a list of tuples (expert, buffer) for scatter operation
        for i in range(indices.shape[0]):
            for j in range(indices.shape[1]):
                expert, buffer = indices[i, j]
                output[expert, buffer] = data[i, j]
        
        return output

    def _gather_indices(self, data: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        """Gather data based on the (expert, buffer) indices."""
        gathered_data = []
        for i in range(indices.shape[0]):
            expert_idx, buffer_idx = indices[i, :, 0], indices[i, :, 1]
            gathered_data.append(data[i, expert_idx, buffer_idx])
        return torch.stack(gathered_data, dim=0)

    def _dispatch(self, data: torch.Tensor) -> torch.Tensor:
        """Handles partitioning (if any)."""
        # If partition_spec is used, partition the data accordingly (this is a placeholder for handling partitioning logic)
        if self.partition_spec:
            # Implement custom partitioning logic if needed
            pass
        return data

    def _receive(self, data: torch.Tensor, num_groups: int) -> torch.Tensor:
        """Handles receiving data from the experts (this is a placeholder function)."""
        # You might need custom logic for receiving and restructuring data depending on partition_spec
        return data



class Bfloat16Dispatcher(BaseDispatcher):
    """Dispatcher wrapper converting data to bfloat16 to save bandwidth."""
    
    def __init__(self, dispatcher: BaseDispatcher):
        self.dispatcher = dispatcher

    def dispatch(self, data: torch.Tensor) -> torch.Tensor:
        dtype = data.dtype
        # Cast data to bfloat16 (if available, else use float16)
        data = self._cast_to_bfloat16(data)
        data = self.dispatcher.dispatch(data)
        return data.to(dtype)

    def combine(self, data: torch.Tensor) -> torch.Tensor:
        dtype = data.dtype
        # Cast data to bfloat16 (if available, else use float16)
        data = self._cast_to_bfloat16(data)
        data = self.dispatcher.combine(data)
        return data.to(dtype)

    def _cast_to_bfloat16(self, data: torch.Tensor) -> torch.Tensor:
        """Converts data to bfloat16 (or float16 if bfloat16 is not available)."""
        if torch.has_bf16:
            return data.to(torch.bfloat16)
        else:
            # Fallback to float16 if bfloat16 is not supported by the hardware
            return data.to(torch.float16)



# def compute_capacity(
#     num_tokens: int,
#     num_experts: int,
#     capacity_factor: float,
#     ceil_or_round: str = "ceil",
#     multiple_of: Optional[int] = 4
# ) -> int:
#     """Returns the capacity per expert needed to distribute num_tokens among num_experts."""
#     if ceil_or_round == "ceil":
#         capacity = math.ceil(num_tokens * capacity_factor / num_experts)
#     elif ceil_or_round == "round":
#         capacity = round(num_tokens * capacity_factor / num_experts)
#     else:
#         raise ValueError(f"Unsupported ceil_or_round value: {ceil_or_round}")
    
#     if capacity < 1:
#         raise ValueError(
#             f"Invalid capacity: {capacity}. It must be >= 1. Given num_tokens={num_tokens}, num_experts={num_experts}, capacity_factor={capacity_factor}."
#         )
    
#     if multiple_of and multiple_of > 0:
#         capacity += (-capacity) % multiple_of
    
#     actual_capacity_factor = capacity * num_experts / num_tokens
#     if abs(actual_capacity_factor - capacity_factor) > 1e-6:
#         logging.warning(
#             "Target capacity_factor is %f, but actual capacity_factor is %f.",
#             capacity_factor, actual_capacity_factor
#         )
#     return capacity

def get_dense_einsum_dispatcher(gates,
                                **dispatcher_kwargs) -> DenseEinsumDispatcher:
  # The dispatching algorithm is trivial, because all tokens are sent to
  # all experts this is coded implicitly in the DenseEinsumDispatcher class.
  return DenseEinsumDispatcher(combine_weights=gates, **dispatcher_kwargs)

def get_top_experts_per_item_dispatcher(
    gates: torch.Tensor,
    name: str,
    num_selected_experts: int,
    batch_priority: bool = False,
    capacity: Optional[int] = None,
    capacity_factor: Optional[float] = 2,
    capacity_ceil_or_round: str = "ceil",
    capacity_multiple_of: Optional[int] = 4,
    **dispatcher_kwargs
):
    """Returns a dispatcher implementing Top-Experts-Per-Item routing."""
    if (capacity is None) == (capacity_factor is None):
        raise ValueError(
            "You must specify either 'capacity' or 'capacity_factor', but not both."
        )
    
    if capacity is None:
        # print(gates.shape)
        group_size, num_experts = gates.shape
        capacity = compute_capacity(
            num_tokens=group_size * num_selected_experts,
            num_experts=num_experts,
            capacity_factor=capacity_factor,
            ceil_or_round=capacity_ceil_or_round,
            multiple_of=capacity_multiple_of,
        )
    
    fn_map = {
        "einsum": _get_top_experts_per_item_einsum_dispatcher,
        "indices": _get_top_experts_per_item_expert_indices_dispatcher,
    }
    if name not in fn_map:
        raise ValueError(f"Unknown dispatcher type: {name}")
    
    return fn_map[name](gates, num_selected_experts, capacity, batch_priority, **dispatcher_kwargs)

# def get_top_items_per_expert_dispatcher(
#     gates: torch.Tensor,
#     name: str,
#     capacity: Optional[int] = None,
#     capacity_factor: Optional[float] = None,
#     capacity_ceil_or_round: str = "ceil",
#     capacity_multiple_of: Optional[int] = 4,
#     **dispatcher_kwargs) -> Tuple[BaseDispatcher, Dict[str, torch.Tensor]]:
#     """Returns a dispatcher implementing Top-Items-Per-Expert routing.

#     Args:
#         gates: (S, E) tensor with the gating values for each (item, expert).
#         name: Type of dispatcher to use (supported values are "einsum").
#         capacity: Maximum number of items processed by each expert.
#         capacity_factor: If given, sets the `capacity` to this factor of S / E.
#         capacity_ceil_or_round: Compute the capacity by either ceiling or rounding.
#         capacity_multiple_of: Ensures that the capacity is a multiple of this number.
#         **dispatcher_kwargs: Additional arguments for the dispatcher object.

#     Returns:
#         A dispatcher and a dictionary of metrics.
#     """
#     if (capacity is None) == (capacity_factor is None):
#         raise ValueError(
#             "You must specify either 'capacity' or 'capacity_factor', and not both."
#             f" Current values are capacity = {capacity!r}, "
#             f"capacity_factor = {capacity_factor!r}")
    
#     if capacity is None:
#         group_size, num_experts = gates.shape
#         capacity = compute_capacity(
#             num_tokens=group_size,
#             num_experts=num_experts,
#             capacity_factor=capacity_factor,
#             ceil_or_round=capacity_ceil_or_round,
#             multiple_of=capacity_multiple_of)
    
#     fn_map = {
#         "einsum": _get_top_items_per_expert_einsum_dispatcher,
#     }
    
#     if name not in fn_map:
#         raise ValueError(f"Unknown dispatcher type: {name!r}")
    
#     return fn_map[name](gates, capacity, **dispatcher_kwargs)



def get_top_items_per_expert_dispatcher(   # 新修改的
    gates: torch.Tensor,
    name: str='einsum',
    capacity: Optional[int] = None,
    capacity_factor: Optional[float] = None,
    capacity_ceil_or_round: str = "ceil",
    capacity_multiple_of: Optional[int] = 4,
    **dispatcher_kwargs) -> Tuple[Any, Dict[str, torch.Tensor]]:
    """PyTorch implementation with identical functionality to JAX version"""
    if (capacity is None) == (capacity_factor is None):
        raise ValueError(
            "You must specify either 'capacity' or 'capacity_factor', and not both."
            f" Current values are capacity = {capacity!r}, "
            f"capacity_factor = {capacity_factor!r}")
    
    if capacity is None:
        group_size, num_experts = gates.shape
        capacity = compute_capacity(
            num_tokens=group_size,
            num_experts=num_experts,
            capacity_factor=capacity_factor,
            ceil_or_round=capacity_ceil_or_round,
            multiple_of=capacity_multiple_of)

    if name != "einsum":
        raise ValueError(f"PyTorch implementation currently only supports 'einsum' dispatcher, got {name!r}")

    return _get_top_items_per_expert_einsum_dispatcher(
        gates, capacity, **dispatcher_kwargs)

def compute_capacity(  # 新修改的
    num_tokens: int,
    num_experts: int,
    capacity_factor: float,
    ceil_or_round: str = "ceil",
    multiple_of: Optional[int] = 4) -> int:
    """Identical capacity computation logic"""
    print(f"[Debug] 输入参数: num_tokens={num_tokens}, num_experts={num_experts}")  # 新增调试
    if ceil_or_round == "ceil":
        capacity = math.ceil(num_tokens * capacity_factor / num_experts)
    elif ceil_or_round == "round":
        capacity = round(num_tokens * capacity_factor / num_experts)
    else:
        raise ValueError(f"Unsupported ceil_or_round={ceil_or_round}")
    
    if capacity < 1:
        raise ValueError(
            f"Invalid capacity={capacity} with num_tokens={num_tokens}, "
            f"num_experts={num_experts}, capacity_factor={capacity_factor}")
    
    if multiple_of and multiple_of > 0:
        remainder = capacity % multiple_of
        if remainder != 0:
            capacity += multiple_of - remainder
    
    actual_capacity_factor = capacity * num_experts / num_tokens
    if abs(actual_capacity_factor - capacity_factor) > 1e-6:
        print(f"Warning: Target capacity_factor {capacity_factor} vs actual {actual_capacity_factor}")
    
    return capacity

def _get_top_items_per_expert_einsum_dispatcher(
    gates: torch.Tensor,
    capacity: int,
    einsum_precision: str = "high",
    **kwargs) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """PyTorch einsum dispatcher with identical logic to JAX version"""
    group_size, num_experts = gates.shape
    
    # Transpose gates to (E, S) for topk selection
    gates_t = gates.t()  # [E, S]
    
    # Select top-k items per expert (different from tokens choosing experts!)
    topk_values, topk_indices = torch.topk(gates_t, k=capacity, dim=1)  # [E, C]
    
    # Create dispatch mask: [E, C, S] -> [S, E, C] after transpose
    dispatch_mask = F.one_hot(topk_indices, num_classes=group_size)  # [E, C, S]
    dispatch_mask = F.one_hot(topk_indices, num_classes=group_size  # 必须等于原始序列长度S=129
    ).permute(2, 0, 1)  # [S, E, C]
    # dispatch_mask = dispatch_mask.permute(2, 0, 1)  # [S, E, C]
    # 验证中间维度
    assert dispatch_mask.shape == (group_size, num_experts, capacity), (
        f"维度错误！应为({group_size}, {num_experts}, {capacity})，"
        f"实际为{dispatch_mask.shape}"
    )
    # print(f"dispatch_mask:{dispatch_mask.shape}")
    dispatch_mask = dispatch_mask.to(torch.bool)
    # Compute combine weights using einsum
    combine_weights = torch.einsum("se,sec->sec", gates, dispatch_mask.float())
    # Compute metrics
    num_experts_per_item = dispatch_mask.sum(dim=(1, 2))  # [S]
    
    metrics = {
        "num_experts_per_item_min": num_experts_per_item.min(),
        "num_experts_per_item_max": num_experts_per_item.max(),
        "min_selected_gate": topk_values.min(),
        "max_selected_gate": topk_values.max(),
    }
    
    # Compute coverage ratios
    log2_num_experts = int(math.log2(num_experts))
    for t in [2**i for i in range(log2_num_experts + 1)] + [num_experts]:
        ratio = (num_experts_per_item >= t).float().mean()
        metrics[f"ratio_processed_items_by_at_least_{t}_experts"] = ratio
    # print(f"应该返回的dispatch_mask:{dispatch_mask.shape}")
    return (dispatch_mask, combine_weights), metrics

# # 参数设置
# num_tokens = 1024
# num_experts = 8
# capacity_factor = 1.5

# # 计算期望容量
# expected_capacity = math.ceil(num_tokens * capacity_factor / num_experts)
# expected_capacity += (-expected_capacity) % 4  # 对齐到4的倍数

# # 生成门控值
# gates = torch.rand(num_tokens, num_experts)

# # 创建dispatcher
# (dispatch_mask, combine_weights), metrics = get_top_items_per_expert_dispatcher(
#     gates=gates,
#     name="einsum",
#     capacity_factor=capacity_factor,
#     capacity_ceil_or_round="ceil"
# )

# # 验证维度
# assert dispatch_mask.shape == (num_tokens, num_experts, expected_capacity)
# assert combine_weights.shape == dispatch_mask.shape

# # 验证指标存在性
# assert "num_experts_per_item_min" in metrics
# assert "ratio_processed_items_by_at_least_1_experts" in metrics

# print("所有测试通过！")

# # 验证维度
# assert dispatch_mask.shape == (num_tokens, num_experts, math.ceil(num_tokens*capacity_factor/num_experts))
# assert combine_weights.shape == dispatch_mask.shape
# def sparse_moe_spmd(
#     target: Callable[..., Any],
#     variable_axes: Dict[str, int],
#     split_rngs: Dict[str, bool],
#     has_aux: bool = False,
#     methods: Optional[str] = None
# ):
#     """Lift transformation that wraps a target with a Sparse MoE using SPMD in PyTorch.

#     Args:
#         target: A function or module representing the expert model.
#         variable_axes: Dict indicating the axis along which each variable collection is "expertified".
#         split_rngs: Dict indicating whether to split each of the PRNGKeys passed to the experts.
#         has_aux: If True, the target returns auxiliary outputs that should not be combined.
#         methods: Not used explicitly in PyTorch, kept for API consistency.

#     Returns:
#         A transformed target function.
#     """
#     def wrapper(expert_fn: Callable[..., Any]):
#         def transformed(dispatcher, *inputs):
#             # Prepare inputs for each expert
#             inputs = tuple(dispatcher.dispatch(inp) for inp in inputs)
            
#             # Apply vmap over the first axis for expert parallelism
#             batched_expert_fn = torch.vmap(expert_fn, in_dims=0, out_dims=0)
#             outputs = batched_expert_fn(*inputs)
            
#             # Combine outputs
#             if has_aux:
#                 outputs, aux = outputs
#             outputs = dispatcher.combine(outputs)
            
#             return (outputs, aux) if has_aux else outputs
        
#         return transformed
    
#     return wrapper(target)

# def sparse_moe_spmd(
#     target: Callable[..., Any],
#     variable_axes: Dict[str, int],
#     split_rngs: Dict[str, bool],
#     has_aux: bool = False,
#     methods: Optional[str] = None
# ):
#     def wrapper(expert_fn: Callable[..., Any]):
#         def transformed(dispatcher, *inputs, **kwargs):
#             # 准备每个 expert 的输入
#             inputs = tuple(dispatcher.dispatch(inp) for inp in inputs)
#             # 使用 vmap 进行批处理
#             batched_expert_fn = torch.vmap(expert_fn, in_dims=0, out_dims=0)
#             outputs = batched_expert_fn(*inputs, **kwargs)
#             if has_aux:
#                 outputs, aux = outputs
#             outputs = dispatcher.combine(outputs)
#             return (outputs, aux) if has_aux else outputs
#         return transformed
#     return wrapper(target)

def sparse_moe_spmd(
    expert_fn: Callable[..., Any],
    variable_axes: Dict[str, int],
    split_rngs: Dict[str, bool],
    has_aux: bool = False
):
    class SparseMoeWrapper(nn.Module):
        def __init__(self, expert_fn, **expert_kwargs):
            # print(f"Expert kwargs: {expert_kwargs}")
            super().__init__()
            self.expert_fn = expert_fn
            self.expert_kwargs = expert_kwargs
            # 初始化多个专家实例（模拟 JAX 的 vmap）
            # self.num_experts = expert_kwargs.get("num_experts", 1)
            self.num_experts = self.expert_kwargs.get("params", 1)
            self.experts = nn.ModuleList([
                expert_fn(**self.expert_kwargs) for _ in range(self.num_experts)
            ])

        def forward(self, dispatcher, inputs: torch.Tensor):
            # 分发输入到不同专家
            expert_inputs = dispatcher.dispatch(inputs)
            # 并行处理（模拟 vmap）
            expert_outputs = [expert(inp) for expert, inp in zip(self.experts, expert_inputs)]
            # expert_outputs = []
            # for expert, inp in zip(self.experts, expert_inputs):
            #     expert_outputs.append(expert(*inp))
            # 合并输出
            outputs = dispatcher.combine(expert_outputs)
            return outputs
        
    def wrapper(**kwargs):
        return SparseMoeWrapper(expert_fn, **kwargs)

    return wrapper

def sparse_moe_spmd_with_axes(
    target: Callable,
    variable_axes: Dict[str, Optional[int]],
    split_rngs: Dict[str, bool],
    partitioning_axis_names: Dict[str, str],
    has_aux: bool = False,
    methods: Optional[List[str]] = None) -> Callable:
    """
    类似于 sparse_moe_spmd，但支持基于命名轴的参数分区。
    
    Args:
        target: 需要转换的函数或模型。
        variable_axes: 指定哪些参数沿哪个维度分区。
        split_rngs: 指定是否拆分随机数生成器（RNG）。
        partitioning_axis_names: 变量集合名称到命名轴的映射。
        has_aux: 是否返回辅助输出。
        methods: 需要转换的方法列表。
    
    Returns:
        转换后的目标函数。
    """
    variable_axes = variable_axes.copy()
    for name in partitioning_axis_names:
        variable_axes[f"{name}_axes"] = None
    
    lifted = sparse_moe_spmd(target, variable_axes, split_rngs, has_aux, methods)
    
    for collection_name, axis in variable_axes.items():
        if collection_name in partitioning_axis_names:
            lifted = _add_axis_to_metadata(
                lifted,
                axis_pos=axis,
                axis_name=partitioning_axis_names[collection_name],
                axis_col=f"{collection_name}_axes"
            )
    
    return lifted


def _cast_to_bfloat16(x: torch.Tensor) -> torch.Tensor:
    """
    如果 x 是浮点类型，则转换为 bfloat16，否则保持不变。
    """
    return x.to(torch.bfloat16) if x.dtype in (torch.float32, torch.float64) else x

def _convert_partition_spec(spec: Optional[Union[str, Tuple[str, ...]]]) -> Optional[Tuple[str, ...]]:
    """
    将分区规范转换为 Tuple[str, ...] 格式，以适应 PyTorch 并行计算的需求。
    """
    if spec is not None and not isinstance(spec, tuple):
        spec = (spec,) if isinstance(spec, str) else tuple(spec)
    return spec

def _dispatch(data: torch.Tensor, partition_spec: Optional[Tuple] = None) -> torch.Tensor:
    """使用 all_to_all 机制分发数据给 experts"""
    partition_spec = _convert_partition_spec(partition_spec)
    num_groups, num_experts, capacity, *item_shape = data.shape
    data = with_sharding_constraint(data, partition_spec)
    
    if num_groups % num_experts == 0:
        data = data.reshape(num_experts, -1, num_experts, capacity, *item_shape)
        data = data.permute(2, 1, 0, 3, *range(4, data.ndim))  # 等价于 JAX 的 swapaxes(0, 2)
    else:
        data = data.permute(1, 0, *range(2, data.ndim))  # 等价于 JAX 的 swapaxes(0, 1)
    
    data = data.reshape(-1, *item_shape)
    data = with_sharding_constraint(data, partition_spec)
    
    return data.reshape(num_experts, num_groups * capacity, *item_shape)

def _receive(data: torch.Tensor, num_groups: int, partition_spec: Optional[Tuple] = None) -> torch.Tensor:
    """使用 all_to_all 机制从 experts 接收数据"""
    partition_spec = _convert_partition_spec(partition_spec)
    num_experts, num_groups_time_capacity, *item_shape = data.shape
    capacity = num_groups_time_capacity // num_groups
    
    data = data.reshape(num_experts * num_groups, capacity, *item_shape)
    data = with_sharding_constraint(data, partition_spec)
    
    if num_groups % num_experts == 0:
        data = data.reshape(num_experts, -1, num_experts, capacity, *item_shape)
        data = data.permute(2, 1, 0, 3, *range(4, data.ndim))  # 等价于 JAX 的 swapaxes(0,2)
        data = data.reshape(num_groups, num_experts, capacity, *item_shape)
    else:
        data = data.reshape(num_experts, num_groups, capacity, *item_shape)
        data = data.permute(1, 0, *range(2, data.ndim))  # 等价于 JAX 的 swapaxes(0,1)
    
    data = with_sharding_constraint(data, partition_spec)
    return data

def _scatter_nd(indices: torch.Tensor, updates: torch.Tensor, shape: Tuple[int, ...]) -> torch.Tensor:
    """
    PyTorch 实现的 scatter_nd，类似于 TensorFlow 的 tf.scatter_nd。

    Args:
        indices: 整数矩阵，形状为 [B, ndim]，表示索引。
        updates: 数据点数组，形状为 [B, ...]，表示需要更新的值。
        shape: 输出张量的形状。

    Returns:
        一个形状为 `shape` 的张量，其中索引指定的位置已更新。
    """
    zeros = torch.zeros(shape, dtype=updates.dtype, device=updates.device)
    indices = tuple(indices.T)  # 转换为索引 tuple
    zeros.index_add_(0, indices[0], updates)  # 使用 index_add_ 累积更新
    return zeros

def _get_top_experts_per_item_common(
    gates: torch.Tensor, num_selected_experts: int, batch_priority: bool
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    计算 Top-Experts-Per-Item 路由的公共张量。

    Args:
        gates: (S, E) 形状的张量，每个 (item, expert) 的 gating 值。
        num_selected_experts: 每个 item 选择的最大专家数。
        batch_priority: 是否使用 batch priority 机制。

    Returns:
        - `combine_weights` 形状为 (S, K)，表示每个 item 选择的 K 个专家的权重。
        - `expert_index` 形状为 (S, K)，表示每个 item 选择的 K 个专家的索引。
        - `buffer_index` 形状为 (S, K, E)，表示 buffer 中 item 和 expert 的索引。
    """
    group_size, num_experts = gates.shape
    combine_weights, expert_index = torch.topk(gates, num_selected_experts, dim=-1)

    if batch_priority:
        # 按照最大 gating 权重进行排序
        perm = torch.argsort(-combine_weights[:, 0])
        expert_index = expert_index[perm]

    # (K, S) -> (K * S,)，确保 top-1 选择的专家比 top-2 选择的专家优先
    expert_index = expert_index.T.reshape(-1)

    # (K * S, E) -> one-hot 编码
    expert_one_hot = F.one_hot(expert_index, num_experts).to(torch.int32)

    # (K * S, E) -> (K, S, E) -> (S, K, E)，使用 cumsum 计算 buffer 索引
    buffer_index = torch.cumsum(expert_one_hot, dim=0) * expert_one_hot - 1
    buffer_index = buffer_index.reshape(num_selected_experts, group_size, num_experts)
    buffer_index = buffer_index.permute(1, 0, 2)  # (S, K, E)

    # (K, S) -> (S, K) 还原 expert_index 形状
    expert_index = expert_index.reshape(num_selected_experts, group_size).T

    if batch_priority:
        # 逆置 perm 以恢复原始顺序
        inv_perm = torch.argsort(perm)
        expert_index = expert_index[inv_perm]
        buffer_index = buffer_index[inv_perm]

    return combine_weights, expert_index, buffer_index

def _get_top_experts_per_item_einsum_dispatcher(
    gates: torch.Tensor, num_selected_experts: int, capacity: int,
    batch_priority: bool, **dispatcher_kwargs) -> EinsumDispatcher:
    """
    使用 EinsumDispatcher 执行 Top-Experts-Per-Item 路由。

    Args:
        gates: (S, E) 张量，每个 (item, expert) 的 gating 值。
        num_selected_experts: 每个 item 选择的最大专家数。
        capacity: 每个专家最多能处理的 item 数。
        batch_priority: 是否使用 batch priority 机制。
        **dispatcher_kwargs: EinsumDispatcher 额外的参数。

    Returns:
        EinsumDispatcher 实例。
    """
    _, _, buffer_idx = _get_top_experts_per_item_common(gates, num_selected_experts, batch_priority)

    # (S, K, E) -> (S, E)，取最大 buffer 索引
    buffer_idx, _ = buffer_idx.max(dim=1)

    # (S, E, C) 将 buffer 索引转换为 one-hot 矩阵
    dispatch_weights = F.one_hot(buffer_idx.clamp(min=0, max=capacity - 1), capacity).to(torch.bool)

    # einsum 计算 "SE,SEC->SEC"
    einsum_precision = dispatcher_kwargs.get("einsum_precision", "default")
    combine_weights = torch.einsum("se,sec->sec", gates, dispatch_weights)

    return EinsumDispatcher(
        combine_weights=combine_weights,
        dispatch_weights=dispatch_weights,
        einsum_precision=einsum_precision,
        **dispatcher_kwargs
    )

def _get_top_experts_per_item_expert_indices_dispatcher(
    gates: torch.Tensor, num_selected_experts: int, capacity: int,
    batch_priority: bool, **dispatcher_kwargs) -> ExpertIndicesDispatcher:
    """
    使用 ExpertIndicesDispatcher 执行 Top-Experts-Per-Item 路由。

    Args:
        gates: (S, E) 张量，每个 (item, expert) 的 gating 值。
        num_selected_experts: 每个 item 选择的最大专家数。
        capacity: 每个专家最多能处理的 item 数。
        batch_priority: 是否使用 batch priority 机制。
        **dispatcher_kwargs: ExpertIndicesDispatcher 额外的参数。

    Returns:
        ExpertIndicesDispatcher 实例。
    """
    _, num_experts = gates.shape

    # 获取 combine_weights, expert_idx, buffer_idx
    combine_weights, expert_idx, buffer_idx = _get_top_experts_per_item_common(
        gates, num_selected_experts, batch_priority
    )

    # (S, K, E) -> (S, K)，取最大 buffer 索引
    buffer_idx, _ = buffer_idx.max(dim=2)

    return ExpertIndicesDispatcher(
        indices=torch.stack([expert_idx, buffer_idx], dim=-1),
        combine_weights=combine_weights,
        num_experts=num_experts,
        capacity=capacity,
        **dispatcher_kwargs
    )


# def _get_top_items_per_expert_einsum_dispatcher(
#     gates: torch.Tensor, capacity: int, **dispatcher_kwargs
# ) -> Tuple[EinsumDispatcher, Dict[str, torch.Tensor]]:
#     """
#     使用 EinsumDispatcher 执行 Top-Items-Per-Expert 路由。

#     Args:
#         gates: (S, E) 张量，每个 (item, expert) 的 gating 值。
#         capacity: 每个专家最多能处理的 item 数。
#         **dispatcher_kwargs: EinsumDispatcher 额外的参数。

#     Returns:
#         EinsumDispatcher 实例和包含训练监控指标的字典。
#     """
#     group_size, num_experts = gates.shape

#     # 获取每个专家的 top-k items (E, C)
#     top_items_gates, top_items_index = torch.topk(gates.T, capacity, dim=1)

#     # 将索引转换为 one-hot 形式 (S, E, C)
#     dispatch_weights = F.one_hot(top_items_index, num_classes=group_size).permute(2, 0, 1).bool()

#     # Einsum 计算 combine_weights
#     einsum_precision = dispatcher_kwargs.get("einsum_precision", "default")
#     combine_weights = torch.einsum("SE,SEC->SEC", gates, dispatch_weights)

#     dispatcher = EinsumDispatcher(
#         dispatch_weights=dispatch_weights,
#         combine_weights=combine_weights,
#         **dispatcher_kwargs
#     )

#     # 计算训练监控指标
#     num_experts_per_item = dispatch_weights.sum(dim=(1, 2)).int()
#     metrics = {
#         "num_experts_per_item_min": num_experts_per_item.min(),
#         "num_experts_per_item_max": num_experts_per_item.max(),
#         "min_selected_gate": top_items_gates.min(),
#         "max_selected_gate": top_items_gates.max(),
#     }

#     log2_num_experts = int(math.log2(num_experts))
#     for t in [2**i for i in range(log2_num_experts + 1)] + [num_experts]:
#         ratio = (num_experts_per_item >= t).sum().float() / group_size
#         metrics[f"ratio_processed_items_by_at_least_{t}_experts"] = ratio

#     return dispatcher, metrics

    