import torch.nn as nn 
import torch
# import torch.distributed as dist

# 模拟分布式环境
class Dist:
    def __init__(self, world_size: int):
        self.world_size = world_size
        self._rank_pool = set(range(world_size))

    def get_rank(self):
        if not self._rank_pool:
            raise RuntimeError("No available rank left")
        r = self._rank_pool.pop()
        print(rf"rank:{r}")
        return r

    def get_world_size(self):
        return self.world_size

# 初始化设备数量
dist = Dist(world_size=4)


class LinearBase(nn.Module):
    """
    A base class for linear layers.
    """

    def __init__(
        self, 
        input_size: int, 
        output_size: int,
        bias: bool = True,
        tp_dim: int | None = None,
    ):
        super().__init__()
        # set tp_dim, tp_rank, tp_world_size for tensor parallelism
        self.tp_dim = tp_dim 
        self.tp_rank = dist.get_rank()
        self.tp_size = dist.get_world_size()
        
        # create weight parameter with custom weight loader
        self.weight = nn.Parameter(torch.zeros(output_size, input_size))
        self.weight.weight_loader = self.weight_loader

        # create bias parameter
        if bias:
            self.bias = nn.Parameter(torch.zeros(output_size))
            self.bias.weight_loader = self.weight_loader 
        else:
            self.register_parameter('bias', None)

    def weight_loader(self, param: nn.Parameter, loaded_weights: torch.Tensor):
        raise NotImplementedError("Subclasses should implement this method.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Subclasses should implement this method.")



class ColumnParallelLinear(LinearBase):
    def __init__(
        self, 
        input_size: int, 
        output_size: int,
        bias: bool = True,
    ):
        tp_size = dist.get_world_size()
        assert output_size % tp_size == 0, "Output size must be divisible by tensor parallel size."
        super().__init__(input_size, output_size//tp_size, bias, tp_dim=0)

    # param: parameter after tensor parallelism
    # loaded_weights: the original full parameter to be loaded into param
    def weight_loader(self, param: nn.Parameter, loaded_weights: torch.Tensor):
        param_data = param.data 
        # full_dim on the output column
        full_data_output_size = loaded_weights.size(0)
        # dim size after sharding
        shard_size = full_data_output_size // self.tp_size
        assert shard_size == param_data.size(0), "Shard size does not match parameter size."
        # starting index
        start_index = self.tp_rank * shard_size
        slided_weight = loaded_weights.narrow(0, start_index, shard_size)
        param_data.copy_(slided_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return nn.functional.linear(x, self.weight, self.bias)



if __name__ == "__main__":
    # 定义初始化参数
    input_size = 4
    output_size = 12
    world_size = dist.get_world_size()

    # 定义权重
    shard_out = output_size // world_size
    shards = []
    for rank in range(world_size):
        shard = torch.full(
            (shard_out, input_size),
            fill_value=rank + 1,   # shard 0 -> 1, shard 1 -> 2, ...
            dtype=torch.float32
        )
        shards.append(shard)
    full_weight = torch.cat(shards, dim=0).cuda()
    print(f"full_weight.shape = {full_weight.shape}")

    # print weight
    def print_weight(layer):
        print(f"Before loading weight:")
        print(layer.weight)
        layer.weight_loader(layer.weight,full_weight)
        print("After loading weight:")
        print(layer.weight)
        print()

    for _ in range(world_size):
        layer = ColumnParallelLinear(input_size=input_size, output_size=output_size)
        print_weight(layer)
