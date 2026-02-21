"""
激活层模块：实现 SiLU + 门控乘法，常用于 LLaMA/SwiGLU 等架构中的 FFN 门控。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import time


class SiluAndMul(nn.Module):
    """
    将输入在最后一维均分为两半，对一半做 SiLU，再与另一半逐元素相乘。
    等价于 SwiGLU 中的 gating 部分：gate(x) * up(x)，这里 gate=SiLU。
    """

    def __init__(self):
        super().__init__()

    # PyTorch 2.0 编译优化：首次调用会编译，后续调用更快，适合在训练/推理循环中重复调用
    @torch.compile
    def forward(self, x):
        # 在最后一维(-1)上切成两段：x 为前半段，y 为后半段（常用于“门控”设计）
        x, y = x.chunk(2, dim=-1)
        # SiLU(x)=x*sigmoid(x)；再与 y 相乘，实现门控输出
        return y * F.silu(x)
    
if __name__ == "__main__":
    # Example usage
    layer = SiluAndMul().cuda()
    input_tensor = torch.randn(8, 4000, 8000).cuda()  # Example input tensor with shape (400, 800)
    
    for _ in range(10):  # Warm-up iterations
        _ = layer(input_tensor)

    times = []
    for _ in range(100):  # Timing iterations
        torch.cuda.synchronize()
        start_time = time.time()
        output_tensor = layer(input_tensor)
        torch.cuda.synchronize()
        end_time = time.time()
        times.append(end_time - start_time)
    avg_time = sum(times) / len(times)
    print(f"Average inference time over 100 runs: {avg_time * 1000:.4f} ms")