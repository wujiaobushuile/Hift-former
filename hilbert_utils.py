import torch
import torch.nn as nn
import numpy as np
from hilbertcurve.hilbertcurve import HilbertCurve


class HilbertPermutation(nn.Module):
    
    
    def __init__(self, h, w):
        super(HilbertPermutation, self).__init__()
        self.h = h
        self.w = w
        self.length = h * w
        
        # 预计算希尔伯特曲线遍历的索引
        # 使用 register_buffer 注册为 buffer，这样：
        # 1. 索引会自动跟随模型移动到正确的设备（CPU/GPU）
        # 2. 在保存/加载模型时会被正确处理
        # 3. 不会被当作模型参数，不参与梯度计算
        # 4. 避免每次 forward 时重复移动到设备，提高效率
        hilbert_indices = self._compute_hilbert_indices()
        inverse_indices = self._compute_inverse_indices(hilbert_indices)
        self.register_buffer('hilbert_indices', hilbert_indices)
        self.register_buffer('inverse_indices', inverse_indices)
        
    def _compute_hilbert_indices(self):
        
        # 确定希尔伯特曲线的阶数（必须是2的幂）
        # 使用最大维度来确定阶数
        max_dim = max(self.h, self.w)
        p = int(np.ceil(np.log2(max_dim)))  # 希尔伯特曲线的阶数
        n = 2  # 维度 (2D)
        
        # 初始化希尔伯特曲线
        hilbert_curve = HilbertCurve(p, n)
        
        # 创建从希尔伯特位置到 (y, x) 坐标的映射
        hilbert_order = []
        for i in range(2 ** (p * n)):
            coords = hilbert_curve.point_from_distance(i)
            y, x = coords[0], coords[1]
            
            # 检查坐标是否在我们的网格范围内
            if y < self.h and x < self.w:
                linear_idx = y * self.w + x  # row-major 索引
                hilbert_order.append(linear_idx)
                
            # 如果已经遍历了所有元素，则停止
            if len(hilbert_order) == self.length:
                break
        
        # 转换为 tensor
        hilbert_indices = torch.tensor(hilbert_order, dtype=torch.long)
        
        return hilbert_indices
    
    def _compute_inverse_indices(self, hilbert_indices):
        
        inverse_indices = torch.zeros_like(hilbert_indices)
        for i, idx in enumerate(hilbert_indices):
            inverse_indices[idx] = i
        
        return inverse_indices
    
    def to_sequence(self, x):
        
        B, C, H, W = x.shape
        assert H == self.h and W == self.w, \
            f"输入大小 ({H}, {W}) 与初始化大小 ({self.h}, {self.w}) 不匹配"
        
        # 转换 [B, C, H, W] -> [B, C, H*W]
        x_flat = x.flatten(2)  # [B, C, L]
        
        # 使用 gather 应用希尔伯特排列
        # 注意：self.hilbert_indices 已经通过 register_buffer 注册，
        # 会自动与模型一起移动到正确的设备，无需手动 .to(device)
        # 扩展索引以适应 batch 和 channel 维度
        indices = self.hilbert_indices.unsqueeze(0).unsqueeze(0).expand(B, C, -1)  # [B, C, L]
        x_hilbert = torch.gather(x_flat, 2, indices)  # [B, C, L]
        
        # 转置为 [B, L, C] 以便输入 Transformer
        x_seq = x_hilbert.permute(0, 2, 1)  # [B, L, C]
        
        return x_seq
    
    def to_image(self, x):
        
        B, L, C = x.shape
        assert L == self.length, \
            f"序列长度 ({L}) 与期望长度 ({self.length}) 不匹配"
        
        # 转置为 [B, C, L]
        x_hilbert = x.permute(0, 2, 1)  # [B, C, L]
        
        # 应用逆排列
        # 注意：self.inverse_indices 已经通过 register_buffer 注册，
        # 会自动与模型一起移动到正确的设备，无需手动 .to(device)
        indices = self.inverse_indices.unsqueeze(0).unsqueeze(0).expand(B, C, -1)  # [B, C, L]
        x_flat = torch.gather(x_hilbert, 2, indices)  # [B, C, L]
        
        # 恢复 2D 形状
        x_2d = x_flat.view(B, C, self.h, self.w)  # [B, C, H, W]
        
        return x_2d


def get_hilbert_permutation(h, w):
    
    return HilbertPermutation(h, w)