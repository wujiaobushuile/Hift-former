import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaptiveLowPassFilter(nn.Module):
   
    
    def __init__(self, in_channels, kernel_size=3):
        super(AdaptiveLowPassFilter, self).__init__()
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2

        self.kernel_predictor = nn.Sequential(
            # 深度卷积：每个通道独立处理
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels),
            # 点卷积：跨通道信息交互
            nn.Conv2d(in_channels, in_channels // 2, kernel_size=1),
            nn.LeakyReLU(0.2, True),
            # 预测滤波核权重
            nn.Conv2d(in_channels // 2, kernel_size * kernel_size, kernel_size=1)
        )
        
    def forward(self, x):
      
        B, C, H, W = x.shape
        
        # 步骤 1: 预测滤波核权重 [B, K*K, H, W]
        # 每个空间位置 (h, w) 都有一组 K*K 个权重
        kernel_weights = self.kernel_predictor(x)
        kernel_weights = F.softmax(kernel_weights, dim=1) 
        
        # 步骤 2: 使用 unfold 提取每个位置的邻域
        # unfold 相当于滑动窗口，提取所有局部邻域
        x_unfold = F.unfold(x, kernel_size=self.kernel_size, padding=self.padding)  # [B, C*K*K, H*W]
        x_unfold = x_unfold.view(B, C, self.kernel_size * self.kernel_size, H * W)  # [B, C, K*K, H*W]
        
        # 步骤 3: 应用空间自适应滤波
        # 对每个位置，用预测的权重对邻域进行加权求和
        kernel_weights = kernel_weights.view(B, 1, self.kernel_size * self.kernel_size, H * W)
        x_smooth = (x_unfold * kernel_weights).sum(dim=2)  # [B, C, H*W]
        x_smooth = x_smooth.view(B, C, H, W)
        
        return x_smooth


class CrossAttentionAlignment(nn.Module):
    
    def __init__(self, dim, num_heads=8):
        super(CrossAttentionAlignment, self).__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        assert dim % num_heads == 0, f"dim ({dim}) 必须能被 num_heads ({num_heads}) 整除"
        
        # Query 来自 Decoder 特征
        self.q_proj = nn.Conv2d(dim, dim, kernel_size=1)
        # Key 和 Value 来自 Encoder 特征
        self.kv_proj = nn.Conv2d(dim, dim * 2, kernel_size=1)
        # 输出投影
        self.out_proj = nn.Conv2d(dim, dim, kernel_size=1)
        
    def forward(self, dec_feat, enc_feat):
        
        B, C, H, W = dec_feat.shape
        
        # 生成 Query, Key, Value
        q = self.q_proj(dec_feat)  # [B, C, H, W]
        kv = self.kv_proj(enc_feat)  # [B, 2*C, H, W]
        k, v = torch.chunk(kv, 2, dim=1)  # 各自 [B, C, H, W]

        # [B, C, H, W] -> [B, num_heads, H*W, head_dim]
        q = q.reshape(B, self.num_heads, self.head_dim, H * W).permute(0, 1, 3, 2)
        k = k.reshape(B, self.num_heads, self.head_dim, H * W).permute(0, 1, 3, 2)
        v = v.reshape(B, self.num_heads, self.head_dim, H * W).permute(0, 1, 3, 2)
        
        # 计算注意力
        attn = (q @ k.transpose(-2, -1)) * self.scale  # [B, num_heads, H*W, H*W]
        attn = attn.softmax(dim=-1)
        
        # 加权求和
        out = attn @ v  # [B, num_heads, H*W, head_dim]
        
        # Reshape
        out = out.permute(0, 1, 3, 2).reshape(B, C, H, W)
        
        # 输出投影
        out = self.out_proj(out)
        
        return out


class AdaptiveHighPassFilter(nn.Module):
   
    
    def __init__(self, in_channels, kernel_size=3):
        super(AdaptiveHighPassFilter, self).__init__()
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        
        # 权重预测网络：预测低通权重，用于反转为高通
        self.weight_predictor = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(in_channels // 2, kernel_size * kernel_size, kernel_size=1)
        )
        
    def forward(self, x):
        
        B, C, H, W = x.shape
        
        # 步骤 1: 预测低通权重并 Softmax 归一化
        W_LP = self.weight_predictor(x)  # [B, K*K, H, W]
        W_LP = F.softmax(W_LP, dim=1)
        
        # 步骤 2: 滤波器反转
        delta = torch.zeros_like(W_LP)
        center_idx = self.kernel_size * self.kernel_size // 2
        delta[:, center_idx, :, :] = 1.0
        W_HP = delta - W_LP  # 高通权重
        
        # 步骤 3: 使用高通权重提取高频
        x_unfold = F.unfold(x, kernel_size=self.kernel_size, padding=self.padding)
        x_unfold = x_unfold.view(B, C, self.kernel_size * self.kernel_size, H * W)
        
        W_HP = W_HP.view(B, 1, self.kernel_size * self.kernel_size, H * W)
        high_freq = (x_unfold * W_HP).sum(dim=2).view(B, C, H, W)
        
        # 步骤 4: 残差连接
        return x + high_freq


class FreqFusionBlock(nn.Module):
   
    def __init__(self, in_channels, kernel_size=3, num_heads=8, 
                 use_alpf=True, use_attention=True, use_ahpf=True):
        super(FreqFusionBlock, self).__init__()
        
        self.use_alpf = use_alpf
        self.use_attention = use_attention
        self.use_ahpf = use_ahpf
        
        # Path 1: 自适应低通滤波（去噪、抗混叠）
        if self.use_alpf:
            self.alpf = AdaptiveLowPassFilter(in_channels, kernel_size)
        
        # Path 2: 跨尺度注意力对齐（隐式特征对齐）
        if self.use_attention:
            self.cross_attn = CrossAttentionAlignment(in_channels, num_heads)
        
        # Path 3: 自适应高通滤波（细节增强）
        if self.use_ahpf:
            self.ahpf = AdaptiveHighPassFilter(in_channels, kernel_size)
        
        # 最终融合卷积
        self.fusion_conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        
    def forward(self, dec_feat, enc_feat, w=1.0):

        
        # ============ Path 1: 低通平滑 ============
        if self.use_alpf:
            f_smooth = self.alpf(dec_feat)
        else:
            f_smooth = dec_feat
        
        # ============ Path 2: 跨尺度对齐 ============
        if self.use_attention:
            # 通过交叉注意力隐式对齐 Decoder 和 Encoder 特征
            f_aligned = self.cross_attn(f_smooth, enc_feat)
        else:
            f_aligned = f_smooth
        
        # ============ Path 3: 高频增强 ============
        if self.use_ahpf:
            f_high = self.ahpf(enc_feat)
        else:
            f_high = enc_feat
        
        # ============ 特征融合 ============
        f_out = f_aligned + w * f_high
        f_out = self.fusion_conv(f_out)
        
        return f_out


class FreqFusionDecoder(nn.Module):
    
    
    def __init__(self, channels_dict, connect_list, kernel_size=3):
        
        super(FreqFusionDecoder, self).__init__()
        
        self.connect_list = connect_list
        
        # 为每个需要融合的分辨率创建一个 FreqFusion 块
        self.freqfusion_blocks = nn.ModuleDict()
        for res in connect_list:
            if res in channels_dict:
                channels = channels_dict[res]
                self.freqfusion_blocks[res] = FreqFusionBlock(channels, kernel_size)
            else:
                raise ValueError(f"分辨率 {res} 不在 channels_dict 中")
    
    def forward(self, enc_feat_dict, dec_feat, resolution, w=1.0):
        
        # 如果当前分辨率在融合列表中，应用 FreqFusion
        if resolution in self.connect_list and resolution in self.freqfusion_blocks:
            enc_feat = enc_feat_dict[resolution]
            dec_feat = self.freqfusion_blocks[resolution](dec_feat, enc_feat, w)
        
        return dec_feat
