import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaptiveLowPassFilter(nn.Module):
   
    
    def __init__(self, in_channels, kernel_size=3):
        super(AdaptiveLowPassFilter, self).__init__()
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        
        # 使用深度可分离卷积减少计算量
        # 输出通道数 = kernel_size * kernel_size，表示每个空间位置的滤波核权重
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
        kernel_weights = F.softmax(kernel_weights, dim=1)  # 在核维度上归一化
        
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


class SimilarityGuidedOffsetGenerator(nn.Module):
    
    
    def __init__(self, in_channels, kernel_size=3, max_offset=0.1):
        super(SimilarityGuidedOffsetGenerator, self).__init__()
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        self.max_offset = max_offset
        
        # 偏移预测网络：输入 = 特征 + 相似度图
        predictor_in_channels = in_channels + kernel_size * kernel_size
        self.offset_predictor = nn.Sequential(
            nn.Conv2d(predictor_in_channels, in_channels // 4, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(in_channels // 4, 2, kernel_size=3, padding=1),
            nn.Tanh()
        )
    
    def compute_local_cosine_similarity(self, x):
       
        B, C, H, W = x.shape
        
        # 使用 unfold 提取所有局部邻域
        # x_unfold: [B, C*K*K, H*W]
        x_unfold = F.unfold(x, kernel_size=self.kernel_size, padding=self.padding)
        x_unfold = x_unfold.view(B, C, self.kernel_size * self.kernel_size, H * W)
        
        # 中心像素（展平）
        center = x.view(B, C, H * W).unsqueeze(2)  # [B, C, 1, H*W]
        
        # 计算余弦相似度
        # 分子：点积
        numerator = (x_unfold * center).sum(dim=1)  # [B, K*K, H*W]
        
        # 分母：L2 范数乘积
        center_norm = torch.norm(center, p=2, dim=1) + 1e-8  # [B, 1, H*W]
        neighbor_norm = torch.norm(x_unfold, p=2, dim=1) + 1e-8  # [B, K*K, H*W]
        denominator = center_norm * neighbor_norm  # [B, K*K, H*W]
        
        # 相似度
        similarity = numerator / denominator  # [B, K*K, H*W]
        similarity = similarity.view(B, self.kernel_size * self.kernel_size, H, W)
        
        return similarity
        
    def forward(self, x):
       
        B, C, H, W = x.shape
        
        # 步骤 1: 计算局部余弦相似度
        similarity = self.compute_local_cosine_similarity(x)  # [B, K*K, H, W]
        
        # 步骤 2: 基于相似度预测偏移量
        features = torch.cat([x, similarity], dim=1)
        offset = self.offset_predictor(features) * self.max_offset  # [B, 2, H, W]
        
        # 步骤 3: 创建采样网格
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1, 1, H, device=x.device),
            torch.linspace(-1, 1, W, device=x.device),
            indexing='ij'
        )
        grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0).expand(B, -1, -1, -1)
        
        # 步骤 4: 应用偏移并重采样
        offset_grid = grid + offset.permute(0, 2, 3, 1)
        x_warped = F.grid_sample(
            x, offset_grid, mode='bilinear', padding_mode='border', align_corners=True
        )
        
        return x_warped


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
        # δ (Dirac delta) 在中心位置为 1，其他位置为 0
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
    
    
    def __init__(self, in_channels, kernel_size=3, use_alpf=True, use_offset=True, use_ahpf=True):
        
        super(FreqFusionBlock, self).__init__()
        
        # 保存配置
        self.use_alpf = use_alpf
        self.use_offset = use_offset
        self.use_ahpf = use_ahpf
        
        # Path 1: ALPF
        if self.use_alpf:
            self.alpf = AdaptiveLowPassFilter(in_channels, kernel_size)
        
        # Path 2: Offset
        if self.use_offset:
            self.offset_generator = SimilarityGuidedOffsetGenerator(in_channels, kernel_size)
        
        # Path 3: AHPF
        if self.use_ahpf:
            self.ahpf = AdaptiveHighPassFilter(in_channels, kernel_size)
        
        # 最终融合卷积
        self.fusion_conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        
    def forward(self, dec_feat, enc_feat, w=1.0):
        
        
        # ============ Path 1: 低频平滑路径 ============
        if self.use_alpf:
            # 对 Decoder 特征进行自适应平滑，去除噪声和混叠
            f_smooth = self.alpf(dec_feat)
        else:
            # 消融实验：跳过 ALPF，直接使用原始 Decoder 特征
            f_smooth = dec_feat
        
        # ============ Path 2: 空间对齐路径 ============
        if self.use_offset:
            # 预测偏移量并对平滑后的特征进行变形，校正空间错位
            f_resamp = self.offset_generator(f_smooth)
        else:
            # 消融实验：跳过 Offset，直接使用 f_smooth
            f_resamp = f_smooth
        
        # ============ Path 3: 高频增强路径 ============
        if self.use_ahpf:
            # 从 Encoder 特征中提取高频细节
            f_high = self.ahpf(enc_feat)
        else:
            # 消融实验：跳过 AHPF，直接使用原始 Encoder 特征
            f_high = enc_feat

        f_out = f_resamp + w * f_high
        
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
