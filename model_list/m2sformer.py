import math
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import kornia.filters.sobel as sobel_filter
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

from model_list import pos_weight_calculator
from model_list.backbone.transformer import load_transformer_backbone_model
from model_list.backbone.transformer.pvt_v2 import Block

def get_freq_indices(method):
    assert method in ['top1','top2','top4','top8','top16','top32',
                      'bot1','bot2','bot4','bot8','bot16','bot32',
                      'low1','low2','low4','low8','low16','low32']
    num_freq = int(method[3:])
    if 'top' in method:
        all_top_indices_x = [0,0,6,0,0,1,1,4,5,1,3,0,0,0,3,2,4,6,3,5,5,2,6,5,5,3,3,4,2,2,6,1]
        all_top_indices_y = [0,1,0,5,2,0,2,0,0,6,0,4,6,3,5,2,6,3,3,3,5,1,1,2,4,2,1,1,3,0,5,3]
        mapper_x = all_top_indices_x[:num_freq]
        mapper_y = all_top_indices_y[:num_freq]
    elif 'low' in method:
        all_low_indices_x = [0,0,1,1,0,2,2,1,2,0,3,4,0,1,3,0,1,2,3,4,5,0,1,2,3,4,5,6,1,2,3,4]
        all_low_indices_y = [0,1,0,1,2,0,1,2,2,3,0,0,4,3,1,5,4,3,2,1,0,6,5,4,3,2,1,0,6,5,4,3]
        mapper_x = all_low_indices_x[:num_freq]
        mapper_y = all_low_indices_y[:num_freq]
    elif 'bot' in method:
        all_bot_indices_x = [6,1,3,3,2,4,1,2,4,4,5,1,4,6,2,5,6,1,6,2,2,4,3,3,5,5,6,2,5,5,3,6]
        all_bot_indices_y = [6,4,4,6,6,3,1,4,4,5,6,5,2,2,5,1,4,3,5,0,3,1,1,2,4,2,1,1,5,3,3,3]
        mapper_x = all_bot_indices_x[:num_freq]
        mapper_y = all_bot_indices_y[:num_freq]
    else:
        raise NotImplementedError
    return mapper_x, mapper_y

class SegmentationHead(nn.Module):
    def __init__(self,
                 in_channels: int,
                 num_classes: int,
                 scale_factor: int) -> None:
        super(SegmentationHead, self).__init__()

        self.region_conv = nn.Sequential(
            nn.Conv2d(in_channels, num_classes, kernel_size=(1, 1), stride=(1, 1)),
            nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=True)
        )
        self.edge_conv = nn.Sequential(
            nn.Conv2d(in_channels, num_classes, kernel_size=(1, 1), stride=(1, 1)),
            nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=True)
        )

    def forward(self, x):
        region_output = self.region_conv(x)
        edge_output = self.edge_conv(x)

        return region_output, edge_output

class UpsampleBlock(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 skip_connection_channels: int) -> None:
        super(UpsampleBlock, self).__init__()

        in_channels = in_channels + skip_connection_channels
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
        )

    def forward(self, x, skip_connection):
        x = F.interpolate(x, size=skip_connection.size()[2:], mode='bilinear', align_corners=True)

        x = torch.cat([x, skip_connection], dim=1)
        x = self.conv1(x)

        return x

class cMFCA(nn.Module):
    def __init__(self,
                 in_channels: int,
                 target_resolution: int,
                 frequency_branches: int=16,
                 frequency_selection: str='top',
                 reduction: int=16) -> None:
        super(cMFCA, self).__init__()

        assert frequency_branches in [1, 2, 4, 8, 16, 32], "Frequency branches must be one of [1, 2, 4, 8, 16, 32]"
        assert frequency_selection in ['top', 'bottom', 'low'], "Frequency selection must be one of ['top', 'bottom', 'low']"

        frequency_selection = frequency_selection + str(frequency_branches)

        self.target_resolution = target_resolution
        self.num_freq = frequency_branches
        self.dct_h, self.dct_w = to_2tuple(target_resolution)

        mapper_x, mapper_y = get_freq_indices(frequency_selection)
        self.num_split = len(mapper_x)
        mapper_x = [temp_x * self.dct_h // 7 for temp_x in mapper_x]
        mapper_y = [temp_y * self.dct_w // 7 for temp_y in mapper_y]

        assert len(mapper_x) == len(mapper_y), "Mapper X length must be equal to the number of splits"

        # fixed DCT init
        for freq_idx in range(frequency_branches):
            self.register_buffer('dct_weight_{}'.format(freq_idx), self.get_dct_filter(self.dct_h, self.dct_w,
                                                                                       mapper_x[freq_idx], mapper_y[freq_idx],
                                                                                       in_channels))

        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1, stride=1, padding=0, bias=False))

        self.avg_channel_pool = nn.AdaptiveAvgPool2d(1)
        self.max_channel_pool = nn.AdaptiveMaxPool2d(1)

    def forward(self, features):
        # STEP0. Save the original resolution for each feature map
        _, _, H1, W1 = features[0].size()
        _, _, H2, W2 = features[1].size()
        _, _, H3, W3 = features[2].size()
        _, _, H4, W4 = features[3].size()

        cross_feature_map_list = []
        # STEP1. Upsample or downsample feature map to match the target resolution
        for feature in features:
            _, _, H, W = feature.size()
            if H != self.dct_h or W != self.dct_w:
                feature = F.interpolate(feature, size=(self.dct_h, self.dct_w), mode='bilinear', align_corners=True)
            cross_feature_map_list.append(feature)

        # STEP2. Collect cross-scale feature map
        cross_feature_map = torch.cat(cross_feature_map_list, dim=1)

        # STEP3. Calculate multi-spectral feature map
        B, C, H, W = cross_feature_map.size()
        x_pooled = cross_feature_map

        if H != self.dct_h or W != self.dct_w:
            x_pooled = torch.nn.functional.adaptive_avg_pool2d(x, (self.dct_h, self.dct_w))

        multi_spectral_feature_avg, multi_spectral_feature_max = 0, 0
        for name, params in self.state_dict().items():
            if 'dct_weight' in name:
                x_pooled_spectral = x_pooled * params
                multi_spectral_feature_avg += self.avg_channel_pool(x_pooled_spectral)
                multi_spectral_feature_max += self.max_channel_pool(x_pooled_spectral)

        multi_spectral_feature_avg = multi_spectral_feature_avg / self.num_freq
        multi_spectral_feature_max = multi_spectral_feature_max / self.num_freq

        multi_spectral_feature_avg = self.fc(multi_spectral_feature_avg).view(B, C, 1, 1)
        multi_spectral_feature_max = self.fc(multi_spectral_feature_max).view(B, C, 1, 1)

        multi_spectral_attention_map = F.sigmoid(multi_spectral_feature_avg + multi_spectral_feature_max)

        cross_feature_map = cross_feature_map * multi_spectral_attention_map.expand_as(cross_feature_map)

        # STEP4. Split the feature map along the channel dimension
        x1, x2, x3, x4 = torch.split(cross_feature_map, C // 4, dim=1)

        # STEP5. Return the original resolution for each feature map
        x1 = F.interpolate(x1, size=(H1, W1), mode='bilinear', align_corners=True)
        x2 = F.interpolate(x2, size=(H2, W2), mode='bilinear', align_corners=True)
        x3 = F.interpolate(x3, size=(H3, W3), mode='bilinear', align_corners=True)
        x4 = F.interpolate(x4, size=(H4, W4), mode='bilinear', align_corners=True)

        return x1, x2, x3, x4

    def get_dct_filter(self, tile_size_x, tile_size_y, mapper_x, mapper_y, in_channels):
        dct_filter = torch.zeros(in_channels, tile_size_x, tile_size_y)

        for t_x in range(tile_size_x):
            for t_y in range(tile_size_y):
                dct_filter[:, t_x, t_y] = self.build_filter(t_x, mapper_x, tile_size_x) * self.build_filter(t_y, mapper_y, tile_size_y)

        return dct_filter

    def build_filter(self, pos, freq, POS):
        result = math.cos(math.pi * freq * (pos + 0.5) / POS) / math.sqrt(POS)
        if freq == 0:
            return result
        else:
            return result * math.sqrt(2)

class MSMSAttentiveBlock(nn.Module):
    def __init__(self,
                 in_channels: int,
                 target_resolution: int,
                 scale_branches: int=1,
                 min_channel: int=64,
                 min_resolution: int=8,
                 frequency_branches: int=16,
                 frequency_selection: str='top',
                 reduction: int=16) -> None:
        super(MSMSAttentiveBlock, self).__init__()

        assert frequency_branches in [1, 2, 4, 8, 16, 32], "Frequency branches must be one of [1, 2, 4, 8, 16, 32]"
        assert frequency_selection in ['top', 'bottom', 'low'], "Frequency selection must be one of ['top', 'bottom', 'low']"

        frequency_selection = frequency_selection + str(frequency_branches)

        self.scale_branches = scale_branches
        self.min_channel = min_channel
        self.min_resolution = min_resolution
        self.target_resolution = target_resolution
        self.num_freq = frequency_branches
        self.dct_h, self.dct_w = to_2tuple(target_resolution)

        # Multi-Spectral Attention Module
        mapper_x, mapper_y = get_freq_indices(frequency_selection)
        self.num_split = len(mapper_x)
        mapper_x = [temp_x * self.dct_h // 7 for temp_x in mapper_x]
        mapper_y = [temp_y * self.dct_w // 7 for temp_y in mapper_y]

        assert len(mapper_x) == len(mapper_y), "Mapper X length must be equal to the number of splits"

        # fixed DCT init
        for freq_idx in range(frequency_branches):
            self.register_buffer('dct_weight_{}'.format(freq_idx), self.get_dct_filter(self.dct_h, self.dct_w,
                                                                                       mapper_x[freq_idx], mapper_y[freq_idx],
                                                                                       in_channels))

        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1, stride=1, padding=0, bias=False))

        self.avg_channel_pool = nn.AdaptiveAvgPool2d(1)
        self.max_channel_pool = nn.AdaptiveMaxPool2d(1)

        # Multi-Scale Attention Module
        for scale_idx in range(scale_branches):
            inter_channels = in_channels // 2 ** scale_idx
            if inter_channels < self.min_channel: inter_channels = self.min_channel

            init_conv = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1 + scale_idx, 1 + scale_idx), dilation=(1 + scale_idx, 1 + scale_idx), bias=False),
                nn.BatchNorm2d(in_channels), nn.ReLU(inplace=True),
                nn.Conv2d(in_channels, inter_channels, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=False),
                nn.BatchNorm2d(inter_channels), nn.ReLU(inplace=True),
            )

            spatial_attention_map = nn.Sequential(
                nn.Conv2d(inter_channels, 1, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=False),
                nn.Sigmoid()
            )

            final_conv = nn.Sequential(
                nn.Conv2d(inter_channels, in_channels, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=False),
                nn.BatchNorm2d(in_channels), nn.ReLU(inplace=True),
            )

            setattr(self, 'init_conv_scale_{}'.format(scale_idx), init_conv)
            setattr(self, 'spatial_attention_map_scale_{}'.format(scale_idx), spatial_attention_map)
            setattr(self, 'final_conv_scale_{}'.format(scale_idx), final_conv)

        self.alpha_list = nn.ParameterList([nn.Parameter(torch.ones(1)) for _ in range(scale_branches)])
        self.beta_list = nn.ParameterList([nn.Parameter(torch.ones(1)) for _ in range(scale_branches)])

    def forward(self, features):
        # STEP0. Save the original resolution for each feature map
        _, _, H1, W1 = features[0].size()
        _, _, H2, W2 = features[1].size()
        _, _, H3, W3 = features[2].size()
        _, _, H4, W4 = features[3].size()

        cross_feature_map_list = []
        # STEP1. Upsample or downsample feature map to match the target resolution
        for feature in features:
            _, _, H, W = feature.size()
            if H != self.dct_h or W != self.dct_w:
                feature = F.interpolate(feature, size=(self.dct_h, self.dct_w), mode='bilinear', align_corners=True)
            cross_feature_map_list.append(feature)

        # STEP2. Collect cross-scale feature map
        cross_feature_map = torch.cat(cross_feature_map_list, dim=1)

        # STEP3. Calculate multi-spectral feature map
        B, C, H, W = cross_feature_map.size()
        x_pooled = cross_feature_map

        if H != self.dct_h or W != self.dct_w:
            x_pooled = torch.nn.functional.adaptive_avg_pool2d(cross_feature_map, (self.dct_h, self.dct_w))

        multi_spectral_feature_avg, multi_spectral_feature_max = 0, 0
        for name, params in self.state_dict().items():
            if 'dct_weight' in name:
                x_pooled_spectral = x_pooled * params
                multi_spectral_feature_avg += self.avg_channel_pool(x_pooled_spectral)
                multi_spectral_feature_max += self.max_channel_pool(x_pooled_spectral)

        multi_spectral_feature_avg = multi_spectral_feature_avg / self.num_freq
        multi_spectral_feature_max = multi_spectral_feature_max / self.num_freq

        multi_spectral_feature_avg = self.fc(multi_spectral_feature_avg).view(B, C, 1, 1)
        multi_spectral_feature_max = self.fc(multi_spectral_feature_max).view(B, C, 1, 1)

        multi_spectral_attention_map = F.sigmoid(multi_spectral_feature_avg + multi_spectral_feature_max)

        cross_feature_map = cross_feature_map * multi_spectral_attention_map.expand_as(cross_feature_map)

        # STEP4. Multi-Scale Attention Module
        refine_feature_map = 0
        for scale_idx in range(self.scale_branches):
            init_conv = getattr(self, 'init_conv_scale_{}'.format(scale_idx))
            spatial_attention_map = getattr(self, 'spatial_attention_map_scale_{}'.format(scale_idx))
            final_conv = getattr(self, 'final_conv_scale_{}'.format(scale_idx))

            if int(cross_feature_map.shape[2] // 2 ** scale_idx) >= self.min_resolution:
                feature = F.avg_pool2d(cross_feature_map, kernel_size=2 ** scale_idx, stride=2 ** scale_idx, padding=0)
            else:
                feature = cross_feature_map

            feature = init_conv(feature)
            spatial_attention_map = spatial_attention_map(feature)
            feature = feature * spatial_attention_map * self.alpha_list[scale_idx] + feature * (1 - spatial_attention_map) * self.beta_list[scale_idx]
            feature = final_conv(feature)

            refine_feature_map += F.interpolate(feature, size=self.target_resolution, mode='bilinear', align_corners=True)
        refine_feature_map = refine_feature_map / self.scale_branches
        refine_feature_map = refine_feature_map + cross_feature_map

        # STEP5. Return the original resolution for each feature map
        x1, x2, x3, x4 = torch.split(refine_feature_map, C // 4, dim=1)
        x1 = F.interpolate(x1, size=(H1, W1), mode='bilinear', align_corners=True)
        x2 = F.interpolate(x2, size=(H2, W2), mode='bilinear', align_corners=True)
        x3 = F.interpolate(x3, size=(H3, W3), mode='bilinear', align_corners=True)
        x4 = F.interpolate(x4, size=(H4, W4), mode='bilinear', align_corners=True)

        return x1, x2, x3, x4

    def get_dct_filter(self, tile_size_x, tile_size_y, mapper_x, mapper_y, in_channels):
        dct_filter = torch.zeros(in_channels, tile_size_x, tile_size_y)

        for t_x in range(tile_size_x):
            for t_y in range(tile_size_y):
                dct_filter[:, t_x, t_y] = self.build_filter(t_x, mapper_x, tile_size_x) * self.build_filter(t_y, mapper_y, tile_size_y)

        return dct_filter

    def build_filter(self, pos, freq, POS):
        result = math.cos(math.pi * freq * (pos + 0.5) / POS) / math.sqrt(POS)
        if freq == 0:
            return result
        else:
            return result * math.sqrt(2)


class SkipConnectionModule(nn.Module):
    def __init__(self,
                 encoder_channel_list: List[int],
                 skip_channel_list: List[int],
                 target_resolution: int=32,
                 scale_branches: int = 1,
                 min_channel: int = 64,
                 min_resolution: int = 8,
                 frequency_branches: int=16,
                 frequency_selection: str='top',
                 reduction: int=16) -> None:
        super(SkipConnectionModule, self).__init__()

        # Skip Connection
        self.skip_connection_reduction1 = nn.Sequential(
            nn.Conv2d(encoder_channel_list[0], skip_channel_list[0], kernel_size=(1, 1), stride=(1, 1)),
            nn.BatchNorm2d(skip_channel_list[0]), nn.ReLU(inplace=True),
        )
        self.skip_connection_reduction2 = nn.Sequential(
            nn.Conv2d(encoder_channel_list[1], skip_channel_list[1], kernel_size=(1, 1), stride=(1, 1)),
            nn.BatchNorm2d(skip_channel_list[1]), nn.ReLU(inplace=True),
        )
        self.skip_connection_reduction3 = nn.Sequential(
            nn.Conv2d(encoder_channel_list[2], skip_channel_list[2], kernel_size=(1, 1), stride=(1, 1)),
            nn.BatchNorm2d(skip_channel_list[2]), nn.ReLU(inplace=True),
        )
        self.skip_connection_reduction4 = nn.Sequential(
            nn.Conv2d(encoder_channel_list[3], skip_channel_list[3], kernel_size=(1, 1), stride=(1, 1)),
            nn.BatchNorm2d(skip_channel_list[3]), nn.ReLU(inplace=True),
        )


        print("Multi-Spectral with Multi-Scale Attention Module")
        print("+++++++++++++++++++++++++++++++++++++++++++++++++")
        print("HYPERPARAMETERS")
        print("target_resolution: ", target_resolution)
        print("scale_branches: ", scale_branches)
        print("min_channel: ", min_channel)
        print("min_resolution: ", min_resolution)
        print("frequency_branches: ", frequency_branches)
        print("frequency_selection: ", frequency_selection)
        print("reduction: ", reduction)
        print("+++++++++++++++++++++++++++++++++++++++++++++++++")

        self.msms_block = MSMSAttentiveBlock(sum(skip_channel_list),
                                             target_resolution=target_resolution,
                                             scale_branches=scale_branches,
                                             min_channel=min_channel,
                                             min_resolution=min_resolution,
                                             frequency_branches=frequency_branches,
                                             frequency_selection=frequency_selection,
                                             reduction=reduction)

    def forward(self, features):
        x1, x2, x3, x4 = features

        x1 = self.skip_connection_reduction1(x1)
        x2 = self.skip_connection_reduction2(x2)
        x3 = self.skip_connection_reduction3(x3)
        x4 = self.skip_connection_reduction4(x4)

        x1, x2, x3, x4 = self.msms_block([x1, x2, x3, x4])

        return [x1, x2, x3, x4]

class OverlapPatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = nn.LayerNorm(embed_dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)

        return x, H, W

class Decoder(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 skip_channels: int,
                 img_size: int=224,
                 patch_size: int=3,
                 stride: int=1,
                 num_heads: int=1,
                 text_embed: int=300,
                 reduction: int=16) -> None:
        super(Decoder, self).__init__()

        self.in_channels = in_channels + skip_channels
        self.patch_embedding = OverlapPatchEmbed(img_size=img_size,
                                                 patch_size=patch_size,
                                                 stride=stride,
                                                 in_chans=self.in_channels,
                                                 embed_dim=out_channels)
        self.block = Block(dim=out_channels, num_heads=num_heads)
        self.norm = nn.LayerNorm(out_channels)
        self.text_guided_attention = TextGuidedAttention(out_channels, text_embed, reduction)

    def forward(self, x, skip, text_feature):
        B, _, _, _ = x.size()
        # x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = F.interpolate(x, size=skip.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat([x, skip], dim=1)

        x, H, W = self.patch_embedding(x)
        x = self.block(x, H, W)
        x = self.norm(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        x = self.text_guided_attention(x, text_feature)

        return x

class TextGuidedAttention(nn.Module):
    def __init__(self,
                 in_channels: int,
                 embedded_feature: int=300,
                 reduction: int=16) -> None:
        super(TextGuidedAttention, self).__init__()

        self.attention_map = nn.Sequential(
            nn.Conv2d(embedded_feature, in_channels//reduction, kernel_size=(1, 1), stride=(1, 1)),
            nn.BatchNorm2d(in_channels//reduction), nn.ReLU(inplace=True),
            nn.Conv2d(in_channels//reduction, in_channels, kernel_size=(1, 1), stride=(1, 1)),
            nn.Sigmoid()
        )

    def forward(self, x, text_feature):
        attention_map = self.attention_map(text_feature.unsqueeze(-1).unsqueeze(-1))
        x = x * attention_map.expand_as(x)

        return x

class M2SFormer(nn.Module):
    def __init__(self, args) -> None:
        super(M2SFormer, self).__init__()

        num_classes = args.num_classes
        skip_channels = args.skip_channels
        transformer_backbone = args.transformer_backbone
        pretrained = args.pretrained

        # MFCA Parameters
        target_resolution = args.target_resolution
        scale_branches = args.scale_branches
        min_channel = args.min_channel
        min_resolution = args.min_resolution
        frequency_branches = args.frequency_branches
        frequency_selection = args.frequency_selection
        reduction = args.reduction
        num_heads = args.num_heads

        text_embedding_length = args.text_embedding_length

        # Load ImageNet Pre-trained Model (Encoder)
        if transformer_backbone in ['pvt_v2_b2',
                                    'p2t_tiny', 'p2t_small', 'p2t_base', 'p2t_large',
                                    'mit_b2']:
            self.feature_encoding = load_transformer_backbone_model(backbone_name=transformer_backbone, pretrained=pretrained)
            self.feature_encoding.head = nn.Identity()

            if transformer_backbone in ['p2t_tiny']:
                self.in_channels = 384
                self.encoder_channel_list = [48, 96, 240]
            elif transformer_backbone in ['pvt_v2_b2',
                                          'p2t_small', 'p2t_base',
                                          'mit_b2']:
                self.in_channels = 512
                self.encoder_channel_list = [64, 128, 320] + [self.in_channels]

            self.skip_channel_list = [skip_channels for _ in range(len(self.encoder_channel_list))]
            self.decoder_filters = [256, 128, 64]
            self.scale_factor_list = [32, 16, 8, 4]
        else:
            print("{} does not support".format(transformer_backbone))
            raise NotImplementedError


        print("Encoder Channel List: ", self.encoder_channel_list)
        print("Skip Channel List: ", self.skip_channel_list)

        self.skip_connection_module = SkipConnectionModule(self.encoder_channel_list, self.skip_channel_list,
                                                           target_resolution=target_resolution,
                                                           scale_branches=scale_branches,
                                                           min_channel=min_channel,
                                                           min_resolution=min_resolution,
                                                           frequency_branches=frequency_branches,
                                                           frequency_selection=frequency_selection,
                                                           reduction=reduction)

        self.decoder_stage1 = Decoder(self.skip_channel_list[0], self.decoder_filters[0], self.skip_channel_list[0], num_heads=num_heads, text_embed=text_embedding_length)
        self.decoder_stage2 = Decoder(self.decoder_filters[0], self.decoder_filters[1], self.skip_channel_list[1], num_heads=num_heads, text_embed=text_embedding_length)
        self.decoder_stage3 = Decoder(self.decoder_filters[1], self.decoder_filters[2], self.skip_channel_list[2], num_heads=num_heads, text_embed=text_embedding_length)

        # Segmentation Head
        self.global_prior = nn.Sequential(
            nn.Conv2d(self.decoder_filters[2], num_classes, kernel_size=(1, 1), stride=(1, 1)),
            nn.Upsample(scale_factor=32, mode='bilinear', align_corners=True)
        )
        self.seg_head = SegmentationHead(self.decoder_filters[2], num_classes, self.scale_factor_list[3])

        # self.vocab_embedding = nn.Embedding(2, 768)
        # vocab_embedding = self.vocab_embedding()
        # self.word_to_index = {word: idx for idx, word in enumerate(["easy", "hard"])}
        # print(self.word_to_index)
        self.dictionary = ["easy", "hard"]
        text_embed_list = []
        self.embed = Text2Embed(text_embedding_length=text_embedding_length)
        for text_information_ in self.dictionary:
            text_embed = self.embed.to_embed(text_information_)[0]
            text_embed_list.append(text_embed)
        self.text_embed = torch.tensor(text_embed_list).cuda()
        print(self.text_embed.shape) # self.text_embed[0] => "easy" & self.text_embed[1] => "hard"

    def forward(self, data):
        x = data['image']
        if x.size()[1] == 1: x = x.repeat(1, 3, 1, 1)

        x1, x2, x3, x4 = self.feature_encoding.forward_feature(x)
        features = [x1, x2, x3, x4]

        skip_features = self.skip_connection_module(features)

        global_prior = self.global_prior(skip_features[3])
        text_feature = self.calculate_curvature(global_prior)

        x1 = self.decoder_stage1(skip_features[3], skip_features[2], text_feature)
        x2 = self.decoder_stage2(x1, skip_features[1], text_feature)
        x3 = self.decoder_stage3(x2, skip_features[0], text_feature)

        region_output, edge_output = self.seg_head(x3)

        output_dict = {'prediction': region_output, 'edge': edge_output,
                       'global_prior': global_prior}
        output_dict = self._calculate_criterion(output_dict, data)

        return output_dict

    def _calculate_criterion(self, output_dict, data):
        pos_weight = pos_weight_calculator(data['target'])
        # Calculate Region Loss
        region_loss = F.binary_cross_entropy_with_logits(output_dict['prediction'], data['target'], pos_weight=pos_weight)
        global_prior_loss = F.binary_cross_entropy_with_logits(output_dict['global_prior'], data['target'], pos_weight=pos_weight)

        # Calculate Edge Loss
        edge_true = sobel_filter(data['target'])
        edge_true[edge_true >= 0.5] = 1; edge_true[edge_true < 0.5] = 0
        edge_loss = F.binary_cross_entropy_with_logits(output_dict['edge'], edge_true)

        loss = region_loss + edge_loss + global_prior_loss
        # print("Total Loss: {:.4f} | Region Loss: {:.4f} | Edge Loss: {:.4f}".format(loss.item(), region_loss.item(), edge_loss.item()))

        output_dict['loss'] = loss

        return output_dict

    def calculate_curvature(self, label_mask):
        # Sobel Edge Detection
        fx, fy = self.sobel_filter(label_mask)  # (B, 1, H, W)

        fxx, fxy = self.sobel_filter(fx)
        fyx, fyy = self.sobel_filter(fy)

        eps = 1e-8
        fx2 = fx * fx
        fy2 = fy * fy
        denom = (fx2 + fy2).pow(1.5) + eps

        numerator = fx2 * fyy - 2.0 * fx * fy * fxy + fy2 * fxx
        kappa = numerator / denom
        score = kappa.sum(dim=[2, 3])
        scores = torch.sigmoid(score).squeeze(dim=1)  # shape (B,)

        hard_mask = (scores > 0.5).long()  # shape (B,)
        text_feature = self.text_embed[hard_mask]  # shape (B, D)

        return text_feature

    def sobel_filter(self, x: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        """
        x: shape (B, 1, H, W)
        return: (Gx, Gy) each shape (B, 1, H, W)
        """
        # Sobel 필터 커널 정의
        sobel_kernel_x = torch.tensor([[1., 0., -1.],
                                       [2., 0., -2.],
                                       [1., 0., -1.]]).view(1, 1, 3, 3)
        sobel_kernel_y = torch.tensor([[1., 2., 1.],
                                       [0., 0., 0.],
                                       [-1., -2., -1.]]).view(1, 1, 3, 3)

        # 커널을 x와 같은 device로 이동
        sobel_kernel_x = sobel_kernel_x.to(x.device)
        sobel_kernel_y = sobel_kernel_y.to(x.device)

        # Convolution을 이용해 Gx, Gy 계산
        Gx = F.conv2d(x, sobel_kernel_x, padding=1)
        Gy = F.conv2d(x, sobel_kernel_y, padding=1)

        return Gx, Gy


def _training_config(args):
    # Model Argument
    args.region_loss = 'DICE_FOCAL'
    args.skip_channels = 64
    args.cnn_backbone = None
    args.transformer_backbone = 'pvt_v2_b2'
    args.pretrained = True

    args.target_resolution = 32
    args.scale_branches = 2
    args.min_channel = 64
    args.min_resolution = 8
    args.frequency_branches = 16
    args.frequency_selection = 'top'
    args.reduction = 16
    args.num_heads = 1

    args.text_embedding_length = 300

    # Dataset Argument
    args.num_channels = 3
    args.num_classes = 1
    args.image_size = 256
    args.metric_list =  ['DSC', 'IoU', 'E-Measure','AUC']
    args.mean = [0.485, 0.456, 0.406]
    args.std = [0.229, 0.224, 0.225]

    # Training Argument
    args.multi_scale_train = False
    args.train_batch_size = 32
    args.test_batch_size = 50
    args.final_epoch = 100

    # Optimizer Argument
    args.optimizer_name = 'AdamW'
    args.lr = 1e-4
    args.weight_decay = 1e-4
    args.adjust_learning_rate = adjust_learning_rate

    return args

def get_lr(step, total_steps, lr_max, lr_min):
  """Compute learning rate according to cosine annealing schedule."""
  return lr_min + (lr_max - lr_min) * 0.5 * (1 + np.cos(step / total_steps * np.pi))

def adjust_learning_rate(optimizer, epochs, train_loader_len, learning_rate):
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: get_lr(  # pylint: disable=g-long-lambda
            step,
            epochs * train_loader_len,
            1,  # lr_lambda computes multiplicative factor
            1e-6 / learning_rate))

    return scheduler

import numpy as np
from bpemb import BPEmb

""" Subword Embeddings: https://nlp.h-its.org/bpemb """

class Text2Embed:
    def __init__(self, text_embedding_length):
        self.bpemb_en = BPEmb(lang="en", vs=100000, dim=text_embedding_length)

    def to_tokens(self, word):
        tokens = self.bpemb_en.encode(word)
        return tokens

    def to_embed(self, word, mean=True):
        embed = self.bpemb_en.embed(word)
        if mean == True and len(embed) > 1:
            embed = np.mean(embed, axis=0)
            embed = np.expand_dims(embed, axis=0)
        return embed