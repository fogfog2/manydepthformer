"""
This file contains all operations about building CMT model
paper: <CMT: Convolutional Neural Networks Meet Vision Transformers>
addr: https://arxiv.org/abs/2107.06263

NOTE: In the paper, in Table 1, authors denote Patch Aggregation and LPU may have different channels,
but in practice, in LPU, residual connection is need, so difference in channels may make summation
impossible, therefore, we use the same channels between Patch Aggregation and LPU. So calculated number of
params differ a little from the paper.

Update: The new version of architecture is released to fix the above bugs.

Created by Kunhong Yu
Date: 2021/07/14
"""

import torch as t
from torch.nn import functional as F
from manydepth.networks.cmtmodules import Stem, PatchAggregation, CMTBlock


#########################
#   CMT Configuration   #
#########################
class CMT(t.nn.Module):
    """Define CMT model"""

    def __init__(self,
                 in_channels = 3,
                 stem_channels = 16,
                 cmt_channelses = [46, 92, 184, 368],
                 pa_channelses = [46, 92, 184, 368],
                 R = 3.6,
                 repeats = [2, 2, 10, 2],
                 input_width = 640,
                 input_height = 192):
        """
        Args :
            --in_channels: default is 3
            --stem_channels: stem channels, default is 16
            --cmt_channelses: list, default is [46, 92, 184, 368]
            --pa_channels: patch aggregation channels, list, default is [46, 92, 184, 368]
            --R: expand ratio, default is 3.6
            --repeats: list, to specify how many CMT blocks stacked together, default is [2, 2, 10, 2]
            --input_size: default is 224
            --num_classes: default is 1000 for ImageNet
        """
        super(CMT, self).__init__()

        # if input_size == 224:
        #     sizes = [56, 28, 14, 7]
        # elif input_size == 160:
        #     sizes = [40, 20, 10, 5]
        # elif input_size == 192:
        #     sizes = [48, 24, 12, 6]
        # elif input_size == 256:
        #     sizes = [64, 32, 16, 8]
        # elif input_size == 288:
        #     sizes = [72, 36, 18, 9]
        # else:
        #     raise Exception('No other input sizes!')
        
        widths = [int(input_width/4), int(input_width/8), int(input_width/16), int(input_width/32)]
        heights = [int(input_height/4), int(input_height/8), int(input_height/16), int(input_height/32)]

        # 1. Stem
        #self.stem = Stem(in_channels = in_channels, out_channels = stem_channels, stride = 2)

        # 2. Patch Aggregation 1
        #self.pa1 = PatchAggregation(in_channels = stem_channels, out_channels = pa_channelses[0], kernel_size=3, stride = 1, padding = 1)
        #self.pa1 = PatchAggregation(in_channels = stem_channels, out_channels = pa_channelses[0], kernel_size=2, stride = 2, padding = 0)
        #self.pa2 = PatchAggregation(in_channels = cmt_channelses[0], out_channels = pa_channelses[1])
        self.pa2 = PatchAggregation(in_channels = 46, out_channels = pa_channelses[1])
        self.pa3 = PatchAggregation(in_channels = cmt_channelses[1], out_channels = pa_channelses[2])
        self.pa4 = PatchAggregation(in_channels = cmt_channelses[2], out_channels = pa_channelses[3])

        # 3. CMT block
        # cmt1 = []
        # for _ in range(repeats[0]):
        #     cmt_layer = CMTBlock(input_width = widths[0],
        #                          input_height = heights[0],
        #                          kernel_size = 8,
        #                          d_k = cmt_channelses[0],
        #                          d_v = cmt_channelses[0],
        #                          num_heads = 1,
        #                          R = R, in_channels = pa_channelses[0])
        #     cmt1.append(cmt_layer)
        # self.cmt1 = t.nn.Sequential(*cmt1)

        cmt2 = []
        for _ in range(repeats[1]):
            cmt_layer = CMTBlock(input_width = widths[1],
                                 input_height = heights[1],
                                 kernel_size = 4,
                                 d_k = cmt_channelses[1] // 2,
                                 d_v = cmt_channelses[1] // 2,
                                 num_heads = 2,
                                 R = R, in_channels = pa_channelses[1])
            cmt2.append(cmt_layer)
        self.cmt2 = t.nn.Sequential(*cmt2)

        cmt3 = []
        for _ in range(repeats[2]):
            cmt_layer = CMTBlock(input_width = widths[2],
                                 input_height = heights[2],
                                 kernel_size = 2,
                                 d_k = cmt_channelses[2] // 4,
                                 d_v = cmt_channelses[2] // 4,
                                 num_heads = 4,
                                 R = R, in_channels = pa_channelses[2])
            cmt3.append(cmt_layer)
        self.cmt3 = t.nn.Sequential(*cmt3)

        cmt4 = []
        for _ in range(repeats[3]):
            cmt_layer = CMTBlock(input_width = widths[3],
                                 input_height = heights[3],
                                 kernel_size = 1,
                                 d_k = cmt_channelses[3] // 8,
                                 d_v = cmt_channelses[3] // 8,
                                 num_heads = 8,
                                 R = R, in_channels = pa_channelses[3])
            cmt4.append(cmt_layer)
        self.cmt4 = t.nn.Sequential(*cmt4)

        # 4. Global Avg Pool
        #self.avg = t.nn.AdaptiveAvgPool2d(1)
        
        numl_layer = len(cmt_channelses)
        num_features = [int(cmt_channelses[0] * 2 ** i) for i in range(numl_layer)]
        self.num_features = num_features

        # add a norm layer for each output
        norm_layer = t.nn.LayerNorm
        out_indices=(0, 1, 2, 3)
        self.out_indices = out_indices
        for i_layer in out_indices:
            layer = norm_layer(num_features[i_layer])
            layer_name = f'norm{i_layer}'
            self.add_module(layer_name, layer)

        # 5. FC
        # self.fc = t.nn.Sequential(
        #     t.nn.Linear(cmt_channelses[-1], 1280),
        #     t.nn.ReLU(inplace = True) # we use ReLU here as default
        # )

        # # 6. Classifier
        # self.classifier = t.nn.Sequential(
        #     t.nn.Linear(1280, num_classes)
        # )

    def forward(self, x):

        
        self.features = []
        
        # 1. Stem
        # x = self.stem(x)

        
        # # 2. PA1 + CMTb1
        # x_pa1 = self.pa1(x)
        # x_cmtb1 = self.cmt1(x_pa1)

        # norm_layer = getattr(self, f'norm0')
        # x_out = norm_layer(x_cmtb1)
        # self.features.append(x_out)

        #3. PA2 + CMTb2
        x_pa2 = self.pa2(x)
        x_cmtb2 = self.cmt2(x_pa2)
        
        norm_layer = getattr(self, f'norm1')
        x_out = x_cmtb2.permute(0, 3, 2, 1).contiguous()
        x_out = norm_layer(x_out)
        x_out = x_out.permute(0, 3, 2, 1).contiguous()
        self.features.append(x_out)
        
        # 4. PA3 + CMTb3
        x_pa3 = self.pa3(x_cmtb2)
        x_cmtb3 = self.cmt3(x_pa3)
        
        norm_layer = getattr(self, f'norm2')
        x_out = x_cmtb3.permute(0, 3, 2, 1).contiguous()
        x_out = norm_layer(x_out)
        x_out = x_out.permute(0, 3, 2, 1).contiguous()
        self.features.append(x_out)

        # 5. PA4 + CMTb4
        x_pa4 = self.pa4(x_cmtb3)
        x_cmtb4 = self.cmt4(x_pa4)     
        norm_layer = getattr(self, f'norm3')

        x_out = x_cmtb4.permute(0, 3, 2, 1).contiguous()
        x_out = norm_layer(x_out)
        x_out = x_out.permute(0, 3, 2, 1).contiguous()
        self.features.append(x_out)

        # # 6. Avg
        # x_avg = self.avg(x_cmtb4)
        # x_avg = x_avg.squeeze()

        # # 7. Linear + Classifier
        # x_fc = self.fc(x_avg)
        # out = self.classifier(x_fc)

        return self.features

class CMT_Layer_Select(t.nn.Module):
    """Define CMT model"""

    def __init__(self,
                 cmt_channelses = [46, 92, 184, 368],
                 pa_channelses = [46, 92, 184, 368],
                 R = 3.6,
                 repeats = [2, 2, 10, 2],
                 input_width = 640,
                 input_height = 192,
                 start_layer = 2,
                 use_upconv = True):
        """
        Args :
            --in_channels: default is 3
            --stem_channels: stem channels, default is 16
            --cmt_channelses: list, default is [46, 92, 184, 368]
            --pa_channels: patch aggregation channels, list, default is [46, 92, 184, 368]
            --R: expand ratio, default is 3.6
            --repeats: list, to specify how many CMT blocks stacked together, default is [2, 2, 10, 2]
            --input_size: default is 224
            --num_classes: default is 1000 for ImageNet
        """
        super(CMT_Layer_Select, self).__init__()

        widths = [int(input_width/4), int(input_width/8), int(input_width/16), int(input_width/32)]
        heights = [int(input_height/4), int(input_height/8), int(input_height/16), int(input_height/32)]
        self.start_layer = start_layer

        

        
        self.pa2 = PatchAggregation(in_channels = cmt_channelses[0], out_channels = pa_channelses[1])
        self.pa3 = PatchAggregation(in_channels = cmt_channelses[1], out_channels = pa_channelses[2])
        self.pa4 = PatchAggregation(in_channels = cmt_channelses[2], out_channels = pa_channelses[3])

        if use_upconv ==False:
            if start_layer==2:
                self.pa2 = PatchAggregation(in_channels = 64, out_channels = pa_channelses[1])
            elif start_layer==3:
                self.pa3 = PatchAggregation(in_channels = 128, out_channels = pa_channelses[2])
            elif start_layer==4:
                self.pa4 = PatchAggregation(in_channels = 256, out_channels = pa_channelses[3])

        cmt2 = []
        for _ in range(repeats[1]):
            cmt_layer = CMTBlock(input_width = widths[1],
                                 input_height = heights[1],
                                 kernel_size = 4,
                                 d_k = cmt_channelses[1] // 2,
                                 d_v = cmt_channelses[1] // 2,
                                 num_heads = 2,
                                 R = R, in_channels = pa_channelses[1])
            cmt2.append(cmt_layer)
        self.cmt2 = t.nn.Sequential(*cmt2)

        cmt3 = []
        for _ in range(repeats[2]):
            cmt_layer = CMTBlock(input_width = widths[2],
                                 input_height = heights[2],
                                 kernel_size = 2,
                                 d_k = cmt_channelses[2] // 4,
                                 d_v = cmt_channelses[2] // 4,
                                 num_heads = 4,
                                 R = R, in_channels = pa_channelses[2])
            cmt3.append(cmt_layer)
        self.cmt3 = t.nn.Sequential(*cmt3)

        cmt4 = []
        for _ in range(repeats[3]):
            cmt_layer = CMTBlock(input_width = widths[3],
                                 input_height = heights[3],
                                 kernel_size = 1,
                                 d_k = cmt_channelses[3] // 8,
                                 d_v = cmt_channelses[3] // 8,
                                 num_heads = 8,
                                 R = R, in_channels = pa_channelses[3])
            cmt4.append(cmt_layer)
        self.cmt4 = t.nn.Sequential(*cmt4)

        # 4. Global Avg Pool
        #self.avg = t.nn.AdaptiveAvgPool2d(1)        
        numl_layer = len(cmt_channelses)
        num_features = [int(cmt_channelses[0] * 2 ** i) for i in range(numl_layer)]
        self.num_features = num_features

        # add a norm layer for each output
        norm_layer = t.nn.LayerNorm
        out_indices=(0, 1, 2, 3)
        self.out_indices = out_indices
        for i_layer in out_indices:
            layer = norm_layer(num_features[i_layer])
            layer_name = f'norm{i_layer}'
            self.add_module(layer_name, layer)

    def forward(self, x):
        
        self.features = []        
        #3. PA2 + CMTb2
        
        if self.start_layer<3:            
            x_pa2 = self.pa2(x)
            x_cmtb2 = self.cmt2(x_pa2)
            
            norm_layer = getattr(self, f'norm1')
            x_out = x_cmtb2.permute(0, 3, 2, 1).contiguous()
            x_out = norm_layer(x_out)
            x_out = x_out.permute(0, 3, 2, 1).contiguous()
            self.features.append(x_out)
        else:
            x_cmtb2 = x
        
        # 4. PA3 + CMTb3
        if self.start_layer<4:            
            x_pa3 = self.pa3(x_cmtb2)
            x_cmtb3 = self.cmt3(x_pa3)
            
            norm_layer = getattr(self, f'norm2')
            x_out = x_cmtb3.permute(0, 3, 2, 1).contiguous()
            x_out = norm_layer(x_out)
            x_out = x_out.permute(0, 3, 2, 1).contiguous()
            self.features.append(x_out)
        else:
            x_cmtb3 = x

        # 5. PA4 + CMTb4
        x_pa4 = self.pa4(x_cmtb3)
        x_cmtb4 = self.cmt4(x_pa4)     
        norm_layer = getattr(self, f'norm3')

        x_out = x_cmtb4.permute(0, 3, 2, 1).contiguous()
        x_out = norm_layer(x_out)
        x_out = x_out.permute(0, 3, 2, 1).contiguous()
        self.features.append(x_out)


        return self.features

#########################
#      CMT Models       #
#########################
# 0 . Just Layer
class CMT_Layer(t.nn.Module):
    """Define CMT-Ti model"""

    def __init__(self, input_width = 640, input_height = 192, embed_dim = 46, start_layer= 2, use_upconv = True):
        """
        Args :
            --in_channels: default is 3
            --input_size: default is 224
            --num_classes: default is 1000 for ImageNet
        """
        super(CMT_Layer, self).__init__()

        self.cmt_layer = CMT_Layer_Select(
                          cmt_channelses = [embed_dim, embed_dim *2 , embed_dim*4, embed_dim * 8],
                          pa_channelses = [embed_dim, embed_dim *2 , embed_dim*4, embed_dim * 8],
                          R = 3.6,
                          repeats = [2, 2, 2, 2],
                          input_width = input_width,
                          input_height = input_height,
                          start_layer=start_layer,
                          use_upconv = use_upconv)

    def forward(self, x):
        x = self.cmt_layer(x)
        return x
    
# 1. CMT-Ti
class CMT_Ti(t.nn.Module):
    """Define CMT-Ti model"""

    def __init__(self, in_channels = 3, input_width = 640, input_height = 192, embed_dim = 46):
        """
        Args :
            --in_channels: default is 3
            --input_size: default is 224
            --num_classes: default is 1000 for ImageNet
        """
        super(CMT_Ti, self).__init__()

        self.cmt_ti = CMT(in_channels = in_channels,
                          stem_channels = 16,
                          cmt_channelses = [embed_dim, embed_dim *2 , embed_dim*4, embed_dim * 8],
                          pa_channelses = [embed_dim, embed_dim *2 , embed_dim*4, embed_dim * 8],
                          R = 3.6,
                          repeats = [2, 2, 2, 2],
                          input_width = input_width,
                          input_height = input_height)

    def forward(self, x):

        x = self.cmt_ti(x)

        return x


# 2. CMT-XS
class CMT_XS(t.nn.Module):
    """Define CMT-XS model"""

    def __init__(self, in_channels = 3, input_size = 224,embed_dim = 52):
        """
        Args :
            --in_channels: default is 3
            --input_size: default is 224
            --num_classes: default is 1000 for ImageNet
        """
        super(CMT_XS, self).__init__()

        self.cmt_xs = CMT(in_channels = in_channels,
                          stem_channels = 16,
                          cmt_channelses = [embed_dim, embed_dim *2 , embed_dim*4, embed_dim * 8],
                          pa_channelses = [embed_dim, embed_dim *2 , embed_dim*4, embed_dim * 8],
                          R = 3.8,
                          repeats = [3, 3, 3, 3],
                          input_size = input_size)

    def forward(self, x):

        x = self.cmt_xs(x)

        return x
# 2. CMT-XS
class CMT_XS2(t.nn.Module):
    """Define CMT-XS model"""

    def __init__(self, in_channels = 3, input_size = 224,embed_dim = 52):
        """
        Args :
            --in_channels: default is 3
            --input_size: default is 224
            --num_classes: default is 1000 for ImageNet
        """
        super(CMT_XS2, self).__init__()

        self.cmt_xs = CMT(in_channels = in_channels,
                          stem_channels = 16,
                          cmt_channelses = [embed_dim, embed_dim *2 , embed_dim*4, embed_dim * 8],
                          pa_channelses = [embed_dim, embed_dim *2 , embed_dim*4, embed_dim * 8],
                          R = 3.8,
                          repeats = [4, 4, 4, 4],
                          input_size = input_size)

    def forward(self, x):

        x = self.cmt_xs(x)

        return x

# 3. CMT-S
class CMT_S(t.nn.Module):
    """Define CMT-S model"""

    def __init__(self, in_channels = 3, input_size = 224,embed_dim = 64):
        """
        Args :
            --in_channels: default is 3
            --input_size: default is 224
            --num_classes: default is 1000 for ImageNet
        """
        super(CMT_S, self).__init__()

        self.cmt_s = CMT(in_channels = in_channels,
                         stem_channels = 32,
                         cmt_channelses = [embed_dim, embed_dim *2 , embed_dim*4, embed_dim * 8],
                         pa_channelses = [embed_dim, embed_dim *2 , embed_dim*4, embed_dim * 8],
                         R = 4.,
                         repeats = [3, 3, 16, 3],
                         input_size = input_size)

    def forward(self, x):

        x = self.cmt_s(x)

        return x


# 4. CMT-B
class CMT_B(t.nn.Module):
    """Define CMT-B model"""

    def __init__(self, in_channels = 3, input_size = 224, embed_dim = 76):
        """
        Args :
            --in_channels: default is 3
            --input_size: default is 224
            --num_classes: default is 1000 for ImageNet
        """
        super(CMT_B, self).__init__()

        self.cmt_b = CMT(in_channels = in_channels,
                         stem_channels = 38,
                         cmt_channelses = [embed_dim, embed_dim *2 , embed_dim*4, embed_dim * 8],
                         pa_channelses = [embed_dim, embed_dim *2 , embed_dim*4, embed_dim * 8],
                         R = 4.,
                         repeats = [2, 2, 2, 2],
                         input_size = input_size)

    def forward(self, x):

        x = self.cmt_b(x)

        return x
