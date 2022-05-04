import torch
import torchvision.models as models
from ptflops import get_model_complexity_info
#import ptflops
from manydepth import networks

from manydepth.options import MonodepthOptions



# current_image = torch.FloatTensor(3, self.opt.height, self.opt.width)
# lookup_images = torch.FloatTensor(3,1, self.opt.height, self.opt.width)
# poses = torch.FloatTensor(1,4,4)
# K = torch.FloatTensor(4,4)
# invK = torch.FloatTensor(4,4)

# res = dict(current_image=current_image, lookup_images = lookup_images, poses= poses, k=K, invK=invK)


def prepare_input(resolution):
    current_image = torch.FloatTensor(1, *resolution)
    lookup_images = torch.FloatTensor(1, 1, *resolution)
    poses = torch.FloatTensor(1, 1,4,4)
    K = torch.FloatTensor(1, 4,4)
    invK = torch.FloatTensor(1, 4,4)
    min_depth = 0.1
    max_depth=10.0
    return  dict(current_image=current_image, lookup_images = lookup_images, poses= poses, K=K, invK=invK, min_depth_bin=min_depth, max_depth_bin=max_depth)

def prepare_decoder_input_resnet(resolution):
    x1 = torch.FloatTensor(2, 64, resolution[0]//2, resolution[1]//2)
    x2 = torch.FloatTensor(2, 64, resolution[0]//4, resolution[1]//4)
    x3 = torch.FloatTensor(2, 128, resolution[0]//8, resolution[1]//8)
    x4 = torch.FloatTensor(2, 256, resolution[0]//16, resolution[1]//16)
    x5 = torch.FloatTensor(2, 512, resolution[0]//32, resolution[1]//32)
    return  dict(input_features=[x1, x2, x3,x4,x5])

def prepare_decoder_input_cmt_l2(resolution):
    x1 = torch.FloatTensor(2, 64, resolution[0]//2, resolution[1]//2)
    x2 = torch.FloatTensor(2, 64, resolution[0]//4, resolution[1]//4)
    x3 = torch.FloatTensor(2, 92, resolution[0]//8, resolution[1]//8)
    x4 = torch.FloatTensor(2, 184, resolution[0]//16, resolution[1]//16)
    x5 = torch.FloatTensor(2, 368, resolution[0]//32, resolution[1]//32)
    return  dict(input_features=[x1, x2, x3,x4,x5])

def prepare_decoder_input_cmt_l3(resolution):
    x1 = torch.FloatTensor(2, 64, resolution[0]//2, resolution[1]//2)
    x2 = torch.FloatTensor(2, 64, resolution[0]//4, resolution[1]//4)
    x3 = torch.FloatTensor(2, 128, resolution[0]//8, resolution[1]//8)
    x4 = torch.FloatTensor(2, 184, resolution[0]//16, resolution[1]//16)
    x5 = torch.FloatTensor(2, 368, resolution[0]//32, resolution[1]//32)
    return  dict(input_features=[x1, x2, x3,x4,x5])


class Flops:
    def __init__(self, options):
      self.opt = options.parse()

    def run(self):            
        with torch.cuda.device(0):                     
              encoder_model = self.opt.train_model 
              if "resnet" in encoder_model:            
                  net = networks.ResnetEncoderMatching(
                      self.opt.num_layers, self.opt.weights_init == "pretrained",
                      input_height=self.opt.height, input_width=self.opt.width,
                      adaptive_bins=True, min_depth_bin=0.1, max_depth_bin=20.0,
                      depth_binning=self.opt.depth_binning, num_depth_bins=self.opt.num_depth_bins)

              elif "swin" in encoder_model:
                  net = networks.SwinEncoderMatching(
                      self.opt.num_layers, self.opt.weights_init == "pretrained",
                      input_height=self.opt.height, input_width=self.opt.width,
                      adaptive_bins=True, min_depth_bin=0.1, max_depth_bin=20.0,
                      depth_binning=self.opt.depth_binning, num_depth_bins=self.opt.num_depth_bins, use_swin_feature = self.opt.swin_use_feature)

              elif "cmt" in encoder_model:
                  net = networks.CMTEncoderMatching(
                      self.opt.num_layers, self.opt.weights_init == "pretrained",
                      input_height=self.opt.height, input_width=self.opt.width,
                      adaptive_bins=True, min_depth_bin=0.1, max_depth_bin=20.0,
                      depth_binning=self.opt.depth_binning, num_depth_bins=self.opt.num_depth_bins, upconv = self.opt.cmt_use_upconv, start_layer = self.opt.cmt_layer, embed_dim = self.opt.cmt_dim, use_cmt_feature = self.opt.cmt_use_feature)


              if self.opt.use_attention_decoder:            
                  decoder = \
                      networks.DepthDecoderAttention(net.num_ch_enc, self.opt.scales, no_spatial= self.opt.attention_only_channel)            
              else:
                  decoder = \
                      networks.DepthDecoder(net.num_ch_enc, self.opt.scales)


              macs, params = get_model_complexity_info(net, (3,self.opt.height,self.opt.width)  ,input_constructor=prepare_input, as_strings=True, print_per_layer_stat=True, verbose=True)
              
              if "resnet" in self.opt.train_model:
                  prepare_decoder_input = prepare_decoder_input_resnet
              elif "cmt" in self.opt.train_model:
                  if self.opt.cmt_layer==2:
                    prepare_decoder_input = prepare_decoder_input_cmt_l2
                  elif self.opt.cmt_layer==3:
                    prepare_decoder_input = prepare_decoder_input_cmt_l3

              d_macs, d_params = get_model_complexity_info(decoder, (self.opt.height,self.opt.width)  ,input_constructor=prepare_decoder_input, as_strings=True, print_per_layer_stat=True, verbose=True)

              print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
              print('{:<30}  {:<8}'.format('Number of parameters: ', params))
              print('{:<30}  {:<8}'.format('Decoder Computational complexity: ', d_macs))
              print('{:<30}  {:<8}'.format('Decoder Number of parameters: ', d_params))

        #net = DepthResNet(version="18pt")
        #net = DepthResNetSwin(version="18pt")

        #net = DepthResNetCMT(version="18pt")

        # net = networks.ResnetEncoderMatching(18, False,
        #                                            input_width=640,
        #                                            input_height=192,
        #                                            adaptive_bins=True,
        #                                            min_depth_bin=0.1,
        #                                            max_depth_bin=20.0,
        #                                            depth_binning='linear',
        #                                            num_depth_bins=96)

        


if __name__ == '__main__':    
    options = MonodepthOptions()
    flops = Flops(options)
    flops.run()
