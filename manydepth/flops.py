import torch
import torchvision.models as models
from ptflops import get_model_complexity_info
#import ptflops
from manydepth import networks
with torch.cuda.device(0):

  #net = DepthResNet(version="18pt")
  #net = DepthResNetSwin(version="18pt")

  #net = DepthResNetCMT(version="18pt")

  net = networks.ResnetEncoderMatching(18, False,
                                             input_width=640,
                                             input_height=192,
                                             adaptive_bins=True,
                                             min_depth_bin=0.1,
                                             max_depth_bin=20.0,
                                             depth_binning='linear',
                                             num_depth_bins=96)

  macs, params = get_model_complexity_info(net, [(3, 192, 640), as_strings=True,
                                           print_per_layer_stat=True, verbose=True)
  print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
  print('{:<30}  {:<8}'.format('Number of parameters: ', params))