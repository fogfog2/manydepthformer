# manydepthformer

Reference Code 

- https://github.com/nianticlabs/manydepth
    
## Overview

Paper:

Self-Supervised Monocular Depth Estimation Using Hybrid Transformer Encoder

- https://ieeexplore.ieee.org/document/9864127

Hybrid Encoder-Decoder 

![structure](https://github.com/fogfog2/manydepthformer/blob/master/assets/manydepthformer_graphical_abstract.png)


Structure

![detailed_structure](https://github.com/fogfog2/manydepthformer/blob/master/assets/manydepthformer_structure.png)


## Pretrained weights and evaluation

You can download weights for some pretrained models here

- [Manydepthformer_KITTI(640x192)]
- [Manydepthformer_KITTI_T50(640x192)]
- [Manydepthformer_Cityscapes(640x192)]
- [Manydepthformer Ablation Encoder(640x192)]
- [Manydepthformer Ablation Decoder(640x192)]

You can use this script to simple evaluate

- [script_table2_eval.sh]
- [script_table3_eval.sh]
- [script_table4_eval.sh]


## Environmental Setting 

Anaconda 

    conda create -n manydepthformer python=3.7
    conda activate manydepthformer
    pip install numpy
    conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch -c nvidia
    pip install protobuf scipy opencv-python matplotlib scikit-image tensorboardX


VSCode Launch (train, evalute, test)

   [.vscode/launch.json]
  




  [Manydepthformer_KITTI(640x192)]: <https://drive.google.com/file/d/124msgklLiVZn1ECe3xSclUB3YeI1ebuh/view?usp=sharing>
  [Manydepthformer_KITTI_T50(640x192)]: <https://drive.google.com/file/d/1jjVNfSRbJHzjIc8CENZK9L0q8t5jTAYD/view?usp=sharing>
  [Manydepthformer_Cityscapes(640x192)]: <https://drive.google.com/file/d/1YRtrw5XXK0Is3fDCUGbktw3y9hixksDB/view?usp=sharing> 
  [Manydepthformer Ablation Encoder(640x192)]: <https://drive.google.com/file/d/1peCTWCx7erRzbqgW2lyqkz29MqQMxJEo/view?usp=sharing>
  [Manydepthformer Ablation Decoder(640x192)]: <https://drive.google.com/file/d/14EGav5Xim83MBRJXB35eKhbMfQBw_Y6G/view?usp=sharing>
  
  [script_table2_eval.sh]: <https://github.com/fogfog2/manydepthformer/blob/master/script_table2_eval.sh>
  [script_table3_eval.sh]: <https://github.com/fogfog2/manydepthformer/blob/master/script_table3_eval.sh>
  [script_table4_eval.sh]: <https://github.com/fogfog2/manydepthformer/blob/master/script_table4_eval.sh>
  [.vscode/launch.json]: <https://github.com/fogfog2/manydepthformer/blob/master/.vscode/launch.json>
