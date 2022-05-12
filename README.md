# manydepthformer

Reference Code 

- https://github.com/nianticlabs/manydepth
    
   
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
  




  [Manydepthformer_KITTI(640x192)]: <https://drive.google.com/drive/folders/1wIvP0uVmxIJBeGKr-7s09qITFAtxT0tI?usp=sharing>
  [Manydepthformer_KITTI_T50(640x192)]: <https://drive.google.com/drive/folders/1C2JTB01F_b0HqajTRmj59CnJsdv4FHOI?usp=sharing>
  [Manydepthformer_Cityscapes(640x192)]: <https://drive.google.com/drive/folders/1eXevxF_doANjD_cnQWgzNqJWpWfr0f9G?usp=sharing> 
  [Manydepthformer Ablation Encoder(640x192)]: <https://drive.google.com/drive/folders/1rmH9e9l1Pd6o3q5Iq3AXUELRljfQi1YJ?usp=sharing>
  [Manydepthformer Ablation Decoder(640x192)]: <https://drive.google.com/drive/folders/1xdiwXmAtD6wpO09L8GKwuCspQesH9uf5?usp=sharing>
  
  [script_table2_eval.sh]: <https://github.com/fogfog2/manydepthformer/blob/master/script_table2_eval.sh>
  [script_table3_eval.sh]: <https://github.com/fogfog2/manydepthformer/blob/master/script_table3_eval.sh>
  [script_table4_eval.sh]: <https://github.com/fogfog2/manydepthformer/blob/master/script_table4_eval.sh>
  [.vscode/launch.json]: <https://github.com/fogfog2/manydepthformer/blob/master/.vscode/launch.json>
