export PYTHONPATH="${PYTHONPATH}:/home/sj/src/manydepthformer" 


#DATA_PATH=/home/sj/kitti
#WEIGHT_PATH=/home/sj/manydepth/kitti_resnet_attention

DATA_PATH=/media/sj/data2/cityscapes
#WEIGHT_PATH=/media/sj/data2/manydepth/pc2/city_cmt_22_attention_s2_r02_56k_r1_adam_s5r01

SET=(18 19 38 39)

for i in ${SET[@]}
do 
 #python manydepth/evaluate_depth_2.py --data_path=$DATA_PATH --load_weights_folder=$WEIGHT_PATH/mdp/models/weights_$i --eval_mono --train_model=resnet --use_attention_decoder --png
 python manydepth/evaluate_depth_2.py --data_path=$DATA_PATH --load_weights_folder=$WEIGHT_PATH/mdp/models/weights_$i --eval_split=cityscapes --eval_mono --train_model=cmt --cmt_layer=3 --use_attention_decoder --png
done





