export PYTHONPATH="${PYTHONPATH}:/home/sj/src/manydepthformer" 


#DATA_PATH=/home/sj/kitti
#WEIGHT_PATH=/home/sj/manydepth/kitti_resnet_attention

DATA_PATH=/media/sj/data/cityscapes
WEIGHT_PATH=/media/sj/data/manydepth/pc6/city_cmt_22_attention_s2_r01_56k_r1_adam_s5r01

SET=$(seq 18 19)
for i in $SET
do 
 #python manydepth/evaluate_depth_2.py --data_path=$DATA_PATH --load_weights_folder=$WEIGHT_PATH/mdp/models/weights_$i --eval_mono --train_model=resnet --use_attention_decoder --png
 python manydepth/evaluate_depth_2.py --data_path=$DATA_PATH --load_weights_folder=$WEIGHT_PATH/mdp/models/weights_$i --eval_split=cityscapes --eval_mono --train_model=cmt --cmt_layer=3 --use_attention_decoder --png
done




