export PYTHONPATH="${PYTHONPATH}:/home/sj/src/manydepthformer" 


# DATA_PATH=/home/sj/kitti
# WEIGHT_PATH=/home/sj/manydepth/test/kitti_resnet_attention_

#DATA_PATH=/media/sj/data2/cityscapes
#WEIGHT_PATH=/media/sj/data2/manydepth/pc2/city_cmt_22_attention_s2_r02_56k_r1_adam_s5r01



DATA_PATH=/home/sj/colon
WEIGHT_PATH=/home/sj/manydepth/colon_cmt_attention_6e5

#SET=(18 19 38 39)

SET=$(seq 38 39)
for i in $SET
do  python manydepth/evaluate_depth_2.py --data_path=$DATA_PATH --load_weights_folder=$WEIGHT_PATH/mdp/models/weights_$i --eval_mono --eval_split=custom_ucl  --train_model=cmt --cmt_layer=2 --use_attention_decoder --png
 #python manydepth/evaluate_depth_2.py --data_path=$DATA_PATH --load_weights_folder=$WEIGHT_PATH/mdp/models/weights_$i --eval_split=cityscapes --eval_mono --train_model=cmt --cmt_layer=3 --use_attention_decoder --png
done





