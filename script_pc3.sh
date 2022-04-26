export PYTHONPATH="${PYTHONPATH}:/home/sj/src/manydepthformer" 


DATA_PATH=/home/sj/kitti
WEIGHT_PATH=/home/sj/manydepth/kitti_cmt_monocmt_r18
#WEIGHT_PATH=/media/sj/data2/manydepth/pc6/kitti_cmt_22_attention_adam_s5r01

#DATA_PATH=/media/sj/data2/cityscapes
#WEIGHT_PATH=/home/sj/manydepth/city_cmt_22_attention_s2_r01_56k_r1_adam_s2r01_t50

SET=(20 21 22 23)
for i in ${SET[@]}
do 
 python manydepth/evaluate_depth_2.py --data_path=$DATA_PATH --load_weights_folder=$WEIGHT_PATH/mdp/models/weights_$i --eval_mono --train_model=cmt --cmt_layer=3 --use_attention_decoder --png
 #python manydepth/evaluate_depth_2.py --data_path=$DATA_PATH --load_weights_folder=$WEIGHT_PATH/mdp/models/weights_$i --eval_split=cityscapes --eval_mono --train_model=cmt --cmt_layer=3 --use_attention_decoder --png
done





