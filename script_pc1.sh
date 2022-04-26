export PYTHONPATH="${PYTHONPATH}:/home/sj/src/manydepthformer" 


#kitti
DATA_PATH=/home/sj/kitti
WEIGHT_PATH=/home/sj/manydepth/kitti_cmt_22_attention_s2_r01_56k_r1_adam_s5r03


#citi
#DATA_PATH=/media/sj/data/cityscapes
#WEIGHT_PATH=/home/sj/manydepth/city_resnet_s2_r01_56k_r1_adam_s2r01
#WEIGHT_PATH=/media/sj/data/manydepth/pc2/city_resnet_s2_r01_56k_r1_adam_s2r01


#set
SET=(18 19 38 39)
for i in ${SET[@]}
do 
#kitti
 python manydepth/evaluate_depth_2.py --data_path=$DATA_PATH --load_weights_folder=$WEIGHT_PATH/mdp/models/weights_$i --eval_mono --train_model=cmt --cmt_layer=3 --use_attention_decoder --png
#city
 #python manydepth/evaluate_depth_2.py --data_path=$DATA_PATH --load_weights_folder=$WEIGHT_PATH/mdp/models/weights_$i --eval_split=cityscapes --eval_mono --train_model=resnet --png
done

#seq
# SET=$(seq 18 19)
# for i in $SET
# do 
#  #python manydepth/evaluate_depth_2.py --data_path=$DATA_PATH --load_weights_folder=$WEIGHT_PATH/mdp/models/weights_$i --eval_mono --train_model=resnet --use_attention_decoder --png
#  python manydepth/evaluate_depth_2.py --data_path=$DATA_PATH --load_weights_folder=$WEIGHT_PATH/mdp/models/weights_$i --eval_split=cityscapes --eval_mono --train_model=cmt --cmt_layer=3 --use_attention_decoder --png
# done





