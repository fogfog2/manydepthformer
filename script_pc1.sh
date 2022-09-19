export PYTHONPATH="${PYTHONPATH}:/home/sj/src/manydepthformer" 


#kitti
DATA_PATH=/home/sj/kitti
WEIGHT_PATH=/home/sj/manydepth/kitti_cmt_attention_22_mw
#WEIGHT_PATH=/media/sj/data/manydepth/pc1/kitti_cmt_attention_22_mm

#citi
#DATA_PATH=/media/sj/data/cityscapes
#WEIGHT_PATH=/home/sj/manydepth/city_resnet_s2_r01_56k_r1_adam_s2r01
#WEIGHT_PATH=/media/sj/data/manydepth/pc2/city_resnet_s2_r01_56k_r1_adam_s2r01

#ucl
#DATA_PATH=/home/sj/colon
#WEIGHT_PATH=/home/sj/src/manydepth/colon_resnet




#set
#SET=(20 39)
#for i in ${SET[@]}

SET=$(seq 20 39)
for i in $SET
do 
#kitti
 python manydepth/evaluate_depth_2.py --data_path=$DATA_PATH --load_weights_folder=$WEIGHT_PATH/mdp/models/weights_$i --eval_mono --train_model=cmt --cmt_layer=3 --use_attention_decoder  --png
 #python manydepth/evaluate_depth_2.py --data_path=$DATA_PATH --load_weights_folder=$WEIGHT_PATH/mdp/models/weights_$i --eval_mono --train_model=resnet --png
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





