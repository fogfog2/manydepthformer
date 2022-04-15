export PYTHONPATH="${PYTHONPATH}:/home/sj/src/manydepthformer" 


#DATA_PATH=/home/sj/kitti
#WEIGHT_PATH=/home/sj/manydepth/kitti_resnet_attention

DATA_PATH=/media/sj/data2/cityscapes
WEIGHT_PATH=/home/sj/manydepth/city_cmt_26_attention_s2_r05_48k_r2

SET=$(seq 18 19)
for i in $SET
do 
 #python manydepth/evaluate_depth_2.py --data_path=$DATA_PATH --load_weights_folder=$WEIGHT_PATH/mdp/models/weights_$i --eval_mono --train_model=resnet --use_attention_decoder --png
 python manydepth/evaluate_depth_2.py --data_path=$DATA_PATH --load_weights_folder=$WEIGHT_PATH/mdp/models/weights_$i --eval_split=cityscapes --eval_mono --train_model=cmt --cmt_layer=3 --use_attention_decoder --png
done





