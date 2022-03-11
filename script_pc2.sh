export PYTHONPATH="${PYTHONPATH}:/home/sj/src/manydepthformer" 


DATA_PATH=/home/sj/kitti
WEIGHT_PATH=/home/sj/manydepth/kitti_resnet_attention
SET=$(seq 20 39)
for i in $SET
do 
 python manydepth/evaluate_depth_2.py --data_path=$DATA_PATH --load_weights_folder=$WEIGHT_PATH/mdp/models/weights_$i --eval_mono --train_model=resnet --use_attention_decoder --png
done





