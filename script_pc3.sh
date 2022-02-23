export PYTHONPATH="${PYTHONPATH}:/home/sj/src/manydepthformer" 


DATA_PATH=/home/sj/kitti
WEIGHT_PATH=/home/sj/manydepth/kitti_cmt_attention_cw_t50
SET=$(seq 25 35)
for i in $SET
do 
 python manydepth/evaluate_depth_2.py --data_path=$DATA_PATH --load_weights_folder=$WEIGHT_PATH/mdp/models/weights_$i --eval_mono --cmt_layer=2 --cmt_dim=46 --train_model=cmt --use_attention_decoder --attention_only_channel --png
done





