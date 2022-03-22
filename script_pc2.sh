export PYTHONPATH="${PYTHONPATH}:/home/sj/src/manydepthformer" 


DATA_PATH=/media/sj/data/cityscapes
WEIGHT_PATH=/media/sj/data/manydepth/pc3/city_cmt_22_attention/
SET=$(seq 35 39)
for i in $SET
do 
 python manydepth/evaluate_depth_2.py --data_path=$DATA_PATH --load_weights_folder=$WEIGHT_PATH/mdp/models/weights_$i --eval_mono --eval_split=cityscapes --train_model=cmt  --cmt_layer=3 --use_attention_decoder --png
done





