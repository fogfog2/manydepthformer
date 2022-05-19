export PYTHONPATH="${PYTHONPATH}:/home/sj/src/manydepthformer" 


#cityscapes
DATA_PATH=/media/sj/data2/cityscapes
WEIGHT_PATH=/media/sj/data2/manydepth_paper/t3_cityscapes

python manydepth/evaluate_depth_2.py --data_path=$DATA_PATH --load_weights_folder=$WEIGHT_PATH --eval_split=cityscapes --eval_mono --train_model=cmt --cmt_layer=3 --use_attention_decoder --png
