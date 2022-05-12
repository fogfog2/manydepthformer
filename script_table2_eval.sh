export PYTHONPATH="${PYTHONPATH}:/home/sj/src/manydepthformer" 


#kitti
DATA_PATH=/home/sj/kitti
WEIGHT_PATH=/media/sj/data2/manydepth_paper/t2_our_full
python manydepth/evaluate_depth_2.py --data_path=$DATA_PATH --load_weights_folder=$WEIGHT_PATH/model --eval_mono --train_model=cmt --cmt_layer=3 --use_attention_decoder --png

WEIGHT_PATH=/media/sj/data2/manydepth_paper/t2_our_full_t50
python manydepth/evaluate_depth_2.py --data_path=$DATA_PATH --load_weights_folder=$WEIGHT_PATH/model --eval_mono --train_model=cmt --cmt_layer=3 --use_attention_decoder --png


