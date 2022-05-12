export PYTHONPATH="${PYTHONPATH}:/home/sj/src/manydepthformer" 


#kitti
DATA_PATH=/home/sj/kitti
WEIGHT_PATH=/media/sj/data2/manydepth_paper/t4_ablation_attention_decoder
python manydepth/evaluate_depth_2.py --data_path=$DATA_PATH --load_weights_folder=$WEIGHT_PATH --eval_mono --train_model=resnet  --use_attention_decoder --png


WEIGHT_PATH=/media/sj/data2/manydepth_paper/t4_ablation_hybrid_encoder
python manydepth/evaluate_depth_2.py --data_path=$DATA_PATH --load_weights_folder=$WEIGHT_PATH --eval_mono --train_model=cmt --cmt_layer=3 --png

