export PYTHONPATH="${PYTHONPATH}:/home/sj/src/manydepthformer" 

python manydepth/train.py --data_path=/home/sj/kitti --log_dir=/media/sj/data/manydepth/kitti_cmt_l2_b8 --batch_size=8 --height=640 --width=192 --num_epochs=20 --freeze_teacher_epoch=15 --scheduler_step_size=5 --cmt_use_upconv --cmt_layer=2 --cmt_dim=46 --train_mode=cmt --png

python manydepth/evaluate_depth_2.py --data_path=/home/sj/kitti --load_weights_folder=/media/sj/data/manydepth/kitti_cmt_l2_b8/mdp/models/weights_19 -eval_mono --cmt_use_upconv --cmt_layer=2 --cmt_dim=46 --png

python manydepth/train.py --data_path=/home/sj/kitti --log_dir=/media/sj/data/manydepth/kitti_cmt_l3_b8 --batch_size=8 --height=640 --width=192 --num_epochs=20 --freeze_teacher_epoch=15 --scheduler_step_size=5 --cmt_use_upconv --cmt_layer=3 --cmt_dim=46 --train_mode=cmt --png

python manydepth/evaluate_depth_2.py --data_path=/home/sj/kitti --load_weights_folder=/media/sj/data/manydepth/kitti_cmt_l3_b8/mdp/models/weights_19 -eval_mono --cmt_use_upconv --cmt_layer=3 --cmt_dim=46 --png


python manydepth/train.py --data_path=/home/sj/kitti --log_dir=/media/sj/data/manydepth/kitti_cmt_l4_b8 --batch_size=8 --height=640 --width=192 --num_epochs=20 --freeze_teacher_epoch=15 --scheduler_step_size=5 --cmt_use_upconv --cmt_layer=4 --cmt_dim=46 --train_mode=cmt --png

python manydepth/evaluate_depth_2.py --data_path=/home/sj/kitti --load_weights_folder=/media/sj/data/manydepth/kitti_cmt_l4_b8/mdp/models/weights_19 -eval_mono --cmt_use_upconv --cmt_layer=4 --cmt_dim=46 --png
