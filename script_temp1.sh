export PYTHONPATH="${PYTHONPATH}:/home/sj/src/manydepthformer" 


 
python manydepth/train.py --data_path=/home/sj/kitti --log_dir=/home/sj/manydepth/kitti_cmt_s5_lr2e4_r05 --batch_size=8 --height=192 --width=640 --num_epochs=40 --freeze_teacher_epoch=15 --learning_rate=2e-4 --scheduler_step_size=5 --scheduler_step_ratio=0.5 --scheduler_step_freeze_after_size=5 --scheduler_step_freeze_after_ratio=0.5 --cmt_layer=3 --train_model=cmt --use_attention_decoder --png --cuda_device=1
 

   
