export PYTHONPATH="${PYTHONPATH}:/home/sj/src/manydepthformer" 

python manydepth/train.py --data_path=/home/sj/Documents/cityscapes_preprocessed --log_dir=/home/sj/manydepth/city_cmt_22_attention_s2_r01_64k_r1_adam --dataset=cityscapes_preprocessed --split=cityscapes_preprocessed --batch_size=8 --width=512 --height=192 --num_epochs=40 --scheduler_step_size=2 --scheduler_step_ratio=0.1 --pytorch_random_seed=1 --cmt_layer=3 --freeze_teacher_step=64000 --train_mode=cmt --use_attention_decoder --png
