python main_base_bak.py ucf101 RGB --arch resnet50 --num_segments 8 --gd 20 --lr 0.02 --wd 1e-4 --lr_steps 20 40 --epochs 3 --batch-size 12 -j 4 --dropout 0.5 --consensus_type=avg --eval-freq=1 --npb --gpus 0 1 2 --exp_header toy