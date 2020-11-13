# Baseline EfficientNet-b0
python main_gate.py something RGB --arch effb0 --num_segments 8 --lr 0.01 --lr_steps 20 40 --epochs 50 --wd 5e-4 --npb --batch-size 64 -j 72 --exp_header sthv1_seg8_effb0_b64_lr.01step --gpus 0 1 2 3

python main_gate.py something RGB --arch effb1 --num_segments 8 --lr 0.01 --lr_steps 20 40 --epochs 50 --wd 5e-4 --npb --batch-size 64 -j 72 --exp_header sthv1_seg8_effb1_b64_lr.01step --gpus 0 1 2 3

python main_gate.py something RGB --arch effb2 --num_segments 8 --lr 0.01 --lr_steps 20 40 --epochs 50 --wd 5e-4 --npb --batch-size 64 -j 72 --exp_header sthv1_seg8_effb2_b64_lr.01step --gpus 0 1 2 3

python main_gate.py something RGB --arch effb3 --num_segments 8 --lr 0.01 --lr_steps 20 40 --epochs 50 --wd 5e-4 --npb --batch-size 64 -j 72 --exp_header sthv1_seg8_effb3_b64_lr.01step --gpus 0 1 2 3

python main_gate.py something RGB --arch effb4 --num_segments 8 --lr 0.01 --lr_steps 20 40 --epochs 50 --wd 5e-4 --npb --batch-size 64 -j 72 --exp_header sthv1_seg8_effb4_b64_lr.01step --gpus 0 1 2 3

python main_gate.py something RGB --arch effb5 --num_segments 8 --lr 0.01 --lr_steps 20 40 --epochs 50 --wd 5e-4 --npb --batch-size 64 -j 72 --exp_header sthv1_seg8_effb5_b64_lr.01step --gpus 0 1 2 3

python main_gate.py something RGB --arch effb6 --num_segments 8 --lr 0.01 --lr_steps 20 40 --epochs 50 --wd 5e-4 --npb --batch-size 64 -j 72 --exp_header sthv1_seg8_effb6_b64_lr.01step --gpus 0 1 2 3



# Baseline MobileNet-v2
python main_gate.py something RGB --arch mobilenetv2 --num_segments 8 --lr 0.01 --lr_steps 20 40 --epochs 50 --wd 5e-4 --npb --batch-size 64 -j 72 --exp_header sthv1_seg8_mobv2_b64_lr.01step --gpus 0 1 2 3




# Adaptive version of it