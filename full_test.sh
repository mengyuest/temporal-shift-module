# Test for Adaptive
python main.py actnet RGB --arch resnet50 --num_segments 16 --lr 0.0005 --epochs 50 --npb  --exp_header X --ada_reso_skip --policy_backbone mobilenet_v2 --reso_list 224 168 112 84 --backbone_list resnet50 resnet34 resnet18 --skip_list 1 2 4 --accuracy_weight 0.9 --efficency_weight 0.1 --folder_suffix _self --exp_decay --init_tau 0.000001 --policy_also_backbone --policy_input_offset 3 --uniform_loss_weight 3.0 --use_gflops_loss --batch-size 48 -j 36 --gpus 0 1 2 3 --test_from act_best_practice --uno_time

# Test for Adaptive EfficientNet


# Test for SCSampler


# Test for Baseline TSN


# Test for Baseline LSTM


# Test for Adaptive ALL


# Test for Adaptive RANDOM

