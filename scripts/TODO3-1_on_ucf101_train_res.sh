#TODO on satori or WSC, j=36 for 72

# TODO(phase 0) Base Models
# exp-0 resnet50, t=16, size 224, batchsize 72, train 10 epochs
python main.py ucf101 RGB --arch resnet50 --num_segments 16 --lr 0.02 --epochs 10 --batch-size 72 -j 36 --npb --rescale_to 224 --folder_suffix _self --exp_header ucf_res50_e10_224_0 --gpus 0 1 2 3 4 5 --with_test

# exp-1 resnet34, t=16, size 168, batchsize 72, train 10 epochs
python main.py ucf101 RGB --arch resnet34 --num_segments 16 --lr 0.02 --epochs 10 --batch-size 72 -j 36 --npb --rescale_to 168 --folder_suffix _self --exp_header ucf_res34_e10_168_0 --gpus 0 1 2 3 4 5 --with_test

# exp-2 resnet18, t=16, size 112, batchsize 72, train 10 epochs
python main.py ucf101 RGB --arch resnet18 --num_segments 16 --lr 0.02 --epochs 10 --batch-size 72 -j 36 --npb --rescale_to 112 --folder_suffix _self --exp_header ucf_res18_e10_112_0 --gpus 0 1 2 3 4 5 --with_test

# exp-3 resnet50, t=16, size 224, batchsize 48, train 50 epochs
python main.py ucf101 RGB --arch resnet50 --num_segments 16 --lr 0.02 --epochs 50 --lr_steps 20 40 --batch-size 48 -j 36 --npb --rescale_to 224 --folder_suffix _self --exp_header ucf_res50_s224_e50_b48 --gpus 0 1 2 3 4 5 --with_test

# exp-4 lstm, resnet50, t=16, size 224, batchsize 48, train 50 epochs
python main.py ucf101 RGB --arch resnet50 --num_segments 16 --lr 0.02 --epochs 50 --lr_steps 20 40 --batch-size 48 -j 36 --npb --folder_suffix _self --ada_reso_skip  --policy_backbone resnet50 --offline_lstm_all --exp_header ucf_res50_s224_e50_b48_lstm_all --gpus 0 1 2 3 4 5 --with_test

# TODO(phase 1) Baselines and Joint
# exp-5 joint
python main.py ucf101 RGB --arch resnet50 --num_segments 16 --lr 0.001 --epochs 50 --lr_steps 50 100 --batch-size 36 -j 48 --npb --folder_suffix _self --ada_reso_skip --policy_backbone mobilenet_v2 --reso_list 224 168 112 84 --backbone_list resnet50 resnet34 resnet18 --skip_list 1 2 4 --policy_also_backbone --policy_input_offset 3 --model_paths ucf_res50_e10_224_0/m.pth.tar ucf_res34_e10_168_0/m.pth.tar ucf_res18_e10_112_0/m.pth.tar --accuracy_weight 0.97 --efficency_weight 0.03 --uniform_loss_weight 3.0  --use_gflops_loss --exp_decay --init_tau 5 --exp_header ucf_res_3m124_a.97e.03_ed5_ft10ds_lr.001_gu3 --gpus 0 1 2 3 4 5 --with_test

# exp-6 random-policy
python main.py ucf101 RGB --arch resnet50 --num_segments 16 --lr 0.001 --epochs 50 --lr_steps 50 100 --batch-size 48 -j 12 --npb --folder_suffix _self --ada_reso_skip --policy_backbone mobilenet_v2 --reso_list 224 168 112 84 --backbone_list resnet50 resnet34 resnet18 --skip_list 1 2 4 --policy_also_backbone --policy_input_offset 3 --model_paths ucf_res50_e10_224_0/m.pth.tar ucf_res34_e10_168_0/m.pth.tar ucf_res18_e10_112_0/m.pth.tar  --use_gflops_loss  --exp_decay --init_tau 5 --random_policy --exp_header ucf_res_3m124_a.9e.1_ed5_ft10ds_lr.001_rand --gpus 0 1 2 3 4 5 --with_test

# exp-7 all-policy
python main.py ucf101 RGB --arch resnet50 --num_segments 16 --lr 0.001 --epochs 50 --lr_steps 50 100 --batch-size 48 -j 12 --npb --folder_suffix _self --ada_reso_skip --policy_backbone mobilenet_v2 --reso_list 224 168 112 84 --backbone_list resnet50 resnet34 resnet18 --skip_list 1 2 4 --policy_also_backbone --policy_input_offset 3 --model_paths ucf_res50_e10_224_0/m.pth.tar ucf_res34_e10_168_0/m.pth.tar ucf_res18_e10_112_0/m.pth.tar  --use_gflops_loss --exp_decay --init_tau 5 --all_policy --exp_header ucf_res_3m124_a.9e.1_ed5_ft10ds_lr.001_all --gpus 0 1 2 3 4 5 --with_test

# exp-8 scsampler (use 50 epochs) #TODO(PENDING)
python main.py ucf101 RGB --arch resnet50 --num_segments 16 --lr 0.02 --epochs 50 --lr_steps 20 40 --batch-size 48 -j 12 --npb --folder_suffix _self  --ada_reso_skip --policy_backbone mobilenet_v2 --backbone_list resnet50 --reso_list 224 84 --policy_input_offset 1 --model_paths ucf_res50_s224_e50_b48/models/ckpt.best.pth.tar --frame_independent --real_scsampler --consensus_type scsampler --freeze_backbone --exp_header ucf_res50_scs_ac_fd_k10_b48 --gpus 0 1 2 3 4 5 --with_test

# TODO(phase 2) Finetuning (remember smaller lr, --init_tau, and --freeze_policy, and --base_pretrained_from, and remove model_paths)
# exp-9 finetune
python main.py ucf101 RGB --arch resnet50 --num_segments 16 --lr 0.0005 --epochs 50 --lr_steps 50 100 --batch-size 48 -j 12 --npb --folder_suffix _self --ada_reso_skip --policy_backbone mobilenet_v2 --reso_list 224 168 112 84 --backbone_list resnet50 resnet34 resnet18 --skip_list 1 2 4 --policy_also_backbone --policy_input_offset 3 --accuracy_weight 0.9 --efficency_weight 0.1 --uniform_loss_weight 3.0 --use_gflops_loss --exp_decay --init_tau 0.6309 --freeze_policy --base_pretrained_from ucf_res_3m124_a.9e.1_ed5_ft10ds_lr.001_gu3/models/ckpt.best.pth.tar --exp_header ucf_res_3m124_a.9e.1_ed5_ft10ds_lr.001_gu3_lr.0005 --gpus 0 1 2 3 4 5 --with_test
