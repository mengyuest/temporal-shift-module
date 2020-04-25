#TODO on satori or WSC, j=36 for 72

# TODO(phase 0) Base Models
# exp-0 efficientnet-b3, t=8, size 224, batchsize 72, train 10 epochs
python main.py fcvid RGB --arch efficientnet-b3 --num_segments 8 --lr 0.02 --epochs 10 --batch-size 72 -j 36 --npb --rescale_to 224 --folder_suffix _self --exp_header fcvid8_effb3_e10_224_0 --gpus 0 1 2 3 --with_test

# exp-1 efficientnet-b1, t=8, size 168, batchsize 72, train 10 epochs
python main.py fcvid RGB --arch efficientnet-b1 --num_segments 8 --lr 0.02 --epochs 10 --batch-size 72 -j 36 --npb --rescale_to 168 --folder_suffix _self --exp_header fcvid8_effb1_e10_168_0 --gpus 0 1 2 3 --with_test

# exp-2 efficientnet-b0, t=8, size 112, batchsize 72, train 10 epochs
python main.py fcvid RGB --arch efficientnet-b0 --num_segments 8 --lr 0.02 --epochs 10 --batch-size 72 -j 36 --npb --rescale_to 112 --folder_suffix _self --exp_header fcvid8_effb0_e10_112_0 --gpus 0 1 2 3 --with_test

# exp-3 efficientnet-b3, t=8, size 224, batchsize 48, train 50 epochs
python main.py fcvid RGB --arch efficientnet-b3 --num_segments 8 --lr 0.02 --epochs 50 --lr_steps 20 40 --batch-size 48 -j 36 --npb --rescale_to 224 --folder_suffix _self --exp_header fcvid8_effb3_t8_s224_e50_b48 --gpus 0 1 2 3 --with_test --partial_fcvid_eval

# exp-4 lstm, efficientnet-b3, t=8, size 224, batchsize 48, train 50 epochs
python main.py fcvid RGB --arch efficientnet-b3 --num_segments 8 --lr 0.02 --epochs 50 --lr_steps 20 40 --batch-size 48 -j 36 --npb --folder_suffix _self --ada_reso_skip  --policy_backbone efficientnet-b3 --offline_lstm_all --exp_header fcvid8_effb3_t8_s224_e50_b48_lstm_all --gpus 0 1 2 3 --with_test --partial_fcvid_eval

# TODO(phase 1) Baselines and Joint
# exp-5 joint
python main.py fcvid RGB --arch efficientnet-b3 --num_segments 8 --lr 0.001 --epochs 50 --lr_steps 50 100 --batch-size 48 -j 12 --npb --folder_suffix _self --ada_reso_skip --policy_backbone mobilenet_v2 --reso_list 224 168 112 84 --backbone_list efficientnet-b3 efficientnet-b1 efficientnet-b0 --skip_list 1 2 4 --policy_also_backbone --policy_input_offset 3 --model_paths fcvid8_effb3_e10_224_0/m.pth.tar fcvid8_effb1_e10_168_0/m.pth.tar fcvid8_effb0_e10_112_0/m.pth.tar --accuracy_weight 0.9 --efficency_weight 0.1 --uniform_loss_weight 3.0  --use_gflops_loss --exp_decay --init_tau 5 --exp_header fcvid8_eff_3m124_a.9e.1_ed5_ft10ds_lr.001_gu3 --gpus 0 1 2 3 --with_test --partial_fcvid_eval

# exp-6 random-policy
python main.py fcvid RGB --arch efficientnet-b3 --num_segments 8 --lr 0.001 --epochs 50 --lr_steps 50 100 --batch-size 48 -j 12 --npb --folder_suffix _self --ada_reso_skip --policy_backbone mobilenet_v2 --reso_list 224 168 112 84 --backbone_list efficientnet-b3 efficientnet-b1 efficientnet-b0 --skip_list 1 2 4 --policy_also_backbone --policy_input_offset 3 --model_paths fcvid8_effb3_e10_224_0/m.pth.tar fcvid8_effb1_e10_168_0/m.pth.tar fcvid8_effb0_e10_112_0/m.pth.tar --use_gflops_loss  --exp_decay --init_tau 5 --random_policy --exp_header fcvid8_eff_3m124_a.9e.1_ed5_ft10ds_lr.001_rand --gpus 0 1 2 3 --with_test --partial_fcvid_eval

# exp-7 all-policy
python main.py fcvid RGB --arch efficientnet-b3 --num_segments 8 --lr 0.001 --epochs 50 --lr_steps 50 100 --batch-size 48 -j 12 --npb --folder_suffix _self --ada_reso_skip --policy_backbone mobilenet_v2 --reso_list 224 168 112 84 --backbone_list efficientnet-b3 efficientnet-b1 efficientnet-b0 --skip_list 1 2 4 --policy_also_backbone --policy_input_offset 3 --model_paths fcvid8_effb3_e10_224_0/m.pth.tar fcvid8_effb1_e10_168_0/m.pth.tar fcvid8_effb0_e10_112_0/m.pth.tar --use_gflops_loss --exp_decay --init_tau 5 --all_policy --exp_header fcvid8_eff_3m124_a.9e.1_ed5_ft10ds_lr.001_all --gpus 0 1 2 3 --with_test --partial_fcvid_eval

# exp-8 scsampler (use 50 epochs TODO k=5! because we only have 8 frames here)
python main.py fcvid RGB --arch efficientnet-b3 --num_segments 8 --lr 0.02 --epochs 50 --lr_steps 20 40 --batch-size 48 -j 36 --npb --folder_suffix _self  --ada_reso_skip --policy_backbone mobilenet_v2 --backbone_list efficientnet-b3 --reso_list 224 84 --policy_input_offset 1 --model_paths g0225-002939_fcvid8_effb3_t8_s224_e50_b48/imta_models/ckpt.best.pth.tar --frame_independent --real_scsampler --consensus_type scsampler --freeze_backbone --exp_header fcvid8_effb3_scs_ac_fd_k5_b48 --gpus 0 1 2 3 --with_test --partial_fcvid_eval --top_k 5

# TODO(phase 2) Finetuning (remember smaller lr, --init_tau, and --freeze_policy, and --base_pretrained_from, and remove model_paths) (PENDING)
# exp-9 finetune
python main.py fcvid RGB --arch efficientnet-b3 --num_segments 8 --lr 0.0005 --epochs 50 --lr_steps 50 100 --batch-size 48 -j 12 --npb --folder_suffix _self --ada_reso_skip --policy_backbone mobilenet_v2 --reso_list 224 168 112 84 --backbone_list efficientnet-b3 efficientnet-b1 efficientnet-b0 --skip_list 1 2 4 --policy_also_backbone --policy_input_offset 3 --accuracy_weight 0.9 --efficency_weight 0.1 --uniform_loss_weight 3.0 --use_gflops_loss --exp_decay --init_tau TTTTT --freeze_policy --base_pretrained_from g0225-223616_fcvid8_eff_3m124_a.9e.1_ed5_ft10ds_lr.001_gu3/imta_models/ckpt.best.pth.tar --exp_header fcvid8_eff_3m124_a.9e.1_ed5_ft10ds_lr.001_gu3_lr.0005 --gpus 0 1 2 3 --with_test --partial_fcvid_eval
