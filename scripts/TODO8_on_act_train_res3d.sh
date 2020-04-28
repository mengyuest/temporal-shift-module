#TODO(yue) assuming num_segments (frames) = 40, seg_len=8 for adaptive learning

#TODO PHASE-I
#TODO 0. train basemodel res3d50 on 224*224 for 10 epochs
python main.py actnet RGB --arch res3d50 --num_segments 8 --lr 0.02 --epochs 10 --batch-size 72 -j 12 --npb --gpus 0 1 2 3 --exp_header act_t8_res3d50_sz224_b72_e10 --folder_suffix _self --cnn3d --seg_len 8 --rescale_to 224 --pretrain imagenet

#TODO 1. train basemodel res3d34 on 168*168 for 10 epochs
python main.py actnet RGB --arch res3d34 --num_segments 8 --lr 0.02 --epochs 10 --batch-size 72 -j 12 --npb --gpus 0 1 2 3 --exp_header act_t8_res3d34_sz168_b72_e10 --folder_suffix _self --cnn3d --seg_len 8 --rescale_to 168 --pretrain imagenet

#TODO 2. train basemodel res3d18 on 112*112 for 10 epochs
python main.py actnet RGB --arch res3d18 --num_segments 8 --lr 0.02 --epochs 10 --batch-size 72 -j 12 --npb --gpus 0 1 2 3 --exp_header act_t8_res3d18_sz112_b72_e10 --folder_suffix _self --cnn3d --seg_len 8 --rescale_to 112 --pretrain imagenet

#TODO 3. train basemodel res3d34 on 168*168 for 50 epochs
python main.py actnet RGB --arch res3d34 --num_segments 40 --lr 0.02 --epochs 50 --lr_steps 20 40 --batch-size 48 -j 12 --npb --gpus 0 1 2 3 --exp_header act_t40_res3d34_sz168_b48_e50 --folder_suffix _self --cnn3d --seg_len 8 --rescale_to 168 --pretrain imagenet
#python main.py MINISTH RGB --arch res3d50 --num_segments 40 --lr 0.02 --epochs 50 --lr_steps 20 40 --batch-size 48 -j 12 --npb --gpus 0 1 2 3 --exp_header ms2_t40_res3d50_sz224_b48_e50 --folder_suffix _self --cnn3d --seg_len 8 --rescale_to 224 --pretrain imagenet

#TODO 4. train basemodel res3d34, lstm on 168*168 for 50 epochs
python main.py actnet RGB --arch res3d34 --num_segments 40 --lr 0.02 --epochs 50 --lr_steps 20 40 --batch-size 48 -j 12 --npb --gpus 0 1 2 3 --exp_header ms2_t40_res3d34_sz168_b48_e50_lstm_all --folder_suffix _self --cnn3d --seg_len 8 --pretrain imagenet --ada_reso_skip --policy_backbone res3d34 --reso_list 168 --offline_lstm_all
#python main.py MINISTH RGB --arch res3d50 --num_segments 40 --lr 0.02 --epochs 50 --lr_steps 20 40 --batch-size 48 -j 12 --npb --gpus 0 1 2 3 --exp_header ms2_t40_res3d50_sz224_b48_e50_lstm_all --folder_suffix _self --cnn3d --seg_len 8 --pretrain imagenet --ada_reso_skip --policy_backbone res3d50 --reso_list 224 --offline_lstm_all


#TODO PHASE-II
#(res3d34+res3d18+mob3dv2)
#TODO 5. joint training adaptive policy
python main.py actnet RGB --arch res3d34 --num_segments 40 --lr 0.001 --epochs 50 --lr_steps 50 100 --batch-size 48 -j 12 --npb --gpus 0 1 2 3 --exp_header act_t40_2m124_a1e0_ed5_ft10ds_lr.001_gu3_debug --ada_reso_skip --policy_backbone mobilenet3dv2 --reso_list 168 112 84 --backbone_list res3d34 res3d18 --skip_list 1 2 4 --accuracy_weight 1.0 --efficency_weight 0.0 --folder_suffix _self --exp_decay --init_tau 5 --policy_also_backbone --policy_input_offset 2 --uniform_loss_weight 3.0 --use_gflops_loss --cnn3d --seg_len 8 --model_paths act_t8_res3d34_sz168_b72_e10/models/ckpt.best.pth.tar act_t8_res3d18_sz112_b72_e10/models/ckpt.best.pth.tar

#TODO 6. train rand policy
python main.py MINISTH RGB --arch res3d34 --num_segments 40 --lr 0.001 --epochs 50 --lr_steps 50 100 --batch-size 48 -j 12 --npb --gpus 0 1 2 3 --exp_header ms2_t40_2m124_a.9e.1_ed5_ft10ds_lr.001_gu3_rand_debug --ada_reso_skip --policy_backbone mobilenet3dv2 --reso_list 168 112 84 --backbone_list res3d34 res3d18 --skip_list 1 2 4 --accuracy_weight 0.9 --efficency_weight 0.1 --folder_suffix _self --exp_decay --init_tau 5 --policy_also_backbone --policy_input_offset 2 --uniform_loss_weight 3.0 --use_gflops_loss --cnn3d --seg_len 8 --random_policy --model_paths R34/models/ckpt.best.pth R18/models/ckpt.best.pth

#TODO 7. train all policy
python main.py MINISTH RGB --arch res3d34 --num_segments 40 --lr 0.001 --epochs 50 --lr_steps 50 100 --batch-size 48 -j 12 --npb --gpus 0 1 2 3 --exp_header ms2_t40_2m124_a.9e.1_ed5_ft10ds_lr.001_gu3_all_debug --ada_reso_skip --policy_backbone mobilenet3dv2 --reso_list 168 112 84 --backbone_list res3d34 res3d18 --skip_list 1 2 4 --accuracy_weight 0.9 --efficency_weight 0.1 --folder_suffix _self --exp_decay --init_tau 5 --policy_also_backbone --policy_input_offset 2 --uniform_loss_weight 3.0 --use_gflops_loss --cnn3d --seg_len 8 --all_policy --model_paths R34/models/ckpt.best.pth R18/models/ckpt.best.pth

#TODO 8. scsampler
python main.py MINISTH RGB --arch res3d34 --num_segments 40 --lr 0.02 --epochs 50 --lr_steps 20 40 --batch-size 48 -j 12 --npb --folder_suffix _self  --ada_reso_skip --policy_backbone mobilenet3dv2 --backbone_list res3d34 --reso_list 168 84 --policy_input_offset 1 --model_paths R34/models/ckpt.best.pth.tar --frame_independent --real_scsampler --consensus_type scsampler --freeze_backbone --cnn3d --seg_len 8 --with_test --exp_header ms2_t40c5_res3d34_scs_ac_fd_k3_b48 --gpus 0 1 2 3  --top_k 3

#TODO PHASE-III
#TODO 9. finetuning adaptive policy (remove model_paths, add base_pretrain_from, change lr=0.0005, change init_tau corr. to the best model from pretrained ones, e.g. 0.6600)
python main.py MINISTH RGB --arch res3d34 --num_segments 40 --lr 0.0005 --epochs 50 --lr_steps 50 100 --batch-size 48 -j 12 --npb --gpus 0 1 2 3 --exp_header ms2_t40_2m124_a.9e.1_ed5_ft10ds_lr.001_gu3_ft_lr.0005_debug --ada_reso_skip --policy_backbone mobilenet3dv2 --reso_list 168 112 84 --backbone_list res3d34 res3d18 --skip_list 1 2 4 --accuracy_weight 0.9 --efficency_weight 0.1 --folder_suffix _self --exp_decay --init_tau XXXXXX --policy_also_backbone --policy_input_offset 2 --uniform_loss_weight 3.0 --use_gflops_loss --cnn3d --seg_len 8 --freeze_policy --base_pretrained_from JOINT/models/ckpt.best.pth

