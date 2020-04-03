#TODO-1 (224->168->112)
#TODO 10 epo
#TODO(done)
python main.py actnet RGB --arch dmynet50 --num_segments 16 --lr 0.02 --epochs 10 --lr_steps 20 40 --npb --folder_suffix _self --rescale_to 224 --dmy --num_filters_list 64 48 32 --default_signal 0 --batch-size 48 -j 72 --gpus 0 1 2 3 --exp_header act_dres3_chain0_224_0_e10_b48_lcs --last_conv_same

#TODO 50 epo
#TODO(done)
python main.py actnet RGB --arch dmynet50 --num_segments 16 --lr 0.02 --epochs 50 --lr_steps 20 40 --npb --folder_suffix _self --rescale_to 224 --dmy --num_filters_list 64 48 32 --default_signal 0 --batch-size 48 -j 72 --gpus 0 1 2 3 --exp_header act_dres3_chain0_224_0_e50_b48_lcs --last_conv_same


#TODO-2 distillation (distill_policy, by_pred, two-step training, scratch/10/50)
python main.py actnet RGB --arch dmynet50 --num_segments 16 --lr 0.001 --epochs 50 --lr_steps 50 100 --batch-size 48 -j 72 --npb --folder_suffix _self --ada_reso_skip --policy_backbone mobilenet_v2 --reso_list 224 168 112 --backbone_list dmynet50 --skip_list 1 2 4 --policy_input_offset 2 --dmy --num_filters_list 64 48 32  --accuracy_weight 1.0 --efficency_weight 0.0 --uniform_loss_weight 0.0  --use_gflops_loss --exp_decay --init_tau 5 --exp_header act_ada_a.25e.0d.75_ed5_scratch_lr.001_lcs_dst_debug --gpus 0 1 2 3  --distill_policy --distillation_weight 0.25 --last_conv_same


#TODO-3 distillation (distill_policy, not shared, by_pred, scratch/10/50)
#TODO scratch, distill policy
python main.py actnet RGB --arch dmynet50 --num_segments 16 --lr 0.001 --epochs 50 --lr_steps 50 100 --batch-size 48 -j 72 --npb --folder_suffix _self --ada_reso_skip --policy_backbone mobilenet_v2 --reso_list 224 168 112 --backbone_list dmynet50 --skip_list 1 2 4 --policy_input_offset 2 --dmy --num_filters_list 64 48 32  --accuracy_weight 1.0 --efficency_weight 0.0 --uniform_loss_weight 0.0  --use_gflops_loss --exp_decay --init_tau 5 --exp_header act_ada_a.75e.0d.25_ed5_scratch_lr.001_dst_debug --gpus 0 1 2 3  --distill_policy --distillation_weight 0.25

#TODO load from prev models, distill policy
python main.py actnet RGB --arch dmynet50 --num_segments 16 --lr 0.001 --epochs 50 --lr_steps 50 100 --batch-size 48 -j 72 --npb --folder_suffix _self --ada_reso_skip --policy_backbone mobilenet_v2 --reso_list 224 168 112 --backbone_list dmynet50 --skip_list 1 2 4 --policy_input_offset 2 --dmy --num_filters_list 64 48 32  --accuracy_weight 1.0 --efficency_weight 0.0 --uniform_loss_weight 0.0  --use_gflops_loss --exp_decay --init_tau 5 --exp_header act_ada_a.75e.0d.25_ed5_cft10_lr.001_dst --gpus 0 1 2 3  --distill_policy --distillation_weight 0.25 --model_paths g0329-073458_act_dres3_chain0_224_0_e10_b48/models/ckpt.best.pth.tar g0329-082233_act_dres3_chain1_168_1_e10_b48_pre224/models/ckpt.best.pth.tar g0329-085151_act_dres3_chain2_112_2_e10_b48_pre168/models/ckpt.best.pth.tar