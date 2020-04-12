python main.py actnet RGB --arch msdnet --num_segments 16 --lr 0.02 --epochs 2 --lr_steps 20 40 --batch-size 4 -j 4 --npb --folder_suffix _self --ada_reso_skip --policy_backbone mobilenet_v2 --reso_list 224 168 112 --backbone_list msdnet --skip_list 1 2 4 --policy_input_offset 2 --accuracy_weight 1.0 --efficency_weight 0.0 --uniform_loss_weight 0.0 --use_gflops_loss --exp_decay --init_tau 5 --exp_header act_ada_a1e0_ed5_scratch_lr.02_msd_debug --gpus 0  --msd --msd_indices_list 3 1 0


#TODO STAGE 1:   ImageNet->ActivityNet pretraining
#TODO: 10/50/
#TODO: 0. all_policy, 1. random_policy, 2. last_exit
python main.py actnet RGB --arch msdnet --num_segments 16 --lr 0.02 --epochs 10 --lr_steps 20 40 --batch-size 48 -j 72 --npb --folder_suffix _self --ada_reso_skip --policy_backbone mobilenet_v2 --reso_list 224 112 --backbone_list msdnet --skip_list 1 2 4 --policy_input_offset 1 --accuracy_weight 1.0 --efficency_weight 0.0 --uniform_loss_weight 0.0 --use_gflops_loss --exp_decay --init_tau 5 --exp_header act_msd_lr.02_unoreso_310_all_e10 --gpus 0 1 2 3  --msd --msd_indices_list 3 1 0 --uno_reso --all_policy

python main.py actnet RGB --arch msdnet --num_segments 16 --lr 0.02 --epochs 50 --lr_steps 20 40 --batch-size 48 -j 72 --npb --folder_suffix _self --ada_reso_skip --policy_backbone mobilenet_v2 --reso_list 224 112 --backbone_list msdnet --skip_list 1 2 4 --policy_input_offset 1 --accuracy_weight 1.0 --efficency_weight 0.0 --uniform_loss_weight 0.0 --use_gflops_loss --exp_decay --init_tau 5 --exp_header act_msd_lr.02_unoreso_310_all_e50 --gpus 0 1 2 3  --msd --msd_indices_list 3 1 0 --uno_reso --all_policy


python main.py actnet RGB --arch msdnet --num_segments 16 --lr 0.02 --epochs 10 --lr_steps 20 40 --batch-size 48 -j 72 --npb --folder_suffix _self --ada_reso_skip --policy_backbone mobilenet_v2 --reso_list 224 112 --backbone_list msdnet --skip_list 1 2 4 --policy_input_offset 1 --accuracy_weight 1.0 --efficency_weight 0.0 --uniform_loss_weight 0.0 --use_gflops_loss --exp_decay --init_tau 5 --exp_header act_msd_lr.02_unoreso_310_rand_e10 --gpus 0 1 2 3  --msd --msd_indices_list 3 1 0 --uno_reso --random_policy

python main.py actnet RGB --arch msdnet --num_segments 16 --lr 0.02 --epochs 50 --lr_steps 20 40 --batch-size 48 -j 72 --npb --folder_suffix _self --ada_reso_skip --policy_backbone mobilenet_v2 --reso_list 224 112 --backbone_list msdnet --skip_list 1 2 4 --policy_input_offset 1 --accuracy_weight 1.0 --efficency_weight 0.0 --uniform_loss_weight 0.0 --use_gflops_loss --exp_decay --init_tau 5 --exp_header act_msd_lr.02_unoreso_310_rand_e50 --gpus 0 1 2 3  --msd --msd_indices_list 3 1 0 --uno_reso --random_policy


python main.py actnet RGB --arch msdnet --num_segments 16 --lr 0.02 --epochs 10 --lr_steps 20 40 --batch-size 48 -j 72 --npb --folder_suffix _self --ada_reso_skip --policy_backbone mobilenet_v2 --reso_list 224 112 --backbone_list msdnet --skip_list 1 2 4 --policy_input_offset 1 --accuracy_weight 1.0 --efficency_weight 0.0 --uniform_loss_weight 0.0 --use_gflops_loss --exp_decay --init_tau 5 --exp_header act_msd_lr.02_unoreso_310_distill_e10 --gpus 0 1 2 3  --msd --msd_indices_list 3 1 0 --uno_reso --distill_policy

python main.py actnet RGB --arch msdnet --num_segments 16 --lr 0.02 --epochs 50 --lr_steps 20 40 --batch-size 48 -j 72 --npb --folder_suffix _self --ada_reso_skip --policy_backbone mobilenet_v2 --reso_list 224 112 --backbone_list msdnet --skip_list 1 2 4 --policy_input_offset 1 --accuracy_weight 1.0 --efficency_weight 0.0 --uniform_loss_weight 0.0 --use_gflops_loss --exp_decay --init_tau 5 --exp_header act_msd_lr.02_unoreso_310_distill_e50 --gpus 0 1 2 3  --msd --msd_indices_list 3 1 0 --uno_reso --distill_policy



#TODO STAGE 2: adaptive one






#TODO ETC: 100 epochs, for baselines)