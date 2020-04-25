python main.py actnet RGB --arch mernet50 --num_segments 16 --lr 0.02 --epochs 2 --lr_steps 20 40 --batch-size 4 -j 4 --npb --folder_suffix _self --ada_reso_skip --policy_backbone mobilenet_v2 --reso_list 224 168 112 --backbone_list mernet50 --skip_list 1 2 4 --policy_input_offset 2 --accuracy_weight 1.0 --efficency_weight 0.0 --uniform_loss_weight 0.0 --use_gflops_loss --exp_decay --init_tau 5 --exp_header act_ada_a1e0_ed5_scratch_lr.02_mer_debug --gpus 0  --mer --mer_indices_list 3 2 1


#TODO STAGE 1-1:   ImageNet->ActivityNet pretraining
#TODO: 10/50/
#TODO: 0. all_policy, 1. random_policy, 2. last_exit
#python main.py actnet RGB --arch mernet50 --num_segments 16 --lr 0.02 --epochs 10 --lr_steps 20 40 --batch-size 48 -j 16 --npb --folder_suffix _self --ada_reso_skip --policy_backbone mobilenet_v2 --reso_list 224 112 --backbone_list mernet50 --skip_list 1 2 4 --policy_input_offset 1 --accuracy_weight 1.0 --efficency_weight 0.0 --uniform_loss_weight 0.0 --use_gflops_loss --exp_decay --init_tau 5 --exp_header act_mer_lr.02_unoreso_310_all_e10 --gpus 0 1 2 3 --mer --mer_indices_list 3 2 1 --uno_reso --all_policy

python main.py actnet RGB --arch mernet50 --num_segments 16 --lr 0.02 --epochs 50 --lr_steps 20 40 --batch-size 48 -j 72 --npb --folder_suffix _self --ada_reso_skip --policy_backbone mobilenet_v2 --reso_list 224 112 --backbone_list mernet50 --skip_list 1 2 4 --policy_input_offset 1 --accuracy_weight 1.0 --efficency_weight 0.0 --uniform_loss_weight 0.0 --use_gflops_loss --exp_decay --init_tau 5 --exp_header act_mer_lr.02_unoreso_310_reall_e50 --gpus 0 1 2 3 --mer --mer_indices_list 3 2 1 --uno_reso --real_all_policy


#python main.py actnet RGB --arch msdnet --num_segments 16 --lr 0.02 --epochs 10 --lr_steps 20 40 --batch-size 48 -j 72 --npb --folder_suffix _self --ada_reso_skip --policy_backbone mobilenet_v2 --reso_list 224 112 --backbone_list msdnet --skip_list 1 2 4 --policy_input_offset 1 --accuracy_weight 1.0 --efficency_weight 0.0 --uniform_loss_weight 0.0 --use_gflops_loss --exp_decay --init_tau 5 --exp_header act_msd_lr.02_unoreso_310_rand_e10 --gpus 0 1 2 3  --msd --msd_indices_list 3 1 0 --uno_reso --random_policy

python main.py actnet RGB --arch mernet50 --num_segments 16 --lr 0.02 --epochs 50 --lr_steps 20 40 --batch-size 48 -j 16 --npb --folder_suffix _self --ada_reso_skip --policy_backbone mobilenet_v2 --reso_list 224 112 --backbone_list mernet50 --skip_list 1 2 4 --policy_input_offset 1 --accuracy_weight 1.0 --efficency_weight 0.0 --uniform_loss_weight 0.0 --use_gflops_loss --exp_decay --init_tau 5 --exp_header act_mer_lr.02_unoreso_310_rand_e50 --gpus 0 1 2 3 --mer --mer_indices_list 3 2 1 --uno_reso --random_policy


#python main.py actnet RGB --arch msdnet --num_segments 16 --lr 0.02 --epochs 10 --lr_steps 20 40 --batch-size 48 -j 72 --npb --folder_suffix _self --ada_reso_skip --policy_backbone mobilenet_v2 --reso_list 224 112 --backbone_list msdnet --skip_list 1 2 4 --policy_input_offset 1 --accuracy_weight 1.0 --efficency_weight 0.0 --uniform_loss_weight 0.0 --use_gflops_loss --exp_decay --init_tau 5 --exp_header act_msd_lr.02_unoreso_310_distill_e10 --gpus 0 1 2 3  --msd --msd_indices_list 3 1 0 --uno_reso --distill_policy

python main.py actnet RGB --arch mernet50 --num_segments 16 --lr 0.02 --epochs 50 --lr_steps 20 40 --batch-size 48 -j 16 --npb --folder_suffix _self --ada_reso_skip --policy_backbone mobilenet_v2 --reso_list 224 112 --backbone_list mernet50 --skip_list 1 2 4 --policy_input_offset 1 --accuracy_weight 1.0 --efficency_weight 0.0 --uniform_loss_weight 0.0 --use_gflops_loss --exp_decay --init_tau 5 --exp_header act_mer_lr.02_unoreso_310_distill_e50 --gpus 0 1 2 3 --mer --mer_indices_list 3 2 1 --uno_reso --distill_policy



#TODO STAGE 1-2: Distillation learn (random/last,       acc/distill 0.9/0.1, 0.7/0.3, 0.5/0.5)
#TODO real all
python main.py actnet RGB --arch mernet50 --num_segments 16 --lr 0.02 --epochs 50 --lr_steps 20 40 --batch-size 48 -j 72 --npb --folder_suffix _self --ada_reso_skip --policy_backbone mobilenet_v2 --reso_list 224 112 --backbone_list mernet50 --skip_list 1 2 4 --policy_input_offset 1 --accuracy_weight 1.0 --efficency_weight 0.0 --uniform_loss_weight 0.0 --use_gflops_loss --exp_decay --init_tau 5 --exp_header act_mer_lr.02_unoreso_310_reall_e50_a.9d.1 --gpus 0 1 2 3 --mer --mer_indices_list 3 2 1 --uno_reso --real_all_policy --distillation_weight 0.1

python main.py actnet RGB --arch mernet50 --num_segments 16 --lr 0.02 --epochs 50 --lr_steps 20 40 --batch-size 48 -j 72 --npb --folder_suffix _self --ada_reso_skip --policy_backbone mobilenet_v2 --reso_list 224 112 --backbone_list mernet50 --skip_list 1 2 4 --policy_input_offset 1 --accuracy_weight 1.0 --efficency_weight 0.0 --uniform_loss_weight 0.0 --use_gflops_loss --exp_decay --init_tau 5 --exp_header act_mer_lr.02_unoreso_310_reall_e50_a.7d.3 --gpus 0 1 2 3 --mer --mer_indices_list 3 2 1 --uno_reso --real_all_policy --distillation_weight 0.3

python main.py actnet RGB --arch mernet50 --num_segments 16 --lr 0.02 --epochs 50 --lr_steps 20 40 --batch-size 48 -j 72 --npb --folder_suffix _self --ada_reso_skip --policy_backbone mobilenet_v2 --reso_list 224 112 --backbone_list mernet50 --skip_list 1 2 4 --policy_input_offset 1 --accuracy_weight 1.0 --efficency_weight 0.0 --uniform_loss_weight 0.0 --use_gflops_loss --exp_decay --init_tau 5 --exp_header act_mer_lr.02_unoreso_310_reall_e50_a.5d.5 --gpus 0 1 2 3 --mer --mer_indices_list 3 2 1 --uno_reso --real_all_policy --distillation_weight 0.5

#TODO random
python main.py actnet RGB --arch mernet50 --num_segments 16 --lr 0.02 --epochs 50 --lr_steps 20 40 --batch-size 48 -j 72 --npb --folder_suffix _self --ada_reso_skip --policy_backbone mobilenet_v2 --reso_list 224 112 --backbone_list mernet50 --skip_list 1 2 4 --policy_input_offset 1 --accuracy_weight 1.0 --efficency_weight 0.0 --uniform_loss_weight 0.0 --use_gflops_loss --exp_decay --init_tau 5 --exp_header act_mer_lr.02_unoreso_310_rand_e50_a.9d.1 --gpus 0 1 2 3 --mer --mer_indices_list 3 2 1 --uno_reso --random_policy --distillation_weight 0.1

python main.py actnet RGB --arch mernet50 --num_segments 16 --lr 0.02 --epochs 50 --lr_steps 20 40 --batch-size 48 -j 72 --npb --folder_suffix _self --ada_reso_skip --policy_backbone mobilenet_v2 --reso_list 224 112 --backbone_list mernet50 --skip_list 1 2 4 --policy_input_offset 1 --accuracy_weight 1.0 --efficency_weight 0.0 --uniform_loss_weight 0.0 --use_gflops_loss --exp_decay --init_tau 5 --exp_header act_mer_lr.02_unoreso_310_rand_e50_a.7d.3 --gpus 0 1 2 3 --mer --mer_indices_list 3 2 1 --uno_reso --random_policy --distillation_weight 0.3

python main.py actnet RGB --arch mernet50 --num_segments 16 --lr 0.02 --epochs 50 --lr_steps 20 40 --batch-size 48 -j 72 --npb --folder_suffix _self --ada_reso_skip --policy_backbone mobilenet_v2 --reso_list 224 112 --backbone_list mernet50 --skip_list 1 2 4 --policy_input_offset 1 --accuracy_weight 1.0 --efficency_weight 0.0 --uniform_loss_weight 0.0 --use_gflops_loss --exp_decay --init_tau 5 --exp_header act_mer_lr.02_unoreso_310_rand_e50_a.5d.5 --gpus 0 1 2 3 --mer --mer_indices_list 3 2 1 --uno_reso --random_policy --distillation_weight 0.5


#TODO random (noskip)
python main.py actnet RGB --arch mernet50 --num_segments 16 --lr 0.02 --epochs 50 --lr_steps 20 40 --batch-size 48 -j 72 --npb --folder_suffix _self --ada_reso_skip --policy_backbone mobilenet_v2 --reso_list 224 112 --backbone_list mernet50  --policy_input_offset 1 --accuracy_weight 1.0 --efficency_weight 0.0 --uniform_loss_weight 0.0 --use_gflops_loss --exp_decay --init_tau 5 --exp_header act_mer_lr.02_unoreso_310_randns_e50_a.9d.1 --gpus 0 1 2 3 --mer --mer_indices_list 3 2 1 --uno_reso --random_policy --distillation_weight 0.1

python main.py actnet RGB --arch mernet50 --num_segments 16 --lr 0.02 --epochs 50 --lr_steps 20 40 --batch-size 48 -j 72 --npb --folder_suffix _self --ada_reso_skip --policy_backbone mobilenet_v2 --reso_list 224 112 --backbone_list mernet50  --policy_input_offset 1 --accuracy_weight 1.0 --efficency_weight 0.0 --uniform_loss_weight 0.0 --use_gflops_loss --exp_decay --init_tau 5 --exp_header act_mer_lr.02_unoreso_310_randns_e50_a.7d.3 --gpus 0 1 2 3 --mer --mer_indices_list 3 2 1 --uno_reso --random_policy --distillation_weight 0.3

python main.py actnet RGB --arch mernet50 --num_segments 16 --lr 0.02 --epochs 50 --lr_steps 20 40 --batch-size 48 -j 72 --npb --folder_suffix _self --ada_reso_skip --policy_backbone mobilenet_v2 --reso_list 224 112 --backbone_list mernet50  --policy_input_offset 1 --accuracy_weight 1.0 --efficency_weight 0.0 --uniform_loss_weight 0.0 --use_gflops_loss --exp_decay --init_tau 5 --exp_header act_mer_lr.02_unoreso_310_randns_e50_a.5d.5 --gpus 0 1 2 3 --mer --mer_indices_list 3 2 1 --uno_reso --random_policy --distillation_weight 0.5

#TODO last one
python main.py actnet RGB --arch mernet50 --num_segments 16 --lr 0.02 --epochs 50 --lr_steps 20 40 --batch-size 48 -j 72 --npb --folder_suffix _self --ada_reso_skip --policy_backbone mobilenet_v2 --reso_list 224 112 --backbone_list mernet50 --skip_list 1 2 4 --policy_input_offset 1 --accuracy_weight 1.0 --efficency_weight 0.0 --uniform_loss_weight 0.0 --use_gflops_loss --exp_decay --init_tau 5 --exp_header act_mer_lr.02_unoreso_310_distill_e50_a.9d.1 --gpus 0 1 2 3 --mer --mer_indices_list 3 2 1 --uno_reso --distill_policy --distillation_weight 0.1

python main.py actnet RGB --arch mernet50 --num_segments 16 --lr 0.02 --epochs 50 --lr_steps 20 40 --batch-size 48 -j 72 --npb --folder_suffix _self --ada_reso_skip --policy_backbone mobilenet_v2 --reso_list 224 112 --backbone_list mernet50 --skip_list 1 2 4 --policy_input_offset 1 --accuracy_weight 1.0 --efficency_weight 0.0 --uniform_loss_weight 0.0 --use_gflops_loss --exp_decay --init_tau 5 --exp_header act_mer_lr.02_unoreso_310_distill_e50_a.7d.3 --gpus 0 1 2 3 --mer --mer_indices_list 3 2 1 --uno_reso --distill_policy --distillation_weight 0.3

python main.py actnet RGB --arch mernet50 --num_segments 16 --lr 0.02 --epochs 50 --lr_steps 20 40 --batch-size 48 -j 72 --npb --folder_suffix _self --ada_reso_skip --policy_backbone mobilenet_v2 --reso_list 224 112 --backbone_list mernet50 --skip_list 1 2 4 --policy_input_offset 1 --accuracy_weight 1.0 --efficency_weight 0.0 --uniform_loss_weight 0.0 --use_gflops_loss --exp_decay --init_tau 5 --exp_header act_mer_lr.02_unoreso_310_distill_e50_a.5d.5 --gpus 0 1 2 3 --mer --mer_indices_list 3 2 1 --uno_reso --distill_policy --distillation_weight 0.5




#TODO STAGE 1-3:   ImageNet->ActivityNet pretraining
# using (3,2,1), train 1 first for 10/50 epochs
#TODO epo10-epo10-epo10
#TODO epo50-epo50-eop50
#TODO epo50-epo10-epo10
#TODO epo10-epo10-epo50
python main.py actnet RGB --arch mernet50 --num_segments 16 --lr 0.02 --epochs 10 --lr_steps 20 40 --batch-size 48 -j 72 --npb --folder_suffix _self --ada_reso_skip --policy_backbone mobilenet_v2 --reso_list 224 112 --backbone_list mernet50 --skip_list 1 2 4 --policy_input_offset 1 --accuracy_weight 1.0 --efficency_weight 0.0 --uniform_loss_weight 0.0 --use_gflops_loss --exp_decay --init_tau 5 --exp_header act_mer_lr.02_unoreso_chain1_e10 --gpus 0 1 2 3 --mer --mer_indices_list 1 --uno_reso --distill_policy

# then fixed all up to layer 2, train another 10/50 epochs (change lr)
python main.py actnet RGB --arch mernet50 --num_segments 16 --lr 0.02 --epochs 10 --lr_steps 20 40 --batch-size 48 -j 72 --npb --folder_suffix _self --ada_reso_skip --policy_backbone mobilenet_v2 --reso_list 224 112 --backbone_list mernet50 --skip_list 1 2 4 --policy_input_offset 1 --accuracy_weight 1.0 --efficency_weight 0.0 --uniform_loss_weight 0.0 --use_gflops_loss --exp_decay --init_tau 5 --exp_header act_mer_lr.02_unoreso_chain2_e10_prechain1e10 --gpus 0 1 2 3 --mer --mer_indices_list 2 1 --uno_reso --distill_policy  --base_pretrained_from g0417-084428_act_mer_lr.02_unoreso_chain1_e10/imta_models/ckpt.best.pth.tar --no_weights_from_linear --frozen_layers 0 1 2 --incremental_load_new_fcs


python main.py actnet RGB --arch mernet50 --num_segments 16 --lr 0.02 --epochs 10 --lr_steps 20 40 --batch-size 48 -j 72 --npb --folder_suffix _self --ada_reso_skip --policy_backbone mobilenet_v2 --reso_list 224 112 --backbone_list mernet50 --skip_list 1 2 4 --policy_input_offset 1 --accuracy_weight 1.0 --efficency_weight 0.0 --uniform_loss_weight 0.0 --use_gflops_loss --exp_decay --init_tau 5 --exp_header act_mer_lr.02_unoreso_chain3_e10_prechain2e10 --gpus 0 1 2 3 --mer --mer_indices_list 3 2 1 --uno_reso --distill_policy  --base_pretrained_from xxx/imta_models/ckpt.best.pth.tar --no_weights_from_linear --frozen_layers 0 1 2 3

# then fixed all up to layer 3, train another 10/50 epochs

# then fixed all up to layer 4, train another 10/50 epochs

#TODO STAGE 2: adaptive one
#TODO (use all/random) 10epo, 50epo
#TODO (lr) .0002,                           .0005, .001
#TODO (combinations) a.9e.1u1,              a.9e.1u3, a.9e.1u5, a.9e.1u10, a.8e.2u1, a.8e.2u3, a.8e.2u5, a.8e.2u10,

#TODO best model, diff lrs
python main.py actnet RGB --arch mernet50 --num_segments 16 --lr 0.0002 --epochs 50 --lr_steps 50 100 --batch-size 48 -j 72 --npb --folder_suffix _self --ada_reso_skip --policy_backbone mobilenet_v2 --reso_list 224 112 --backbone_list mernet50 --skip_list 1 2 4 --policy_input_offset 1 --accuracy_weight 0.9 --efficency_weight 0.1 --uniform_loss_weight 3.0 --use_gflops_loss --exp_decay --init_tau 5 --exp_header act_mer_unoreso_32124_from_realldistill_adalr.0002 --gpus 0 1 2 3 --mer --mer_indices_list 3 2 1 --uno_reso --base_pretrained_from g0419-124023_act_mer_lr.02_unoreso_310_reall_e50_a.7d.3/imta_models/ckpt.best.pth.tar

python main.py actnet RGB --arch mernet50 --num_segments 16 --lr 0.0005 --epochs 50 --lr_steps 50 100 --batch-size 48 -j 72 --npb --folder_suffix _self --ada_reso_skip --policy_backbone mobilenet_v2 --reso_list 224 112 --backbone_list mernet50 --skip_list 1 2 4 --policy_input_offset 1 --accuracy_weight 0.9 --efficency_weight 0.1 --uniform_loss_weight 3.0 --use_gflops_loss --exp_decay --init_tau 5 --exp_header act_mer_unoreso_32124_from_realldistill_adalr.0005 --gpus 0 1 2 3 --mer --mer_indices_list 3 2 1 --uno_reso --base_pretrained_from g0419-124023_act_mer_lr.02_unoreso_310_reall_e50_a.7d.3/imta_models/ckpt.best.pth.tar

python main.py actnet RGB --arch mernet50 --num_segments 16 --lr 0.001 --epochs 50 --lr_steps 50 100 --batch-size 48 -j 72 --npb --folder_suffix _self --ada_reso_skip --policy_backbone mobilenet_v2 --reso_list 224 112 --backbone_list mernet50 --skip_list 1 2 4 --policy_input_offset 1 --accuracy_weight 0.9 --efficency_weight 0.1 --uniform_loss_weight 3.0 --use_gflops_loss --exp_decay --init_tau 5 --exp_header act_mer_unoreso_32124_from_realldistill_adalr.001 --gpus 0 1 2 3 --mer --mer_indices_list 3 2 1 --uno_reso --base_pretrained_from g0419-124023_act_mer_lr.02_unoreso_310_reall_e50_a.7d.3/imta_models/ckpt.best.pth.tar

#TODO ETC: 100 epochs, for baselines)