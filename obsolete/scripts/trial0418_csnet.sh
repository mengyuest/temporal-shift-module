
#TODO baseline csnet
python main_base_bak.py actnet RGB --arch mernet50 --num_segments 16 --lr 0.02 --epochs 2 --lr_steps 20 40 --batch-size 4 -j 4 --npb --folder_suffix _self --ada_reso_skip --policy_backbone mobilenet_v2 --reso_list 224 168 112 --backbone_list mernet50 --skip_list 1 2 4 --policy_input_offset 2 --accuracy_weight 1.0 --efficency_weight 0.0 --uniform_loss_weight 0.0 --use_gflops_loss --exp_decay --init_tau 5 --exp_header act_ada_a1e0_ed5_scratch_lr.02_mer_debug --gpus 0  --mer --mer_indices_list 3 2 1

python main_base_bak.py actnet RGB --arch ir_csn_50 --num_segments 16 --lr 0.02 --epochs 50 --lr_steps 20 40 --batch-size 5 -j 36 --npb --rescale_to 224 --folder_suffix _self --exp_header actnet_ircsn50_t16_s224_e50_b48_debug --gpus 0 --cnn3d --csn