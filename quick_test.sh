# TSM(small)
python main_gate.py teenysth RGB --arch resnet18 --num_segments 4 --lr 0.002 --epochs 2 --batch-size 12 -j 12 --npb --shift --exp_header teenysth_tsm4_res18_b12_debug --gpus 0

# TSM(full)
python main_gate.py teenysth RGB --arch resnet18 --num_segments 8 --lr 0.002 --epochs 2 --batch-size 48 -j 72 --npb --shift --exp_header teenysth_tsm8_res18_b12_debug --gpus 0 1 2 3

# Adaptive(full)
python main_gate.py tinysth RGB --arch batenet18 --num_segments 8 --epochs 50 --lr 0.02 --lr_steps 20 40 --npb --ada_reso_skip --init_tau 0.67 --gsmx --gate_history --gate_hidden_dim 512 --gbn --grelu --gate_gflops_loss_weight 0.125 --gflops_loss_type upb --batch-size 48 -j 72 --gpus 0 1 2 3 --exp_header tinysth_8_bate18_h512_gsmx_g.125_upbg_scratch

# Distributed
python main_gate.py tinysth RGB --arch batenet18 --num_segments 8 --epochs 50 --lr 0.02 --lr_steps 20 40 --npb --ada_reso_skip --init_tau 0.67 --gsmx --gate_history --gate_hidden_dim 512 --gbn --grelu --gate_gflops_loss_weight 0.125 --gflops_loss_type upb --batch-size 48 -j 72 --gpus 0 1 2 3 --exp_header tinysth_8_bate18_h512_gsmx_g.125_upbg_scratch_dual --ws 2 --rank 0 --dist-url tcp://node0051:10601


python main_gate.py tinysth RGB --arch batenet18 --num_segments 8 --epochs 50 --lr 0.02 --lr_steps 20 40 --npb --ada_reso_skip --init_tau 0.67 --gsmx --gate_history --gate_hidden_dim 512 --gbn --grelu --gate_gflops_loss_weight 0.125 --gflops_loss_type upb --batch-size 48 -j 72 --gpus 0 1 2 3 --exp_header tinysth_8_bate18_h512_gsmx_g.125_upbg_scratch_dual --ws 2 --rank 0 --dist-url tcp://node0051:10601