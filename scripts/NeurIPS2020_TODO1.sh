#TODO Baseline models
#TODO (single node, TSN/TSM)
python main_gate.py somethingv2 RGB --arch  resnet50 --num_segments 8 --lr 0.01 --lr_steps 20 40 --epochs 50 --wd 5e-4 --npb --batch-size 64 -j 72 --exp_header sthv2_seg8_res50_b64_lr.01step --gpus 0 1 2 3

python main_gate.py somethingv2 RGB --arch  resnet50 --num_segments 8 --lr 0.01 --lr_steps 20 40 --epochs 50 --wd 5e-4 --npb --shift --batch-size 64 -j 72 --exp_header sthv2_tsm8_res50_b64_lr.01step --gpus 0 1 2 3



#TODO (distributed, dist-url mode)
python main_gate.py somethingv2 RGB --arch  resnet50 --num_segments 8 --lr 0.01 --lr_steps 20 40 --epochs 50 --wd 5e-4 --npb --batch-size 64 -j 72 --exp_header sthv2_seg8_res50_b64_lr.01step_N8 --gpus 0 1 2 3 --ws 8 --rank 0 --dist-url tcp://node0002:10601

python main_gate.py somethingv2 RGB --arch  resnet50 --num_segments 8 --lr 0.01 --lr_steps 20 40 --epochs 50 --wd 5e-4 --npb --shift --batch-size 64 -j 72 --exp_header sthv2_tsm8_res50_b64_lr.01step_N8 --gpus 0 1 2 3 --ws 8 --rank 0 --dist-url tcp://node0002:10601


#TODO Adaptive models
#TODO (single node, adaptive tsn/tsm)
python main_gate.py somethingv2 RGB --arch batenet50 --num_segments 8 --lr 0.01 --lr_steps 20 40 --epochs 50 --wd 5e-4 --npb --ada_reso_skip --init_tau 0.67 --gsmx --gate_history --hidden_quota 196608 --shared_policy_net --gbn --grelu --gate_gflops_loss_weight 0.125 --gflops_loss_type upb --batch-size 64 -j 72 --gpus 0 1 2 3 --exp_header sthv2_8_bate50_q196ksha_gsmx_g.125_upbg_b64_lr.01step

python main_gate.py somethingv2 RGB --arch batenet50 --num_segments 8 --lr 0.01 --lr_steps 20 40 --epochs 50 --wd 5e-4 --npb --ada_reso_skip --init_tau 0.67 --gsmx --gate_history --hidden_quota 196608 --shared_policy_net --gbn --grelu --gate_gflops_loss_weight 0.125 --gflops_loss_type upb --shift --batch-size 64 -j 72 --gpus 0 1 2 3 --exp_header sthv2_8_bate50_q196ksha_gsmx_g.125_tsm_upbg_b64_lr.01step



#TODO (distributed, dist-url mode, adaptive tsn/tsm)
python main_gate.py somethingv2 RGB --arch batenet50 --num_segments 8 --lr 0.01 --lr_steps 20 40 --epochs 50 --wd 5e-4 --npb --ada_reso_skip --init_tau 0.67 --gsmx --gate_history --hidden_quota 196608 --shared_policy_net --gbn --grelu --gate_gflops_loss_weight 0.125 --gflops_loss_type upb --batch-size 64 -j 72 --gpus 0 1 2 3 --exp_header sthv2_8_bate50_q196ksha_gsmx_g.125_upbg_b64_lr.01step_N8 --ws 8 --rank 0 --dist-url tcp://node0002:10601


python main_gate.py somethingv2 RGB --arch batenet50 --num_segments 8 --lr 0.01 --lr_steps 20 40 --epochs 50 --wd 5e-4 --npb --ada_reso_skip --init_tau 0.67 --gsmx --gate_history --hidden_quota 196608 --shared_policy_net --gbn --grelu --gate_gflops_loss_weight 0.125 --gflops_loss_type upb --shift --batch-size 64 -j 72 --gpus 0 1 2 3 --exp_header sthv2_8_bate50_q196ksha_gsmx_g.125_tsm_upbg_b64_lr.01step_N8 --ws 8 --rank 0 --dist-url tcp://node0002:10601