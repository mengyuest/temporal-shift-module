# TODO TSN dresnet50 (signal=0,1,2,3)
python main_base_bak.py actnet RGB --arch dmynet50 --num_segments 4 --lr 0.02 --epochs 2 --npb --folder_suffix _self  --exp_header debug --batch-size 6 -j 36 --gpus 0 1 --dmy --default_signal 0 --num_filters_list 64 32 16

python main_base_bak.py actnet RGB --arch dmynet50 --num_segments 4 --lr 0.02 --epochs 2 --npb --folder_suffix _self  --exp_header debug --batch-size 6 -j 36 --gpus 0 1 --rescale_to  --dmy --default_signal 1 --num_filters_list 64 32 16


# TODO TSN activity net
python main_base_bak.py actnet RGB --arch dmynet50 --num_segments 16 --lr 0.02 --epochs 50 --lr_steps 20 40  --npb --folder_suffix _self --rescale_to 224 --dmy --num_filters_list 64 32 16 8 --default_signal 0 --batch-size 48 -j 72 --gpus 0 1 2 3 --exp_header act_dres4_224_0_e50_b48

python main_base_bak.py actnet RGB --arch dmynet50 --num_segments 16 --lr 0.02 --epochs 50 --lr_steps 20 40  --npb --folder_suffix _self --rescale_to 168 --dmy --num_filters_list 64 32 16 8 --default_signal 1 --batch-size 48 -j 72 --gpus 0 1 2 3 --exp_header act_dres4_168_1_e50_b48

python main_base_bak.py actnet RGB --arch dmynet50 --num_segments 16 --lr 0.02 --epochs 50 --lr_steps 20 40  --npb --folder_suffix _self --rescale_to 112 --dmy --num_filters_list 64 32 16 8 --default_signal 2 --batch-size 48 -j 72 --gpus 0 1 2 3 --exp_header act_dres4_112_2_e50_b48

python main_base_bak.py actnet RGB --arch dmynet50 --num_segments 16 --lr 0.02 --epochs 50 --lr_steps 20 40  --npb --folder_suffix _self --rescale_to  84 --dmy --num_filters_list 64 32 16 8 --default_signal 3 --batch-size 48 -j 72 --gpus 0 1 2 3 --exp_header act_dres4_84_3_e50_b48

python main_base_bak.py actnet RGB --arch dmynet50 --num_segments 16 --lr 0.02 --epochs 50 --lr_steps 20 40  --npb --folder_suffix _self --rescale_to 224 --dmy --num_filters_list 64 32 16 8 --default_signal 1 --batch-size 48 -j 48 --gpus 0 1 2 3 --exp_header act_dres4_224_1_e50_b48

python main_base_bak.py actnet RGB --arch dmynet50 --num_segments 16 --lr 0.02 --epochs 50 --lr_steps 20 40  --npb --folder_suffix _self --rescale_to 224 --dmy --num_filters_list 64 32 16 8 --default_signal 2 --batch-size 48 -j 48 --gpus 0 1 2 3 --exp_header act_dres4_224_2_e50_b48

python main_base_bak.py actnet RGB --arch dmynet50 --num_segments 16 --lr 0.02 --epochs 50 --lr_steps 20 40  --npb --folder_suffix _self --rescale_to 224 --dmy --num_filters_list 64 32 16 8 --default_signal 3 --batch-size 48 -j 48 --gpus 0 1 2 3 --exp_header act_dres4_224_3_e50_b48

python main_base_bak.py actnet RGB --arch dmynet50 --num_segments 16 --lr 0.02 --epochs 10 --lr_steps 20 40  --npb --folder_suffix _self --rescale_to 224 --dmy --num_filters_list 64 32 16 8 --default_signal 0 --batch-size 48 -j 48 --gpus 0 1 2 3 --exp_header act_dres4_224_0_e10_b48


# TODO TSN (ablation)
python main_base_bak.py actnet RGB --arch resnet50 --num_segments 16 --lr 0.02 --epochs 50 --lr_steps 20 40 --batch-size 48 -j 36 --npb --rescale_to 224 --folder_suffix _self --exp_header actnet_resnet50_t16_s224_e50_b48 --gpus 0 1 2 3

#TODO adaptive
python main_base_bak.py actnet RGB --arch dmynet50 --num_segments 16 --lr 0.001 --epochs 50 --lr_steps 50 100 --batch-size 8 -j 72 --npb --folder_suffix _self --ada_reso_skip --policy_backbone mobilenet_v2 --reso_list 224 168 112 84 --backbone_list dmynet50 --skip_list 1 2 4 --policy_input_offset 3 --dmy --num_filters_list 64 32 16 8  --accuracy_weight 0.9 --efficency_weight 0.1 --uniform_loss_weight 3.0  --use_gflops_loss --exp_decay --init_tau 5 --exp_header act_t16_dmy3124_a.9e.1_ed5_ft10ds_lr.001_gu3_debug --gpus 0 1




#TODO adaptive (load from 50 epoch tsn)
python main_base_bak.py actnet RGB --arch dmynet50 --num_segments 16 --lr 0.001 --epochs 50 --lr_steps 50 100 --batch-size 8 -j 72 --npb --folder_suffix _self --ada_reso_skip --policy_backbone mobilenet_v2 --reso_list 224 168 112 84 --backbone_list dmynet50 --skip_list 1 2 4 --policy_input_offset 3 --dmy --num_filters_list 64 32 16 8  --accuracy_weight 0.9 --efficency_weight 0.1 --uniform_loss_weight 3.0  --use_gflops_loss --exp_decay --init_tau 5 --exp_header act_t16_dmy3124_a.9e.1_ed5_ft50_lr.001_gu3_debug --gpus 0 1 --model_paths g0322-003509_act_dres4_224_0_e50_b48/models/ckpt.best.pth.tar

#TODO adaptive (load from 10 epoch tsn)
python main_base_bak.py actnet RGB --arch dmynet50 --num_segments 16 --lr 0.001 --epochs 50 --lr_steps 50 100 --batch-size 48 -j 72 --npb --folder_suffix _self --ada_reso_skip --policy_backbone mobilenet_v2 --reso_list 224 168 112 84 --backbone_list dmynet50 --skip_list 1 2 4 --policy_input_offset 3 --dmy --num_filters_list 64 32 16 8  --accuracy_weight 0.9 --efficency_weight 0.1 --uniform_loss_weight 3.0  --use_gflops_loss --exp_decay --init_tau 5 --exp_header act_t16_dmy3124_a.9e.1_ed5_ft10_lr.001_gu3 --gpus 0 1 2 3 --model_paths g0322-084931_act_dres4_224_0_e10_b48/models/ckpt.best.pth.tar



#TODO channel {64,48,32,16}
#TODO(same reso, different width)
python main_base_bak.py actnet RGB --arch dmynet50 --num_segments 16 --lr 0.02 --epochs 50 --lr_steps 20 40  --npb --folder_suffix _self --rescale_to 224 --dmy --num_filters_list 64 48 32 16 --default_signal 0 --batch-size 48 -j 12 --gpus 0 1 2 3 --exp_header act_dres4Q_224_0_e50_b48

python main_base_bak.py actnet RGB --arch dmynet50 --num_segments 16 --lr 0.02 --epochs 50 --lr_steps 20 40  --npb --folder_suffix _self --rescale_to 224 --dmy --num_filters_list 64 48 32 16 --default_signal 1 --batch-size 48 -j 12 --gpus 0 1 2 3 --exp_header act_dres4Q_224_1_e50_b48

python main_base_bak.py actnet RGB --arch dmynet50 --num_segments 16 --lr 0.02 --epochs 50 --lr_steps 20 40  --npb --folder_suffix _self --rescale_to 224 --dmy --num_filters_list 64 48 32 16 --default_signal 2 --batch-size 48 -j 12 --gpus 0 1 2 3 --exp_header act_dres4Q_224_2_e50_b48

python main_base_bak.py actnet RGB --arch dmynet50 --num_segments 16 --lr 0.02 --epochs 50 --lr_steps 20 40  --npb --folder_suffix _self --rescale_to 224 --dmy --num_filters_list 64 48 32 16 --default_signal 3 --batch-size 48 -j 12 --gpus 0 1 2 3 --exp_header act_dres4Q_224_3_e50_b48

# TODO (diff reso, same width)
python main_base_bak.py actnet RGB --arch dmynet50 --num_segments 16 --lr 0.02 --epochs 50 --lr_steps 20 40  --npb --folder_suffix _self --rescale_to 168 --dmy --num_filters_list 64 48 32 16 --default_signal 0 --batch-size 48 -j 12 --gpus 0 1 2 3 --exp_header act_dres4Q_168_0_e50_b48

python main_base_bak.py actnet RGB --arch dmynet50 --num_segments 16 --lr 0.02 --epochs 50 --lr_steps 20 40  --npb --folder_suffix _self --rescale_to 112 --dmy --num_filters_list 64 48 32 16 --default_signal 0 --batch-size 48 -j 12 --gpus 0 1 2 3 --exp_header act_dres4Q_112_0_e50_b48

python main_base_bak.py actnet RGB --arch dmynet50 --num_segments 16 --lr 0.02 --epochs 50 --lr_steps 20 40  --npb --folder_suffix _self --rescale_to 84 --dmy --num_filters_list 64 48 32 16 --default_signal 0 --batch-size 48 -j 12 --gpus 0 1 2 3 --exp_header act_dres4Q_84_0_e50_b48

# TODO (diff reso, diff width)
python main_base_bak.py actnet RGB --arch dmynet50 --num_segments 16 --lr 0.02 --epochs 50 --lr_steps 20 40  --npb --folder_suffix _self --rescale_to 168 --dmy --num_filters_list 64 48 32 16 --default_signal 1 --batch-size 48 -j 12 --gpus 0 1 2 3 --exp_header act_dres4Q_168_1_e50_b48

python main_base_bak.py actnet RGB --arch dmynet50 --num_segments 16 --lr 0.02 --epochs 50 --lr_steps 20 40  --npb --folder_suffix _self --rescale_to 112 --dmy --num_filters_list 64 48 32 16 --default_signal 2 --batch-size 48 -j 12 --gpus 0 1 2 3 --exp_header act_dres4Q_112_2_e50_b48

python main_base_bak.py actnet RGB --arch dmynet50 --num_segments 16 --lr 0.02 --epochs 50 --lr_steps 20 40  --npb --folder_suffix _self --rescale_to 84 --dmy --num_filters_list 64 48 32 16 --default_signal 3 --batch-size 48 -j 12 --gpus 0 1 2 3 --exp_header act_dres4Q_84_3_e50_b48

# TODO (adaptive, pretrain epo=50)


# TODO(debug)
python main_base_bak.py actnet RGB --arch dmynet50 --num_segments 16 --lr 0.02 --epochs 50 --lr_steps 20 40  --npb --folder_suffix _self --rescale_to 224 --dmy --num_filters_list 64 64 32 16 --default_signal 1 --batch-size 48 -j 48 --gpus 0 1 2 3 --exp_header act_dres4dbg64_224_1_e50_b48
python main_base_bak.py actnet RGB --arch dmynet50 --num_segments 16 --lr 0.02 --epochs 50 --lr_steps 20 40  --npb --folder_suffix _self --rescale_to 224 --dmy --num_filters_list 64 48 32 16 --default_signal 1 --batch-size 48 -j 48 --gpus 0 1 2 3 --exp_header act_dres4dbg48_224_1_e50_b48

# TODO (adaptive, pretrain epo=10)
python main_base_bak.py actnet RGB --arch dmynet50 --num_segments 16 --lr 0.001 --epochs 50 --lr_steps 50 100 --batch-size 48 -j 12 --npb --folder_suffix _self --ada_reso_skip --policy_backbone mobilenet_v2 --reso_list 224 168 112 84 --backbone_list dmynet50 --skip_list 1 2 4 --policy_input_offset 3 --dmy --num_filters_list 64 48 32 16  --accuracy_weight 0.9 --efficency_weight 0.1 --uniform_loss_weight 3.0  --use_gflops_loss --exp_decay --init_tau 5 --exp_header act_ada_dres4Q_a.9e.1_ed5_ft10_lr.001_gu3 --gpus 0 1 2 3 --model_paths g0322-165225_act_dres4Q_224_0_e10_b48/models/ckpt.best.pth.tar

python main_base_bak.py actnet RGB --arch dmynet50 --num_segments 16 --lr 0.001 --epochs 50 --lr_steps 50 100 --batch-size 48 -j 12 --npb --folder_suffix _self --ada_reso_skip --policy_backbone mobilenet_v2 --reso_list 224 168 112 84 --backbone_list dmynet50 --skip_list 1 2 4 --policy_input_offset 3 --dmy --num_filters_list 64 48 32 16  --accuracy_weight 0.9 --efficency_weight 0.1 --uniform_loss_weight 3.0  --use_gflops_loss --exp_decay --init_tau 5 --exp_header act_ada_dres4Q_a.9e.1_ed5_ft50_lr.001_gu3 --gpus 0 1 2 3 --model_paths g0322-165206_act_dres4Q_224_0_e50_b48/models/ckpt.best.pth.tar

# TODO (random, epo=10/50)
python main_base_bak.py actnet RGB --arch dmynet50 --num_segments 16 --lr 0.001 --epochs 50 --lr_steps 50 100 --batch-size 48 -j 12 --npb --folder_suffix _self --ada_reso_skip --policy_backbone mobilenet_v2 --reso_list 224 168 112 84 --backbone_list dmynet50 --skip_list 1 2 4 --policy_input_offset 3 --dmy --num_filters_list 64 48 32 16 --use_gflops_loss  --exp_decay --init_tau 5 --random_policy --exp_header act_ada_dres4Q_ft10_lr.001_rand --gpus 0 1 2 3 --model_paths g0322-165225_act_dres4Q_224_0_e10_b48/models/ckpt.best.pth.tar

python main_base_bak.py actnet RGB --arch dmynet50 --num_segments 16 --lr 0.001 --epochs 50 --lr_steps 50 100 --batch-size 48 -j 12 --npb --folder_suffix _self --ada_reso_skip --policy_backbone mobilenet_v2 --reso_list 224 168 112 84 --backbone_list dmynet50 --skip_list 1 2 4 --policy_input_offset 3 --dmy --num_filters_list 64 48 32 16 --use_gflops_loss  --exp_decay --init_tau 5 --random_policy --exp_header act_ada_dres4Q_ft50_lr.001_rand --gpus 0 1 2 3 --model_paths g0322-165206_act_dres4Q_224_0_e50_b48/models/ckpt.best.pth.tar


# TODO (all)
python main_base_bak.py actnet RGB --arch dmynet50 --num_segments 16 --lr 0.001 --epochs 50 --lr_steps 50 100 --batch-size 48 -j 12 --npb --folder_suffix _self --ada_reso_skip --policy_backbone mobilenet_v2 --reso_list 224 168 112 84 --backbone_list dmynet50 --skip_list 1 2 4 --policy_input_offset 3 --dmy --num_filters_list 64 48 32 16 --use_gflops_loss  --exp_decay --init_tau 5 --all_policy --exp_header act_ada_dres4Q_ft10_lr.001_all --gpus 0 1 2 3 --model_paths g0322-165225_act_dres4Q_224_0_e10_b48/models/ckpt.best.pth.tar

python main_base_bak.py actnet RGB --arch dmynet50 --num_segments 16 --lr 0.001 --epochs 50 --lr_steps 50 100 --batch-size 48 -j 12 --npb --folder_suffix _self --ada_reso_skip --policy_backbone mobilenet_v2 --reso_list 224 168 112 84 --backbone_list dmynet50 --skip_list 1 2 4 --policy_input_offset 3 --dmy --num_filters_list 64 48 32 16 --use_gflops_loss  --exp_decay --init_tau 5 --all_policy --exp_header act_ada_dres4Q_ft50_lr.001_all --gpus 0 1 2 3 --model_paths g0322-165206_act_dres4Q_224_0_e50_b48/models/ckpt.best.pth.tar