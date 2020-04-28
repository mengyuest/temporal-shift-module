
#TODO we just try 224,168,112 with 64,48,32, and adaptive one
#TODO for dilation we try [0,0,0] or [1,0,0] or [1,1,0] or [2,1,0]

# (comparison group) ~ [0, 0, 0]
python main.py actnet RGB --arch dmynet50 --num_segments 16 --lr 0.02 --epochs 50 --lr_steps 20 40 --npb --folder_suffix _self --rescale_to 224 --dmy --num_filters_list 64 48 32 --default_signal 0 --batch-size 48 -j 72 --gpus 0 1 2 3 --exp_header act_dres4_dila000_224_0_e50_b48

python main.py actnet RGB --arch dmynet50 --num_segments 16 --lr 0.02 --epochs 50 --lr_steps 20 40 --npb --folder_suffix _self --rescale_to 168 --dmy --num_filters_list 64 48 32 --default_signal 1 --batch-size 48 -j 72 --gpus 0 1 2 3 --exp_header act_dres4_dila000_168_1_e50_b48

python main.py actnet RGB --arch dmynet50 --num_segments 16 --lr 0.02 --epochs 50 --lr_steps 20 40 --npb --folder_suffix _self --rescale_to 112 --dmy --num_filters_list 64 48 32 --default_signal 2 --batch-size 48 -j 72 --gpus 0 1 2 3 --exp_header act_dres4_dila000_112_2_e50_b48




# try [1,0,0]
python main.py actnet RGB --arch dmynet50 --num_segments 16 --lr 0.02 --epochs 50 --lr_steps 20 40 --npb --folder_suffix _self --rescale_to 224 --dmy --num_filters_list 64 48 32 --default_signal 0 --dilation_list 1 0 0 --batch-size 48 -j 72 --gpus 0 1 2 3 --exp_header act_dres4_dila100_224_0_e50_b48

python main.py actnet RGB --arch dmynet50 --num_segments 16 --lr 0.02 --epochs 50 --lr_steps 20 40 --npb --folder_suffix _self --rescale_to 168 --dmy --num_filters_list 64 48 32 --default_signal 1 --dilation_list 1 0 0 --batch-size 48 -j 72 --gpus 0 1 2 3 --exp_header act_dres4_dila100_168_1_e50_b48

python main.py actnet RGB --arch dmynet50 --num_segments 16 --lr 0.02 --epochs 50 --lr_steps 20 40 --npb --folder_suffix _self --rescale_to 112 --dmy --num_filters_list 64 48 32 --default_signal 2 --dilation_list 1 0 0 --batch-size 48 -j 72 --gpus 0 1 2 3 --exp_header act_dres4_dila100_112_2_e50_b48



# try [1,1,0]
python main.py actnet RGB --arch dmynet50 --num_segments 16 --lr 0.02 --epochs 50 --lr_steps 20 40 --npb --folder_suffix _self --rescale_to 224 --dmy --num_filters_list 64 48 32 --default_signal 0 --dilation_list 1 1 0 --batch-size 48 -j 72 --gpus 0 1 2 3 --exp_header act_dres4_dila110_224_0_e50_b48

python main.py actnet RGB --arch dmynet50 --num_segments 16 --lr 0.02 --epochs 50 --lr_steps 20 40 --npb --folder_suffix _self --rescale_to 168 --dmy --num_filters_list 64 48 32 --default_signal 1 --dilation_list 1 1 0 --batch-size 48 -j 72 --gpus 0 1 2 3 --exp_header act_dres4_dila110_168_1_e50_b48

python main.py actnet RGB --arch dmynet50 --num_segments 16 --lr 0.02 --epochs 50 --lr_steps 20 40 --npb --folder_suffix _self --rescale_to 112 --dmy --num_filters_list 64 48 32 --default_signal 2 --dilation_list 1 1 0 --batch-size 48 -j 72 --gpus 0 1 2 3 --exp_header act_dres4_dila110_112_2_e50_b48



# try [2,1,0]
python main.py actnet RGB --arch dmynet50 --num_segments 16 --lr 0.02 --epochs 50 --lr_steps 20 40 --npb --folder_suffix _self --rescale_to 224 --dmy --num_filters_list 64 48 32 --default_signal 0 --dilation_list 2 1 0 --batch-size 48 -j 72 --gpus 0 1 2 3 --exp_header act_dres4_dila210_224_0_e50_b48

python main.py actnet RGB --arch dmynet50 --num_segments 16 --lr 0.02 --epochs 50 --lr_steps 20 40 --npb --folder_suffix _self --rescale_to 168 --dmy --num_filters_list 64 48 32 --default_signal 1 --dilation_list 2 1 0 --batch-size 48 -j 72 --gpus 0 1 2 3 --exp_header act_dres4_dila210_168_1_e50_b48

python main.py actnet RGB --arch dmynet50 --num_segments 16 --lr 0.02 --epochs 50 --lr_steps 20 40 --npb --folder_suffix _self --rescale_to 112 --dmy --num_filters_list 64 48 32 --default_signal 2 --dilation_list 2 1 0 --batch-size 48 -j 72 --gpus 0 1 2 3 --exp_header act_dres4_dila210_112_2_e50_b48


#TODO phase-2 finetuning (w/wo cross-entropy, 10/50 epo)
python main.py actnet RGB --arch dmynet50 --num_segments 16 --lr 0.001 --epochs 50 --lr_steps 50 100 --batch-size 48 -j 36 --npb --folder_suffix _self --ada_reso_skip --policy_backbone mobilenet_v2 --reso_list 224 168 112 84 --backbone_list dmynet50 --skip_list 1 2 4 --policy_input_offset 3 --dmy --num_filters_list 64 48 32 16  --accuracy_weight 0.9 --efficency_weight 0.1 --uniform_loss_weight 3.0  --use_gflops_loss --exp_decay --init_tau 5 --exp_header act_ada_dres4Q_a.9e.1_ed5_ft10_lr.001_gu3 --gpus 0 1 2 3 --model_paths g0322-165225_act_dres4Q_224_0_e10_b48/models/ckpt.best.pth.tar

python main.py actnet RGB --arch dmynet50 --num_segments 16 --lr 0.001 --epochs 50 --lr_steps 50 100 --batch-size 48 -j 36 --npb --folder_suffix _self --ada_reso_skip --policy_backbone mobilenet_v2 --reso_list 224 168 112 84 --backbone_list dmynet50 --skip_list 1 2 4 --policy_input_offset 3 --dmy --num_filters_list 64 48 32 16  --accuracy_weight 0.9 --efficency_weight 0.1 --uniform_loss_weight 3.0  --use_gflops_loss --exp_decay --init_tau 5 --exp_header act_ada_dres4Q_a.9e.1_ed5_ft50_lr.001_gu3 --gpus 0 1 2 3 --model_paths g0322-165206_act_dres4Q_224_0_e50_b48/models/ckpt.best.pth.tar


python main.py actnet RGB --arch dmynet50 --num_segments 16 --lr 0.001 --epochs 50 --lr_steps 50 100 --batch-size 48 -j 36 --npb --folder_suffix _self --ada_reso_skip --policy_backbone mobilenet_v2 --reso_list 224 168 112 84 --backbone_list dmynet50 --skip_list 1 2 4 --policy_input_offset 3 --dmy --num_filters_list 64 48 32 16  --accuracy_weight 0.9 --efficency_weight 0.1 --uniform_loss_weight 3.0  --use_gflops_loss --uniform_cross_entropy --exp_decay --init_tau 5 --exp_header act_ada_dres4Q_a.9e.1_ed5_ft10_lr.001_gu3_uc --gpus 0 1 2 3 --model_paths g0322-165225_act_dres4Q_224_0_e10_b48/models/ckpt.best.pth.tar

python main.py actnet RGB --arch dmynet50 --num_segments 16 --lr 0.001 --epochs 50 --lr_steps 50 100 --batch-size 48 -j 36 --npb --folder_suffix _self --ada_reso_skip --policy_backbone mobilenet_v2 --reso_list 224 168 112 84 --backbone_list dmynet50 --skip_list 1 2 4 --policy_input_offset 3 --dmy --num_filters_list 64 48 32 16  --accuracy_weight 0.9 --efficency_weight 0.1 --uniform_loss_weight 3.0  --use_gflops_loss --uniform_cross_entropy --exp_decay --init_tau 5 --exp_header act_ada_dres4Q_a.9e.1_ed5_ft50_lr.001_gu3_uc --gpus 0 1 2 3 --model_paths g0322-165206_act_dres4Q_224_0_e50_b48/models/ckpt.best.pth.tar

