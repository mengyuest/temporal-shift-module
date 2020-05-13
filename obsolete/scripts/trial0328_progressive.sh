# TODO progressive training (64 48 32 ) x (224 168 112)
#TODO-1 (224->168->112)
#TODO 10 epo
#TODO(done)
python main_base.py actnet RGB --arch dmynet50 --num_segments 16 --lr 0.02 --epochs 10 --lr_steps 20 40 --npb --folder_suffix _self --rescale_to 224 --dmy --num_filters_list 64 48 32 --default_signal 0 --batch-size 48 -j 72 --gpus 0 1 2 3 --exp_header act_dres3_chain0_224_0_e10_b48

#TODO(done)
python main_base.py actnet RGB --arch dmynet50 --num_segments 16 --lr 0.02 --epochs 10 --lr_steps 20 40 --npb --folder_suffix _self --rescale_to 168 --dmy --num_filters_list 64 48 32 --default_signal 1 --batch-size 48 -j 72 --gpus 0 1 2 3 --exp_header act_dres3_chain1_168_1_e10_b48_pre224 --base_pretrained_from g0329-073458_act_dres3_chain0_224_0_e10_b48/models/ckpt.best.pth.tar --ignore_new_fc_weight

#TODO(done)
python main_base.py actnet RGB --arch dmynet50 --num_segments 16 --lr 0.02 --epochs 10 --lr_steps 20 40 --npb --folder_suffix _self --rescale_to 112 --dmy --num_filters_list 64 48 32 --default_signal 2 --batch-size 48 -j 72 --gpus 0 1 2 3 --exp_header act_dres3_chain2_112_2_e10_b48_pre168 --base_pretrained_from g0329-082233_act_dres3_chain1_168_1_e10_b48_pre224/models/ckpt.best.pth.tar --ignore_new_fc_weight




#TODO-2 (112->168->224)
#TODO 10 epo
#TODO(done)
python main_base.py actnet RGB --arch dmynet50 --num_segments 16 --lr 0.02 --epochs 10 --lr_steps 20 40 --npb --folder_suffix _self --rescale_to 112 --dmy --num_filters_list 64 48 32 --default_signal 2 --batch-size 48 -j 72 --gpus 0 1 2 3 --exp_header act_dres3_rchain0_112_2_e10_b48

#TODO(done)
python main_base.py actnet RGB --arch dmynet50 --num_segments 16 --lr 0.02 --epochs 10 --lr_steps 20 40 --npb --folder_suffix _self --rescale_to 168 --dmy --num_filters_list 64 48 32 --default_signal 1 --batch-size 48 -j 72 --gpus 0 1 2 3 --exp_header act_dres3_rchain1_168_1_e10_b48_pre112 --base_pretrained_from g0329-081552_act_dres3_rchain0_112_2_e10_b48/models/ckpt.best.pth.tar  --ignore_new_fc_weight

#TODO(done)
python main_base.py actnet RGB --arch dmynet50 --num_segments 16 --lr 0.02 --epochs 10 --lr_steps 20 40 --npb --folder_suffix _self --rescale_to 224 --dmy --num_filters_list 64 48 32 --default_signal 0 --batch-size 48 -j 72 --gpus 0 1 2 3 --exp_header act_dres3_rchain2_224_0_e10_b48_pre168 --base_pretrained_from g0329-091018_act_dres3_rchain1_168_1_e10_b48_pre112/models/ckpt.best.pth.tar  --ignore_new_fc_weight


#TODO-3 （224->168->112）, dilation = ([1,0,0] or [1,1,0] or [2,1,0])
#TODO 10 epo
#TODO(done)
python main_base.py actnet RGB --arch dmynet50 --num_segments 16 --lr 0.02 --epochs 10 --lr_steps 20 40 --npb --folder_suffix _self --rescale_to 224 --dmy --num_filters_list 64 48 32 --default_signal 0 --dilation_list 1 0 0 --batch-size 48 -j 72 --gpus 0 1 2 3 --exp_header act_dres3_d0chain0_224_0_e10_b48

#TODO(done)
python main_base.py actnet RGB --arch dmynet50 --num_segments 16 --lr 0.02 --epochs 10 --lr_steps 20 40 --npb --folder_suffix _self --rescale_to 168 --dmy --num_filters_list 64 48 32 --default_signal 1 --dilation_list 1 0 0 --batch-size 48 -j 72 --gpus 0 1 2 3 --exp_header act_dres3_d0chain1_168_1_e10_b48_pre224 --base_pretrained_from g0329-091438_act_dres3_d0chain0_224_0_e10_b48/models/ckpt.best.pth.tar --ignore_new_fc_weight

#TODO(done)
python main_base.py actnet RGB --arch dmynet50 --num_segments 16 --lr 0.02 --epochs 10 --lr_steps 20 40 --npb --folder_suffix _self --rescale_to 112 --dmy --num_filters_list 64 48 32 --default_signal 2 --dilation_list 1 0 0 --batch-size 48 -j 72 --gpus 0 1 2 3 --exp_header act_dres3_d0chain2_112_2_e10_b48_pre168 --base_pretrained_from g0329-095950_act_dres3_d0chain1_168_1_e10_b48_pre224/models/ckpt.best.pth.tar --ignore_new_fc_weight


#TODO(done)
python main_base.py actnet RGB --arch dmynet50 --num_segments 16 --lr 0.02 --epochs 10 --lr_steps 20 40 --npb --folder_suffix _self --rescale_to 224 --dmy --num_filters_list 64 48 32 --default_signal 0 --dilation_list 1 1 0 --batch-size 48 -j 72 --gpus 0 1 2 3 --exp_header act_dres3_d1chain0_224_0_e10_b48
#TODO(done)
python main_base.py actnet RGB --arch dmynet50 --num_segments 16 --lr 0.02 --epochs 10 --lr_steps 20 40 --npb --folder_suffix _self --rescale_to 168 --dmy --num_filters_list 64 48 32 --default_signal 1 --dilation_list 1 1 0 --batch-size 48 -j 72 --gpus 0 1 2 3 --exp_header act_dres3_d1chain1_168_1_e10_b48_pre224 --base_pretrained_from g0329-101857_act_dres3_d1chain0_224_0_e10_b48/models/ckpt.best.pth.tar --ignore_new_fc_weight

python main_base.py actnet RGB --arch dmynet50 --num_segments 16 --lr 0.02 --epochs 10 --lr_steps 20 40 --npb --folder_suffix _self --rescale_to 112 --dmy --num_filters_list 64 48 32 --default_signal 2 --dilation_list 1 1 0 --batch-size 48 -j 72 --gpus 0 1 2 3 --exp_header act_dres3_d1chain2_112_2_e10_b48_pre168 --base_pretrained_from g0329-110714_act_dres3_d1chain1_168_1_e10_b48_pre224/models/ckpt.best.pth.tar --ignore_new_fc_weight


#TODO(done)
python main_base.py actnet RGB --arch dmynet50 --num_segments 16 --lr 0.02 --epochs 10 --lr_steps 20 40 --npb --folder_suffix _self --rescale_to 224 --dmy --num_filters_list 64 48 32 --default_signal 0 --dilation_list 2 1 0 --batch-size 48 -j 72 --gpus 0 1 2 3 --exp_header act_dres3_d2chain0_224_0_e10_b48

python main_base.py actnet RGB --arch dmynet50 --num_segments 16 --lr 0.02 --epochs 10 --lr_steps 20 40 --npb --folder_suffix _self --rescale_to 168 --dmy --num_filters_list 64 48 32 --default_signal 1 --dilation_list 2 1 0 --batch-size 48 -j 72 --gpus 0 1 2 3 --exp_header act_dres3_d2chain1_168_1_e10_b48_pre224 --base_pretrained_from g0329-105301_act_dres3_d2chain0_224_0_e10_b48/models/ckpt.best.pth.tar --ignore_new_fc_weight

python main_base.py actnet RGB --arch dmynet50 --num_segments 16 --lr 0.02 --epochs 10 --lr_steps 20 40 --npb --folder_suffix _self --rescale_to 112 --dmy --num_filters_list 64 48 32 --default_signal 2 --dilation_list 2 1 0 --batch-size 48 -j 72 --gpus 0 1 2 3 --exp_header act_dres3_d2chain2_112_2_e10_b48_pre168 --base_pretrained_from XXXd2chain1_e10 --ignore_new_fc_weight




# TODO-4 Adaptive
#TODO exp-0  0->0->0 224->168->112
python main_base.py actnet RGB --arch dmynet50 --num_segments 16 --lr 0.001 --epochs 50 --lr_steps 50 100 --batch-size 48 -j 72 --npb --folder_suffix _self --ada_reso_skip --policy_backbone mobilenet_v2 --reso_list 224 168 112 --backbone_list dmynet50 --skip_list 1 2 4 --policy_input_offset 2 --dmy --num_filters_list 64 48 32  --accuracy_weight 0.9 --efficency_weight 0.1 --uniform_loss_weight 3.0  --use_gflops_loss --exp_decay --init_tau 5 --exp_header act_ada_chain_a.9e.1_ed5_cft10_lr.001_gu3 --gpus 0 1 2 3 --model_paths g0329-073458_act_dres3_chain0_224_0_e10_b48/models/ckpt.best.pth.tar g0329-082233_act_dres3_chain1_168_1_e10_b48_pre224/models/ckpt.best.pth.tar g0329-085151_act_dres3_chain2_112_2_e10_b48_pre168/models/ckpt.best.pth.tar

python main_base.py actnet RGB --arch dmynet50 --num_segments 16 --lr 0.001 --epochs 50 --lr_steps 50 100 --batch-size 48 -j 72 --npb --folder_suffix _self --ada_reso_skip --policy_backbone mobilenet_v2 --reso_list 224 168 112 --backbone_list dmynet50 --skip_list 1 2 4 --policy_input_offset 2 --dmy --num_filters_list 64 48 32  --accuracy_weight 0.9 --efficency_weight 0.1 --uniform_loss_weight 3.0  --use_gflops_loss --exp_decay --init_tau 5 --exp_header act_ada_chain_a.9e.1_ed5_cft10_lr.001_gu3_uce --gpus 0 1 2 3 --model_paths g0329-073458_act_dres3_chain0_224_0_e10_b48/models/ckpt.best.pth.tar g0329-082233_act_dres3_chain1_168_1_e10_b48_pre224/models/ckpt.best.pth.tar g0329-085151_act_dres3_chain2_112_2_e10_b48_pre168/models/ckpt.best.pth.tar --uniform_cross_entropy

##TODO exp-1  0->0->0 112->168->224
#python main_base.py actnet RGB --arch dmynet50 --num_segments 16 --lr 0.001 --epochs 50 --lr_steps 50 100 --batch-size 48 -j 72 --npb --folder_suffix _self --ada_reso_skip --policy_backbone mobilenet_v2 --reso_list 224 168 112 --backbone_list dmynet50 --skip_list 1 2 4 --policy_input_offset 2 --dmy --num_filters_list 64 48 32  --accuracy_weight 0.9 --efficency_weight 0.1 --uniform_loss_weight 3.0  --use_gflops_loss --exp_decay --init_tau 5 --exp_header act_ada_chain_a.9e.1_ed5_cft10_lr.001_gu3 --gpus 0 1 2 3 --model_paths g0329-073458_act_dres3_chain0_224_0_e10_b48/models/ckpt.best.pth.tar g0329-082233_act_dres3_chain1_168_1_e10_b48_pre224/models/ckpt.best.pth.tar g0329-085151_act_dres3_chain2_112_2_e10_b48_pre168/models/ckpt.best.pth.tar

#TODO exp-2  1->0->0 224->168->112
python main_base.py actnet RGB --arch dmynet50 --num_segments 16 --lr 0.001 --epochs 50 --lr_steps 50 100 --batch-size 48 -j 72 --npb --folder_suffix _self --ada_reso_skip --policy_backbone mobilenet_v2 --reso_list 224 168 112 --backbone_list dmynet50 --skip_list 1 2 4 --policy_input_offset 2 --dmy --num_filters_list 64 48 32  --accuracy_weight 0.9 --efficency_weight 0.1 --uniform_loss_weight 3.0  --use_gflops_loss --exp_decay --init_tau 5 --dilation_list 1 0 0 --exp_header act_ada_d0chain_a.9e.1_ed5_cft10_lr.001_gu3 --gpus 0 1 2 3 --model_paths g0329-091438_act_dres3_d0chain0_224_0_e10_b48/models/ckpt.best.pth.tar g0329-095950_act_dres3_d0chain1_168_1_e10_b48_pre224/models/ckpt.best.pth.tar g0329-102713_act_dres3_d0chain2_112_2_e10_b48_pre168/models/ckpt.best.pth.tar

python main_base.py actnet RGB --arch dmynet50 --num_segments 16 --lr 0.001 --epochs 50 --lr_steps 50 100 --batch-size 48 -j 72 --npb --folder_suffix _self --ada_reso_skip --policy_backbone mobilenet_v2 --reso_list 224 168 112 --backbone_list dmynet50 --skip_list 1 2 4 --policy_input_offset 2 --dmy --num_filters_list 64 48 32  --accuracy_weight 0.9 --efficency_weight 0.1 --uniform_loss_weight 3.0  --use_gflops_loss --exp_decay --init_tau 5 --dilation_list 1 0 0 --exp_header act_ada_d0chain_a.9e.1_ed5_cft10_lr.001_gu3_uce --gpus 0 1 2 3 --model_paths g0329-091438_act_dres3_d0chain0_224_0_e10_b48/models/ckpt.best.pth.tar g0329-095950_act_dres3_d0chain1_168_1_e10_b48_pre224/models/ckpt.best.pth.tar g0329-102713_act_dres3_d0chain2_112_2_e10_b48_pre168/models/ckpt.best.pth.tar --uniform_cross_entropy

##TODO exp-3  1->1->0 224->168->112
#python main_base.py actnet RGB --arch dmynet50 --num_segments 16 --lr 0.001 --epochs 50 --lr_steps 50 100 --batch-size 48 -j 72 --npb --folder_suffix _self --ada_reso_skip --policy_backbone mobilenet_v2 --reso_list 224 168 112 --backbone_list dmynet50 --skip_list 1 2 4 --policy_input_offset 2 --dmy --num_filters_list 64 48 32  --accuracy_weight 0.9 --efficency_weight 0.1 --uniform_loss_weight 3.0  --use_gflops_loss --exp_decay --init_tau 5 --exp_header act_ada_chain_a.9e.1_ed5_cft10_lr.001_gu3 --gpus 0 1 2 3 --model_paths g0329-073458_act_dres3_chain0_224_0_e10_b48/models/ckpt.best.pth.tar g0329-082233_act_dres3_chain1_168_1_e10_b48_pre224/models/ckpt.best.pth.tar g0329-085151_act_dres3_chain2_112_2_e10_b48_pre168/models/ckpt.best.pth.tar




#TODO (50 epo)
python main_base.py actnet RGB --arch dmynet50 --num_segments 16 --lr 0.02 --epochs 50 --lr_steps 20 40 --npb --folder_suffix _self --rescale_to 224 --dmy --num_filters_list 64 48 32 --default_signal 0 --batch-size 48 -j 72 --gpus 0 1 2 3 --exp_header act_dres3_chain0_224_0_e50_b48
python main_base.py actnet RGB --arch dmynet50 --num_segments 16 --lr 0.02 --epochs 50 --lr_steps 20 40 --npb --folder_suffix _self --rescale_to 168 --dmy --num_filters_list 64 48 32 --default_signal 1 --batch-size 48 -j 72 --gpus 0 1 2 3 --exp_header act_dres3_chain0_168_1_e50_b48_pre224 --base_pretrained_from XXXchain0
python main_base.py actnet RGB --arch dmynet50 --num_segments 16 --lr 0.02 --epochs 50 --lr_steps 20 40 --npb --folder_suffix _self --rescale_to 112 --dmy --num_filters_list 64 48 32 --default_signal 2 --batch-size 48 -j 72 --gpus 0 1 2 3 --exp_header act_dres3_chain0_112_2_e50_b48_pre168 --base_pretrained_from XXXchain1

#TODO (50 epo)
python main_base.py actnet RGB --arch dmynet50 --num_segments 16 --lr 0.02 --epochs 50 --lr_steps 20 40 --npb --folder_suffix _self --rescale_to 112 --dmy --num_filters_list 64 48 32 --default_signal 2 --batch-size 48 -j 72 --gpus 0 1 2 3 --exp_header act_dres3_rchain0_112_2_e50_b48
python main_base.py actnet RGB --arch dmynet50 --num_segments 16 --lr 0.02 --epochs 50 --lr_steps 20 40 --npb --folder_suffix _self --rescale_to 168 --dmy --num_filters_list 64 48 32 --default_signal 1 --batch-size 48 -j 72 --gpus 0 1 2 3 --exp_header act_dres3_rchain0_168_1_e50_b48_pre112 --base_pretrained_from XXXrchain0
python main_base.py actnet RGB --arch dmynet50 --num_segments 16 --lr 0.02 --epochs 50 --lr_steps 20 40 --npb --folder_suffix _self --rescale_to 224 --dmy --num_filters_list 64 48 32 --default_signal 0 --batch-size 48 -j 72 --gpus 0 1 2 3 --exp_header act_dres3_rchain0_168_0_e50_b48_pre168 --base_pretrained_from XXXrchain1







#TODO-* (224->168->112)
#TODO 10 epo, batchsize=72
#TODO(done)
python main_base.py actnet RGB --arch dmynet50 --num_segments 16 --lr 0.02 --epochs 10 --lr_steps 20 40 --npb --folder_suffix _self --rescale_to 224 --dmy --num_filters_list 64 48 32 --default_signal 0 --batch-size 72 -j 72 --gpus 0 1 2 3 --exp_header act_dres3_chain0_224_0_e10_b72

#TODO(done)
python main_base.py actnet RGB --arch dmynet50 --num_segments 16 --lr 0.02 --epochs 10 --lr_steps 20 40 --npb --folder_suffix _self --rescale_to 168 --dmy --num_filters_list 64 48 32 --default_signal 1 --batch-size 72 -j 72 --gpus 0 1 2 3 --exp_header act_dres3_chain1_168_1_e10_b72_pre224 --base_pretrained_from g0330-153232_act_dres3_chain0_224_0_e10_b72/models/ckpt.best.pth.tar --ignore_new_fc_weight

#TODO(done)
python main_base.py actnet RGB --arch dmynet50 --num_segments 16 --lr 0.02 --epochs 10 --lr_steps 20 40 --npb --folder_suffix _self --rescale_to 112 --dmy --num_filters_list 64 48 32 --default_signal 2 --batch-size 72 -j 72 --gpus 0 1 2 3 --exp_header act_dres3_chain2_112_2_e10_b72_pre168 --base_pretrained_from g0330-165648_act_dres3_chain1_168_1_e10_b72_pre224/models/ckpt.best.pth.tar --ignore_new_fc_weight


python main_base.py actnet RGB --arch dmynet50 --num_segments 16 --lr 0.001 --epochs 100 --lr_steps 150 200 --batch-size 48 -j 72 --npb --folder_suffix _self --ada_reso_skip --policy_backbone mobilenet_v2 --reso_list 224 168 112 --backbone_list dmynet50 --skip_list 1 2 4 --policy_input_offset 2 --dmy --num_filters_list 64 48 32  --accuracy_weight 0.9 --efficency_weight 0.1 --uniform_loss_weight 3.0  --use_gflops_loss --exp_decay --init_tau 5 --exp_header act_ada_chain_a.9e.1_ed5_cft10_lr.001_gu3 --gpus 0 1 2 3 --model_paths g0330-153232_act_dres3_chain0_224_0_e10_b72/models/ckpt.best.pth.tar g0330-165648_act_dres3_chain1_168_1_e10_b72_pre224/models/ckpt.best.pth.tar g0330-180039_act_dres3_chain2_112_2_e10_b72_pre168/models/ckpt.best.pth.tar