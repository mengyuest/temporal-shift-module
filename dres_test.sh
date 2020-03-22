# TODO TSN dresnet50 (signal=0,1,2,3)
python main.py actnet RGB --arch dmynet50 --num_segments 4 --lr 0.02 --epochs 2 --npb --folder_suffix _self  --exp_header debug --batch-size 6 -j 36 --gpus 0 1 --dmy --default_signal 0 --num_filters_list 64 32 16

python main.py actnet RGB --arch dmynet50 --num_segments 4 --lr 0.02 --epochs 2 --npb --folder_suffix _self  --exp_header debug --batch-size 6 -j 36 --gpus 0 1 --rescale_to  --dmy --default_signal 1 --num_filters_list 64 32 16


# TODO TSN activity net
python main.py actnet RGB --arch dmynet50 --num_segments 16 --lr 0.02 --epochs 50 --lr_steps 20 40  --npb --folder_suffix _self --rescale_to 224 --dmy --num_filters_list 64 32 16 8 --default_signal 0 --batch-size 48 -j 72 --gpus 0 1 2 3 --exp_header act_dres4_224_0_e50_b48_debug

python main.py actnet RGB --arch dmynet50 --num_segments 16 --lr 0.02 --epochs 50 --lr_steps 20 40  --npb --folder_suffix _self --rescale_to 168 --dmy --num_filters_list 64 32 16 8 --default_signal 1 --batch-size 48 -j 72 --gpus 0 1 2 3 --exp_header act_dres4_168_1_e50_b48_debug

python main.py actnet RGB --arch dmynet50 --num_segments 16 --lr 0.02 --epochs 50 --lr_steps 20 40  --npb --folder_suffix _self --rescale_to 112 --dmy --num_filters_list 64 32 16 8 --default_signal 2 --batch-size 48 -j 72 --gpus 0 1 2 3 --exp_header act_dres4_112_2_e50_b48_debug

python main.py actnet RGB --arch dmynet50 --num_segments 16 --lr 0.02 --epochs 50 --lr_steps 20 40  --npb --folder_suffix _self --rescale_to  84 --dmy --num_filters_list 64 32 16 8 --default_signal 3 --batch-size 48 -j 72 --gpus 0 1 2 3 --exp_header act_dres4_84_3_e50_b48_debug

# TODO TSN (ablation)
python main.py actnet RGB --arch resnet50 --num_segments 16 --lr 0.02 --epochs 50 --lr_steps 20 40 --batch-size 48 -j 36 --npb --rescale_to 224 --folder_suffix _self --exp_header actnet_resnet50_t16_s224_e50_b48 --gpus 0 1 2 3

#TODO adaptive
python main.py actnet RGB --arch dmynet50 --num_segments 16 --lr 0.001 --epochs 50 --lr_steps 50 100 --batch-size 8 -j 72 --npb --folder_suffix _self --ada_reso_skip --policy_backbone mobilenet_v2 --reso_list 224 168 112 84 --backbone_list dmynet50 --skip_list 1 2 4 --policy_input_offset 3 --dmy --num_filters_list 64 32 16 8  --accuracy_weight 0.9 --efficency_weight 0.1 --uniform_loss_weight 3.0  --use_gflops_loss --exp_decay --init_tau 5 --exp_header act_t16_dmy3124_a.9e.1_ed5_ft10ds_lr.001_gu3_debug --gpus 0 1