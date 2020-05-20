#TODO 0. Test for Adaptive ResNet (~73.8)
python -u main_base_bak.py actnet RGB --arch resnet50 --num_segments 16 --npb --exp_header X --ada_reso_skip --policy_backbone mobilenet_v2 --reso_list 224 168 112 84 --backbone_list resnet50 resnet34 resnet18 --skip_list 1 2 4 --accuracy_weight 0.9 --efficency_weight 0.1 --folder_suffix _self --exp_decay --init_tau 0.000001 --policy_also_backbone --policy_input_offset 3 --uniform_loss_weight 3.0 --use_gflops_loss --batch-size 48 -j 16 --gpus 0 1 2 3 --test_from test_suite/exp0_res_ada --uno_time | tee tmp_log.txt
OUTPUT0=`cat tmp_log.txt | tail -n 3`

#TODO 1. Test for SCSampler (~72.9)
python -u main_base_bak.py actnet RGB --arch resnet50 --num_segments 16 --npb --exp_header X --ada_reso_skip --policy_backbone mobilenet_v2 --backbone_list resnet50 --folder_suffix _self --reso_list 224 84 --policy_input_offset 0 --frame_independent --real_scsampler --consensus_type scsampler --batch-size 48 -j 16  --gpus 0 1 2 3 --test_from test_suite/exp1_res_scsampler --uno_time --uno_top_k | tee tmp_log.txt
OUTPUT1=`cat tmp_log.txt | tail -n 3`

#TODO 2. Test for Baseline TSN (~72.5)
python -u main_base_bak.py actnet RGB --arch resnet50 --num_segments 16 --npb --exp_header X --ada_reso_skip --policy_backbone resnet50 --offline_lstm_all --folder_suffix _self --frame_independent --batch-size 48 -j 16 --gpus 0 1 2 3 --test_from test_suite/exp2_res_uniform --uno_time | tee tmp_log.txt
OUTPUT2=`cat tmp_log.txt | tail -n 3`

#TODO 3. Test for Baseline LSTM (~71.2)
python -u main_base_bak.py actnet RGB --arch resnet50 --num_segments 16 --npb --exp_header X --ada_reso_skip --policy_backbone resnet50 --backbone_list resnet50 --offline_lstm_all --folder_suffix _self --batch-size 48 -j 16  --gpus 0 1 2 3 --test_from test_suite/exp3_res_lstm --uno_time | tee tmp_log.txt
OUTPUT3=`cat tmp_log.txt | tail -n 3`

#TODO 4. Test for Adaptive RANDOM (~65.7)
python -u main_base_bak.py actnet RGB --arch resnet50 --num_segments 16 --npb --exp_header X --ada_reso_skip --policy_backbone mobilenet_v2 --reso_list 224 168 112 84 --backbone_list resnet50 resnet34 resnet18 --skip_list 1 2 4 --policy_also_backbone --policy_input_offset 3 --folder_suffix _self --use_gflops_loss --random_policy  --batch-size 48 -j 16 --gpus 0 1 2 3 --test_from test_suite/exp4_res_rand --uno_time | tee tmp_log.txt
OUTPUT4=`cat tmp_log.txt | tail -n 3`

#TODO 5. Test for Adaptive ALL (~73.5)
python -u main_base_bak.py actnet RGB --arch resnet50 --num_segments 16 --npb --exp_header X --ada_reso_skip --reso_list 224 168 112 84 --skip_list 1 --backbone_list resnet50 resnet34 resnet18 --all_policy --policy_also_backbone --policy_input_offset 3 --folder_suffix _self --batch-size 48 -j 16  --gpus 0 1 2 3 --test_from test_suite/exp5_res_all --uno_time | tee tmp_log.txt
OUTPUT5=`cat tmp_log.txt | tail -n 3`

#TODO 6. Test for Adaptive EfficientNet (~79.7)
python -u main_base_bak.py actnet RGB --arch resnet50 --num_segments 16 --npb --exp_header X --ada_reso_skip --policy_backbone mobilenet_v2 --reso_list 224 168 112 84 --backbone_list efficientnet-b3 efficientnet-b1 efficientnet-b0 --skip_list 1 2 4 --accuracy_weight 0.9 --efficency_weight 0.1 --folder_suffix _self --policy_also_backbone --policy_input_offset 3 --uniform_loss_weight 5.0 --use_gflops_loss --init_tau 0.000001 --batch-size 48 -j 16  --gpus 0 1 2 3 --test_from test_suite/exp6_eff_ada --uno_time | tee tmp_log.txt
OUTPUT6=`cat tmp_log.txt | tail -n 3`

#TODO after ARNet: (DMY, MSD, MERS, though not good enough)
#TODO 7. Test for DmyNet (dynamic filters)
python -u main_base_bak.py actnet RGB --arch dmynet50 --num_segments 16 --npb --exp_header X --ada_reso_skip --policy_backbone mobilenet_v2 --reso_list 224 168 112 --backbone_list dmynet50 --skip_list 1 2 4  --accuracy_weight 0.9 --efficency_weight 0.1 --folder_suffix _self --policy_input_offset 2 --dmy --num_filters_list 64 48 32 --uniform_loss_weight 3.0 --use_gflops_loss --init_tau 0.000001  --batch-size 48 -j 16 --gpus 0 1 2 3 --test_from test_suite/exp7_dmy_ada --uno_time | tee tmp_log.txt
OUTPUT7=`cat tmp_log.txt | tail -n 3`
#TODO 8. Test for MSDNet (multi-exit hand crafted network)
python -u main_base_bak.py actnet RGB --arch msdnet --num_segments 16 --npb --exp_header X --ada_reso_skip --policy_backbone mobilenet_v2 --reso_list 224 112 --backbone_list msdnet --skip_list 1 2 4  --accuracy_weight 0.9 --efficency_weight 0.1 --folder_suffix _self --policy_input_offset 1 --msd --msd_indices_list 3 1 0 --uno_reso --uniform_loss_weight 3.0 --use_gflops_loss --init_tau 0.000001  --batch-size 48 -j 16 --gpus 0 1 2 3 --test_from test_suite/exp8_msd_ada --uno_time | tee tmp_log.txt
OUTPUT8=`cat tmp_log.txt | tail -n 3`
#TODO 9. Test for MERNet (multi-exit residual network)
python -u main_base_bak.py actnet RGB --arch mernet50 --num_segments 16 --npb --exp_header X --ada_reso_skip --policy_backbone mobilenet_v2 --reso_list 224 112 --backbone_list mernet50 --skip_list 1 2 4  --accuracy_weight 0.9 --efficency_weight 0.1 --folder_suffix _self --policy_input_offset 1 --mer --mer_indices_list 3 2 1 --uno_reso --uniform_loss_weight 3.0 --use_gflops_loss --init_tau 0.000001  --batch-size 48 -j 16 --gpus 0 1 2 3 --test_from test_suite/exp9_mer_ada --uno_time | tee tmp_log.txt
OUTPUT9=`cat tmp_log.txt | tail -n 3`

echo -e "\n\033[1;36mEXPECT   : 73.830 (exp0_res_ada)\033[0m"
echo $OUTPUT0
echo -e "\n\033[1;36mEXPECT   : 72.937 (exp1_res_scsampler)\033[0m"
echo $OUTPUT1
echo -e "\n\033[1;36mEXPECT   : 72.495 (exp2_res_uniform)\033[0m"
echo $OUTPUT2
echo -e "\n\033[1;36mEXPECT   : 71.176 (exp3_res_lstm)\033[0m"
echo $OUTPUT3
echo -e "\n\033[1;36mEXPECT   : 65.657 (exp4_res_rand)\033[0m"
echo $OUTPUT4
echo -e "\n\033[1;36mEXPECT   : 73.454 (exp5_res_all)\033[0m"
echo $OUTPUT5
echo -e "\n\033[1;36mEXPECT   : 79.697 (exp6_eff_ada)\033[0m"
echo $OUTPUT6

echo -e "\n\033[1;36mEXPECT   : 71.099 (exp7_dmy_ada)\033[0m"
echo $OUTPUT7
echo -e "\n\033[1;36mEXPECT   : 68.387 (exp8_msd_ada)\033[0m"
echo $OUTPUT8
echo -e "\n\033[1;36mEXPECT   : 72.329 (exp9_mer_ada)\033[0m"
echo $OUTPUT9