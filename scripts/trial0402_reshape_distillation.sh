#TODO-1 (224->168->112)
#TODO 10 epo
#TODO(done)
python main.py actnet RGB --arch dmynet50 --num_segments 16 --lr 0.02 --epochs 10 --lr_steps 20 40 --npb --folder_suffix _self --rescale_to 224 --dmy --num_filters_list 64 48 32 --default_signal 0 --batch-size 48 -j 72 --gpus 0 1 2 3 --exp_header act_dres3_chain0_224_0_e10_b48_lcs --last_conv_same

#TODO 50 epo
#TODO(done)
python main.py actnet RGB --arch dmynet50 --num_segments 16 --lr 0.02 --epochs 50 --lr_steps 20 40 --npb --folder_suffix _self --rescale_to 224 --dmy --num_filters_list 64 48 32 --default_signal 0 --batch-size 48 -j 72 --gpus 0 1 2 3 --exp_header act_dres3_chain0_224_0_e50_b48_lcs --last_conv_same


#TODO-2 distillation (distill_policy, by_pred, two-step training, scratch/10/50)
python main.py actnet RGB --arch dmynet50 --num_segments 16 --lr 0.001 --epochs 50 --lr_steps 50 100 --batch-size 48 -j 72 --npb --folder_suffix _self --ada_reso_skip --policy_backbone mobilenet_v2 --reso_list 224 168 112 --backbone_list dmynet50 --skip_list 1 2 4 --policy_input_offset 2 --dmy --num_filters_list 64 48 32  --accuracy_weight 1.0 --efficency_weight 0.0 --uniform_loss_weight 0.0  --use_gflops_loss --exp_decay --init_tau 5 --exp_header act_ada_a.25e.0d.75_ed5_scratch_lr.001_lcs_dst_debug --gpus 0 1 2 3  --distill_policy --distillation_weight 0.25 --last_conv_same


#TODO-3 distillation (distill_policy, not shared, by_pred, scratch/10/50)
#      TODO-sub-1 scratch, distill policy
python main.py actnet RGB --arch dmynet50 --num_segments 16 --lr 0.001 --epochs 50 --lr_steps 50 100 --batch-size 48 -j 72 --npb --folder_suffix _self --ada_reso_skip --policy_backbone mobilenet_v2 --reso_list 224 168 112 --backbone_list dmynet50 --skip_list 1 2 4 --policy_input_offset 2 --dmy --num_filters_list 64 48 32  --accuracy_weight 1.0 --efficency_weight 0.0 --uniform_loss_weight 0.0  --use_gflops_loss --exp_decay --init_tau 5 --exp_header act_ada_a.75e.0d.25_ed5_scratch_lr.001_dst --gpus 0 1 2 3  --distill_policy --distillation_weight 0.25

# --accuracy_weight 0.9 --efficency_weight 0.1 --uniform_loss_weight 3.0
#      TODO-sub-2-1 load, adaptive policy (not using distill anymore)
python main.py actnet RGB --arch dmynet50 --num_segments 16 --lr 0.001 --epochs 50 --lr_steps 50 100 --batch-size 48 -j 72 --npb --folder_suffix _self --ada_reso_skip --policy_backbone mobilenet_v2 --reso_list 224 168 112 --backbone_list dmynet50 --skip_list 1 2 4 --policy_input_offset 2 --dmy --num_filters_list 64 48 32  --accuracy_weight 0.9 --efficency_weight 0.1 --uniform_loss_weight 3.0  --use_gflops_loss --exp_decay --init_tau 5 --exp_header act_ada_a.9e.1d.0u3_ed5_joiscr_lr.001 --gpus 0 1 2 3  --base_pretrained_from g0402-235616_act_ada_a.75e.0d.25_ed5_scratch_lr.001_dst/imta_models/ckpt.best.pth.tar


#      TODO-sub-2-2 load, adaptive policy (still using distill)
python main.py actnet RGB --arch dmynet50 --num_segments 16 --lr 0.001 --epochs 50 --lr_steps 50 100 --batch-size 48 -j 72 --npb --folder_suffix _self --ada_reso_skip --policy_backbone mobilenet_v2 --reso_list 224 168 112 --backbone_list dmynet50 --skip_list 1 2 4 --policy_input_offset 2 --dmy --num_filters_list 64 48 32  --accuracy_weight 0.9 --efficency_weight 0.1 --uniform_loss_weight 3.0  --use_gflops_loss --exp_decay --init_tau 5 --exp_header act_ada_a.9e.1d.25u3_ed5_joiscr_lr.001_distill --gpus 0 1 2 3  --base_pretrained_from g0402-235616_act_ada_a.75e.0d.25_ed5_scratch_lr.001_dst/imta_models/ckpt.best.pth.tar --distillation_weight 0.25


#TODO load from prev imta_models, distill policy
#     TODO-sub-1 load from prev imta_models, distill policy
python main.py actnet RGB --arch dmynet50 --num_segments 16 --lr 0.001 --epochs 50 --lr_steps 50 100 --batch-size 48 -j 72 --npb --folder_suffix _self --ada_reso_skip --policy_backbone mobilenet_v2 --reso_list 224 168 112 --backbone_list dmynet50 --skip_list 1 2 4 --policy_input_offset 2 --dmy --num_filters_list 64 48 32  --accuracy_weight 1.0 --efficency_weight 0.0 --uniform_loss_weight 0.0  --use_gflops_loss --exp_decay --init_tau 5 --exp_header act_ada_a.75e.0d.25_ed5_cft10_lr.001_dst --gpus 0 1 2 3  --distill_policy --distillation_weight 0.25 --model_paths g0329-073458_act_dres3_chain0_224_0_e10_b48/imta_models/ckpt.best.pth.tar g0329-082233_act_dres3_chain1_168_1_e10_b48_pre224/imta_models/ckpt.best.pth.tar g0329-085151_act_dres3_chain2_112_2_e10_b48_pre168/imta_models/ckpt.best.pth.tar

##      TODO-sub-2 load, adaptive policy
#python main.py actnet RGB --arch dmynet50 --num_segments 16 --lr 0.001 --epochs 50 --lr_steps 50 100 --batch-size 48 -j 72 --npb --folder_suffix _self --ada_reso_skip --policy_backbone mobilenet_v2 --reso_list 224 168 112 --backbone_list dmynet50 --skip_list 1 2 4 --policy_input_offset 2 --dmy --num_filters_list 64 48 32  --accuracy_weight 1.0 --efficency_weight 0.0 --uniform_loss_weight 0.0  --use_gflops_loss --exp_decay --init_tau 5 --exp_header act_ada_a.75e.0d.25_ed5_scratch_lr.001_dst_debug --gpus 0 1 2 3  --distill_policy --distillation_weight 0.25


#TODO try less num filters [64, 32, 16]
python main.py actnet RGB --arch dmynet50 --num_segments 16 --lr 0.001 --epochs 50 --lr_steps 50 100 --batch-size 48 -j 72 --npb --folder_suffix _self --ada_reso_skip --policy_backbone mobilenet_v2 --reso_list 224 168 112 --backbone_list dmynet50 --skip_list 1 2 4 --policy_input_offset 2 --dmy --num_filters_list 64 32 16  --accuracy_weight 1.0 --efficency_weight 0.0 --uniform_loss_weight 0.0  --use_gflops_loss --exp_decay --init_tau 5 --exp_header act_ada631_a.1e.0d.25_ed5_scratch_lr.001_dst --gpus 0 1 2 3  --distill_policy --distillation_weight 0.25

#TODO joint_from_scratch
python main.py actnet RGB --arch dmynet50 --num_segments 16 --lr 0.001 --epochs 50 --lr_steps 50 100 --batch-size 48 -j 72 --npb --folder_suffix _self --ada_reso_skip --policy_backbone mobilenet_v2 --reso_list 224 168 112 --backbone_list dmynet50 --skip_list 1 2 4 --policy_input_offset 2 --dmy --num_filters_list 64 32 16  --accuracy_weight 0.9 --efficency_weight 0.1 --uniform_loss_weight 3.0  --use_gflops_loss --exp_decay --init_tau 5 --exp_header act_ada631_a.9e.1d.25u3_ed5_adadist_lr.001_dst --gpus 0 1 2 3  --distillation_weight 0.25

#TODO distill_from_scratch (comparison group)
python main.py actnet RGB --arch dmynet50 --num_segments 16 --lr 0.001 --epochs 50 --lr_steps 50 100 --batch-size 48 -j 72 --npb --folder_suffix _self --ada_reso_skip --policy_backbone mobilenet_v2 --reso_list 224 168 112 --backbone_list dmynet50 --skip_list 1 2 4 --policy_input_offset 2 --dmy --num_filters_list 64 32 16  --accuracy_weight 1.0 --efficency_weight 0.0 --uniform_loss_weight 0.0  --use_gflops_loss --exp_decay --init_tau 5 --exp_header act_ada631_a1e0d.25_ed5_dist_lr.001_dst --gpus 0 1 2 3  --distill_policy --distillation_weight 0.25




##TODO WHAT WE CAN TRY 0.02 + 0.001

#TODO use chain finetuned 10 one
python main.py actnet RGB --arch dmynet50 --num_segments 16 --lr 0.001 --epochs 50 --lr_steps 50 100 --batch-size 48 -j 72 --npb --folder_suffix _self --ada_reso_skip --policy_backbone mobilenet_v2 --reso_list 224 168 112 --backbone_list dmynet50 --skip_list 1 2 4 --policy_input_offset 2 --dmy --num_filters_list 64 48 32  --accuracy_weight 0.9 --efficency_weight 0.1 --uniform_loss_weight 3.0  --use_gflops_loss --exp_decay --init_tau 5 --exp_header act_ada_a.9e.1d.0u3_ed5_dist_from_chain10_lr.001 --gpus 0 1 2 3  --base_pretrained_from g0402-235622_act_ada_a.75e.0d.25_ed5_cft10_lr.001_dst/imta_models/ckpt.best.pth.tar


#TODO use chain finetuned 10 one (fixed backbones)
python main.py actnet RGB --arch dmynet50 --num_segments 16 --lr 0.001 --epochs 50 --lr_steps 50 100 --batch-size 48 -j 72 --npb --folder_suffix _self --ada_reso_skip --policy_backbone mobilenet_v2 --reso_list 224 168 112 --backbone_list dmynet50 --skip_list 1 2 4 --policy_input_offset 2 --dmy --num_filters_list 64 48 32  --accuracy_weight 0.9 --efficency_weight 0.1 --uniform_loss_weight 3.0  --use_gflops_loss --exp_decay --init_tau 5 --exp_header act_ada_a.9e.1d.0u3_ed5_dist_from_chain10_lr.001_fixbb --gpus 0 1 2 3  --base_pretrained_from g0402-235622_act_ada_a.75e.0d.25_ed5_cft10_lr.001_dst/imta_models/ckpt.best.pth.tar --freeze_backbone


#TODO old chain series (channel 64, 48, 32)
  #TODO-1 from chain-> first distill-> then adaptive
    #TODO-1-0 first distill (distill_weight=0.25)
    python main.py actnet RGB --arch dmynet50 --num_segments 16 --lr 0.02 --epochs 50 --lr_steps 20 40 --batch-size 48 -j 72 --npb --folder_suffix _self --ada_reso_skip --policy_backbone mobilenet_v2 --reso_list 224 168 112 --backbone_list dmynet50 --skip_list 1 2 4 --policy_input_offset 2 --dmy --num_filters_list 64 48 32 --accuracy_weight 1.0 --efficency_weight 0.0 --uniform_loss_weight 0.0 --use_gflops_loss --exp_decay --init_tau 5 --exp_header act_ada_a1e0d.25_ed5_cft10_lr.02_dst --gpus 0 1 2 3 --distill_policy --distillation_weight 0.25 --model_paths g0329-073458_act_dres3_chain0_224_0_e10_b48/imta_models/ckpt.best.pth.tar g0329-082233_act_dres3_chain1_168_1_e10_b48_pre224/imta_models/ckpt.best.pth.tar g0329-085151_act_dres3_chain2_112_2_e10_b48_pre168/imta_models/ckpt.best.pth.tar

    #TODO-1-1 then adaptvie (lr=0.02)
    python main.py actnet RGB --arch dmynet50 --num_segments 16 --lr 0.02 --epochs 50 --lr_steps 20 40 --batch-size 48 -j 72 --npb --folder_suffix _self --ada_reso_skip --policy_backbone mobilenet_v2 --reso_list 224 168 112 --backbone_list dmynet50 --skip_list 1 2 4 --policy_input_offset 2 --dmy --num_filters_list 64 48 32 --accuracy_weight 0.9 --efficency_weight 0.1 --uniform_loss_weight 3.0 --use_gflops_loss --exp_decay --init_tau 5 --exp_header act_ada_a.9e.1d0u3_ed5_cft10_dst_ada_lr.02 --gpus 0 1 2 3 --base_pretrained_from  g0404-103459_act_ada_a1e0d.25_ed5_cft10_lr.02_dst/imta_models/ckpt.best.pth.tar

    #TODO-1-1' then adaptive (lr=0.001)
    python main.py actnet RGB --arch dmynet50 --num_segments 16 --lr 0.001 --epochs 50 --lr_steps 50 100 --batch-size 48 -j 72 --npb --folder_suffix _self --ada_reso_skip --policy_backbone mobilenet_v2 --reso_list 224 168 112 --backbone_list dmynet50 --skip_list 1 2 4 --policy_input_offset 2 --dmy --num_filters_list 64 48 32 --accuracy_weight 0.9 --efficency_weight 0.1 --uniform_loss_weight 3.0 --use_gflops_loss --exp_decay --init_tau 5 --exp_header act_ada_a.9e.1d0u3_ed5_cft10_dst_ada_lr.001 --gpus 0 1 2 3 --base_pretrained_from  g0404-103459_act_ada_a1e0d.25_ed5_cft10_lr.02_dst/imta_models/ckpt.best.pth.tar


    #TODO-1-1'' test (why mmAP drops from last loading)
    python main.py actnet RGB --arch dmynet50 --num_segments 16 --lr 0.00001 --epochs 3 --lr_steps 20 40 --batch-size 48 -j 72 --npb --folder_suffix _self --ada_reso_skip --policy_backbone mobilenet_v2 --reso_list 224 168 112 --backbone_list dmynet50 --skip_list 1 2 4 --policy_input_offset 2 --dmy --num_filters_list 64 48 32 --accuracy_weight 1.0 --efficency_weight 0.0 --uniform_loss_weight 0.0 --use_gflops_loss --exp_decay --init_tau 5 --exp_header act_ada_a.9e.1d0u3_ed5_cft10_dst_dst_lr.00001_epo3 --gpus 0 1 2 3 --distill_policy --distillation_weight 0.25 --base_pretrained_from  g0404-103459_act_ada_a1e0d.25_ed5_cft10_lr.02_dst/imta_models/ckpt.best.pth.tar

    #TODO-1-1''' test
    python main.py actnet RGB --arch dmynet50 --num_segments 16 --lr 0.00001 --epochs 3 --lr_steps 20 40 --batch-size 48 -j 72 --npb --folder_suffix _self --ada_reso_skip --policy_backbone mobilenet_v2 --reso_list 224 168 112 --backbone_list dmynet50 --skip_list 1 2 4 --policy_input_offset 2 --dmy --num_filters_list 64 48 32 --accuracy_weight 0.9 --efficency_weight 0.1 --uniform_loss_weight 3.0 --use_gflops_loss --exp_decay --init_tau 5 --exp_header act_ada_a.9e.1d0u3_ed5_cft10_dst_ada_lr.00001_epo3 --gpus 0 1 2 3 --base_pretrained_from  g0404-103459_act_ada_a1e0d.25_ed5_cft10_lr.02_dst/imta_models/ckpt.best.pth.tar



  #TODO-2 scratch, first distill, then adaptive
    #TODO-2-0 distill from scratch
    python main.py actnet RGB --arch dmynet50 --num_segments 16 --lr 0.02 --epochs 50 --lr_steps 20 40 --batch-size 48 -j 72 --npb --folder_suffix _self --ada_reso_skip --policy_backbone mobilenet_v2 --reso_list 224 168 112 --backbone_list dmynet50 --skip_list 1 2 4 --policy_input_offset 2 --dmy --num_filters_list 64 48 32 --accuracy_weight 1.0 --efficency_weight 0.0 --uniform_loss_weight 0.0 --use_gflops_loss --exp_decay --init_tau 5 --exp_header act_ada_a1e0d.67_ed5_scratch_lr.02_dst --gpus 0 1 2 3 --distill_policy --distillation_weight 0.67

    #TODO-2-1 adaptive(lr=0.02)
    python main.py actnet RGB --arch dmynet50 --num_segments 16 --lr 0.02 --epochs 50 --lr_steps 20 40 --batch-size 48 -j 72 --npb --folder_suffix _self --ada_reso_skip --policy_backbone mobilenet_v2 --reso_list 224 168 112 --backbone_list dmynet50 --skip_list 1 2 4 --policy_input_offset 2 --dmy --num_filters_list 64 48 32 --accuracy_weight 0.9 --efficency_weight 0.1 --uniform_loss_weight 3.0 --use_gflops_loss --exp_decay --init_tau 5 --exp_header act_ada_a.9e.1d0u3_ed5_scratch_dst_ada_lr.02 --gpus 0 1 2 3 --base_pretrained_from g0404-165906_act_ada_a1e0d.67_ed5_scratch_lr.02_dst/imta_models/ckpt.best.pth.tar

    #TODO-2-1' adaptive(lr=0.001)
    python main.py actnet RGB --arch dmynet50 --num_segments 16 --lr 0.001 --epochs 50 --lr_steps 50 100 --batch-size 48 -j 72 --npb --folder_suffix _self --ada_reso_skip --policy_backbone mobilenet_v2 --reso_list 224 168 112 --backbone_list dmynet50 --skip_list 1 2 4 --policy_input_offset 2 --dmy --num_filters_list 64 48 32 --accuracy_weight 0.9 --efficency_weight 0.1 --uniform_loss_weight 3.0 --use_gflops_loss --exp_decay --init_tau 5 --exp_header act_ada_a.9e.1d0u3_ed5_scratch_dst_ada_lr.001 --gpus 0 1 2 3 --base_pretrained_from g0404-165906_act_ada_a1e0d.67_ed5_scratch_lr.02_dst/imta_models/ckpt.best.pth.tar


  #TODO-3 from chains , joint distill and adaptive
  python main.py actnet RGB --arch dmynet50 --num_segments 16 --lr 0.02 --epochs 50 --lr_steps 20 40 --batch-size 48 -j 72 --npb --folder_suffix _self --ada_reso_skip --policy_backbone mobilenet_v2 --reso_list 224 168 112 --backbone_list dmynet50 --skip_list 1 2 4 --policy_input_offset 2 --dmy --num_filters_list 64 48 32 --accuracy_weight 0.9 --efficency_weight 0.1 --uniform_loss_weight 3.0 --use_gflops_loss --exp_decay --init_tau 5 --exp_header act_ada_a.9e.1d.67u3_ed5_chains_dstada_lr.02 --gpus 0 1 2 3 --distillation_weight 0.67 --model_paths g0329-073458_act_dres3_chain0_224_0_e10_b48/imta_models/ckpt.best.pth.tar g0329-082233_act_dres3_chain1_168_1_e10_b48_pre224/imta_models/ckpt.best.pth.tar g0329-085151_act_dres3_chain2_112_2_e10_b48_pre168/imta_models/ckpt.best.pth.tar

  #TODO-4 from scratch, joint distill and adaptive
  python main.py actnet RGB --arch dmynet50 --num_segments 16 --lr 0.02 --epochs 50 --lr_steps 20 40 --batch-size 48 -j 72 --npb --folder_suffix _self --ada_reso_skip --policy_backbone mobilenet_v2 --reso_list 224 168 112 --backbone_list dmynet50 --skip_list 1 2 4 --policy_input_offset 2 --dmy --num_filters_list 64 48 32 --accuracy_weight 0.9 --efficency_weight 0.1 --uniform_loss_weight 3.0 --use_gflops_loss --exp_decay --init_tau 5 --exp_header act_ada_a.9e.1d.67u3_ed5_scratch_dstada_lr.02 --gpus 0 1 2 3 --distillation_weight 0.67

#TODO new chain series (channel 64, 32, 16)
#TODO-1 (224->168->112)
#TODO 10 epo
#TODO(done)
python main.py actnet RGB --arch dmynet50 --num_segments 16 --lr 0.02 --epochs 10 --lr_steps 20 40 --npb --folder_suffix _self --rescale_to 224 --dmy --num_filters_list 64 32 16 --default_signal 0 --batch-size 48 -j 72 --gpus 0 1 2 3 --exp_header act_d631_chain0_224_0_e10_b48

#TODO(done)
python main.py actnet RGB --arch dmynet50 --num_segments 16 --lr 0.02 --epochs 10 --lr_steps 20 40 --npb --folder_suffix _self --rescale_to 168 --dmy --num_filters_list 64 32 16 --default_signal 1 --batch-size 48 -j 72 --gpus 0 1 2 3 --exp_header act_d631_chain1_168_1_e10_b48_pre224 --base_pretrained_from g0404-105431_act_d631_chain0_224_0_e10_b48/imta_models/ckpt.best.pth.tar --ignore_new_fc_weight

#TODO(done)
python main.py actnet RGB --arch dmynet50 --num_segments 16 --lr 0.02 --epochs 10 --lr_steps 20 40 --npb --folder_suffix _self --rescale_to 112 --dmy --num_filters_list 64 32 16 --default_signal 2 --batch-size 48 -j 72 --gpus 0 1 2 3 --exp_header act_d631_chain2_112_2_e10_b48_pre168 --base_pretrained_from g0404-123245_act_d631_chain1_168_1_e10_b48_pre224/imta_models/ckpt.best.pth.tar --ignore_new_fc_weight


    #TODO 1-0  (distill_policy)
python main.py actnet RGB --arch dmynet50 --num_segments 16 --lr 0.02 --epochs 50 --lr_steps 20 40 --batch-size 48 -j 72 --npb --folder_suffix _self --ada_reso_skip --policy_backbone mobilenet_v2 --reso_list 224 168 112 --backbone_list dmynet50 --skip_list 1 2 4 --policy_input_offset 2 --dmy --num_filters_list 64 32 16  --accuracy_weight 1.0 --efficency_weight 0.0 --uniform_loss_weight 0.0  --use_gflops_loss --exp_decay --init_tau 5 --exp_header act_ada631_a1e0d.25_ed5_cft10_lr.02_dst --gpus 0 1 2 3  --distill_policy --distillation_weight 0.25 --model_paths xxx/imta_models/ckpt.best.pth.tar yyy/imta_models/ckpt.best.pth.tar zzz/imta_models/ckpt.best.pth.tar

python main.py actnet RGB --arch dmynet50 --num_segments 16 --lr 0.02 --epochs 50 --lr_steps 20 40 --batch-size 48 -j 72 --npb --folder_suffix _self --ada_reso_skip --policy_backbone mobilenet_v2 --reso_list 224 168 112 --backbone_list dmynet50 --skip_list 1 2 4 --policy_input_offset 2 --dmy --num_filters_list 64 32 16  --accuracy_weight 1.0 --efficency_weight 0.0 --uniform_loss_weight 0.0  --use_gflops_loss --exp_decay --init_tau 5 --exp_header act_ada631_a1e0d.50_ed5_cft10_lr.02_dst --gpus 0 1 2 3  --distill_policy --distillation_weight 0.50 --model_paths xxx/imta_models/ckpt.best.pth.tar yyy/imta_models/ckpt.best.pth.tar zzz/imta_models/ckpt.best.pth.tar

python main.py actnet RGB --arch dmynet50 --num_segments 16 --lr 0.02 --epochs 50 --lr_steps 20 40 --batch-size 48 -j 72 --npb --folder_suffix _self --ada_reso_skip --policy_backbone mobilenet_v2 --reso_list 224 168 112 --backbone_list dmynet50 --skip_list 1 2 4 --policy_input_offset 2 --dmy --num_filters_list 64 32 16  --accuracy_weight 1.0 --efficency_weight 0.0 --uniform_loss_weight 0.0  --use_gflops_loss --exp_decay --init_tau 5 --exp_header act_ada631_a1e0d.75_ed5_cft10_lr.02_dst --gpus 0 1 2 3  --distill_policy --distillation_weight 0.75 --model_paths xxx/imta_models/ckpt.best.pth.tar yyy/imta_models/ckpt.best.pth.tar zzz/imta_models/ckpt.best.pth.tar

python main.py actnet RGB --arch dmynet50 --num_segments 16 --lr 0.02 --epochs 50 --lr_steps 20 40 --batch-size 48 -j 72 --npb --folder_suffix _self --ada_reso_skip --policy_backbone mobilenet_v2 --reso_list 224 168 112 --backbone_list dmynet50 --skip_list 1 2 4 --policy_input_offset 2 --dmy --num_filters_list 64 32 16  --accuracy_weight 1.0 --efficency_weight 0.0 --uniform_loss_weight 0.0  --use_gflops_loss --exp_decay --init_tau 5 --exp_header act_ada631_a1e0d.9999_ed5_cft10_lr.02_dst --gpus 0 1 2 3  --distill_policy --distillation_weight 0.9999 --model_paths g0404-105431_act_d631_chain0_224_0_e10_b48/imta_models/ckpt.best.pth.tar g0404-123245_act_d631_chain1_168_1_e10_b48_pre224/imta_models/ckpt.best.pth.tar g0404-130219_act_d631_chain2_112_2_e10_b48_pre168/imta_models/ckpt.best.pth.tar

    #TODO 1-1 (distill -> policy)


    #TODO 2-0 (joint policy)
python main.py actnet RGB --arch dmynet50 --num_segments 16 --lr 0.02 --epochs 50 --lr_steps 20 40 --batch-size 48 -j 72 --npb --folder_suffix _self --ada_reso_skip --policy_backbone mobilenet_v2 --reso_list 224 168 112 --backbone_list dmynet50 --skip_list 1 2 4 --policy_input_offset 2 --dmy --num_filters_list 64 32 16  --accuracy_weight 0.9 --efficency_weight 0.1 --uniform_loss_weight 3.0  --use_gflops_loss --exp_decay --init_tau 5 --exp_header act_ada631_a.9e.1d.25u3_ed5_cft10_lr.02_joindst --gpus 0 1 2 3  --distillation_weight 0.25 --model_paths xxx/imta_models/ckpt.best.pth.tar yyy/imta_models/ckpt.best.pth.tar zzz/imta_models/ckpt.best.pth.tar

python main.py actnet RGB --arch dmynet50 --num_segments 16 --lr 0.02 --epochs 50 --lr_steps 20 40 --batch-size 48 -j 72 --npb --folder_suffix _self --ada_reso_skip --policy_backbone mobilenet_v2 --reso_list 224 168 112 --backbone_list dmynet50 --skip_list 1 2 4 --policy_input_offset 2 --dmy --num_filters_list 64 32 16  --accuracy_weight 0.9 --efficency_weight 0.1 --uniform_loss_weight 3.0  --use_gflops_loss --exp_decay --init_tau 5 --exp_header act_ada631_a.9e.1d.50u3_ed5_cft10_lr.02_joindst --gpus 0 1 2 3  --distillation_weight 0.5 --model_paths xxx/imta_models/ckpt.best.pth.tar yyy/imta_models/ckpt.best.pth.tar zzz/imta_models/ckpt.best.pth.tar

python main.py actnet RGB --arch dmynet50 --num_segments 16 --lr 0.02 --epochs 50 --lr_steps 20 40 --batch-size 48 -j 72 --npb --folder_suffix _self --ada_reso_skip --policy_backbone mobilenet_v2 --reso_list 224 168 112 --backbone_list dmynet50 --skip_list 1 2 4 --policy_input_offset 2 --dmy --num_filters_list 64 32 16  --accuracy_weight 0.9 --efficency_weight 0.1 --uniform_loss_weight 3.0  --use_gflops_loss --exp_decay --init_tau 5 --exp_header act_ada631_a.9e.1d.75u3_ed5_cft10_lr.02_joindst --gpus 0 1 2 3  --distillation_weight 0.75 --model_paths xxx/imta_models/ckpt.best.pth.tar yyy/imta_models/ckpt.best.pth.tar zzz/imta_models/ckpt.best.pth.tar




#TODO MODEL-SEPARATE
python main.py actnet RGB --arch dmynet50 --num_segments 16 --lr 0.02 --epochs 50 --lr_steps 20 40 --batch-size 48 -j 72 --npb --folder_suffix _self --ada_reso_skip --policy_backbone mobilenet_v2 --reso_list 224 168 112 --backbone_list dmynet50 --skip_list 1 2 4 --policy_input_offset 2 --dmy --num_filters_list 64 48 32 --accuracy_weight 0.9 --efficency_weight 0.1 --uniform_loss_weight 3.0 --use_gflops_loss --exp_decay --init_tau 5 --exp_header act_ada_a.9e.1d0u3_ed5_scratch_dst_ada_lr.02_separate_debug --gpus 0 1 2 3 --base_pretrained_from g0404-165906_act_ada_a1e0d.67_ed5_scratch_lr.02_dst/imta_models/ckpt.best.pth.tar --separate_dmy