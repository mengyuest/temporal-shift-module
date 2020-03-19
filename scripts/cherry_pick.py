import os,sys,time
import numpy as np

import matplotlib.pyplot as plt
from PIL import Image
from os.path import join as ospj

def softmax(data):
    safe_data = data - np.max(data, axis=-1, keepdims=True)
    exp_data = np.exp(safe_data)
    exp_sum = np.sum(exp_data, axis=-1, keepdims=True)
    return exp_data / exp_sum

log_dir="/store/workspaces/meng/logs_tsm/aligned_models"
d_dir="/store/datasets"
cnt_threshold = 200

qualitative=False
num_sampled_classes=20

dataset_name={"actnet":"ActivityNet",
              "fcvid":"FCVID",
              "minik":"Mini-Kinetics"}


for dataset in ["fcvid"]: #["actnet", "fcvid", "minik"]:
    for option in ["easy", "mid", "hard"]:
        if dataset=="fcvid":
            meta_data_path = "../g0301-214927_fcvid16_res_3m124_a.95e.05_ed5_ft10ds_lr.001_gu3_ft.0005/meta-val-0313-121038.npy.npz" #"g0228-051614_fcvid16_res_3m124_a.9e.1_ed5_ft10ds_lr.001_gu3_lr.0005/meta-val-0304-162505.npy.npz"
            class_idx_path="fcvid/classInd.txt"
            img_path = "fcvid/frames_self"
            save_dir = "fcvid"
            prefix="image_%05d.jpg"
        elif dataset=="actnet":
            meta_data_path = "tmp_practice/meta-val-0304-145622.npy.npz"
            class_idx_path="activity-net-v1.3/classInd.txt"
            img_path = "activity-net-v1.3/frames_self"
            save_dir = "actnet"
            prefix = "image_%05d.jpg"
        else:
            meta_data_path = "../g0302-153742_minik16_res_3m124_a.9e.1_ed5_ft10ds_lr.001_gu3_lr.0005/meta-val-0311-140959.npy.npz"
            class_idx_path="kinetics-100/classIndMiniK.txt"
            img_path = "kinetics-qf"
            save_dir = "minik"
            prefix = "%05d.jpg"

        meta_data_full_path = ospj(log_dir, meta_data_path)
        meta_data=np.load(meta_data_full_path)
        img_full_path=ospj(d_dir, img_path)
        classes={int(line.split(",")[0]):line.split(",")[1].strip() for line in open(ospj(d_dir,class_idx_path)).readlines()}
        save_full_path = ospj("../vis", save_dir, option)
        os.makedirs(save_full_path, exist_ok=True)
        print("Now",save_full_path,"...")

        #rs=npa, names=npb, results=npc, targets=npd, all_preds=npe
        #TODO
        # rs:        (4926, 16, 7)
        # names:     (4926,)
        # results:   (4926, 200)
        # targets:   (4926, 3)
        # all_preds: (4926, 4, 16, 200)
        num_samples = meta_data["names"].shape[0]
        T = meta_data["rs"].shape[1]

        #TODO conditions
        # 1. must be correct
        # 2. maintains all reso and some skips
        # 3. High reso, pred is high
        # 4. Low res, original pred is similar
        # 5. Skip, original pred is very low
        rs = meta_data["rs"]
        names = meta_data["names"]
        results = softmax(meta_data["results"])
        targets = meta_data["targets"]
        all_preds = softmax(meta_data["all_preds"])
        indices = meta_data["indices"]

        if not qualitative:
            if option in ["mid","hard"]:
                continue
            #TODO compute overall distribution
            print("overall distribution")
            mean_rs = np.mean(rs,axis=(0,1))
            for mean_r in mean_rs:
                print(mean_r)

            print()

            print("class-level distribution")
            # TODO compute class distribution
            # 1. use high/all-reso ratio
            # 2. use reso/skip ratio
            num_classes=np.unique(targets).shape[0]-1 #TODO minus -1
            d_tmp = []
            for i in range(num_classes):
                # print(np.where(targets[:,0]==i)[0])
                # print(rs[np.where(targets[:, 0] == i)[0]])
                # print(rs[np.where(targets[:,0]==i)[0], :, :])
                d_tmp.append(np.sum(rs[np.where(targets[:,0]==i)[0], :, :], axis=(0,1))+
                             np.sum(rs[np.where(targets[:,1]==i)[0], :, :], axis=(0,1))+
                             np.sum(rs[np.where(targets[:,2]==i)[0], :, :], axis=(0,1)))

            #print("classes",num_classes, "\ndetails",d_tmp)
            hl_ratios=[(x[0]/(np.sum(x[:4]+1e-7)), classes[i]) for i,x in enumerate(d_tmp)]
            rs_ratios=[(np.sum(x[:4])/(np.sum(x)+1e-7), classes[i]) for i,x in enumerate(d_tmp)]
            hl_ratios=sorted(hl_ratios, key=lambda x:x[0])
            rs_ratios=sorted(rs_ratios, key=lambda x:x[0])

            #TODO plot
            plot_height=6
            plot_width=6
            save_full_path = ospj("../vis", "plots")
            os.makedirs(save_full_path, exist_ok=True)

            x=np.arange(mean_rs.shape[0]-1)
            fig, ax = plt.subplots(figsize=(plot_width,plot_height))
            ax.xaxis.set_tick_params(labelsize=8)
            plt.bar(x, list(100*mean_rs[:3])+list(100*mean_rs[4:]), width=0.8)
            plt.xticks(x, ('224', '168', '112', 'skip-1', 'skip-2', 'skip-4'))

            ax.xaxis.set_tick_params(labelsize=16)
            ax.yaxis.set_tick_params(labelsize=16)
            plt.ylabel("Policy percentage (%)", fontsize=16)
            plt.title("Overall policy distribution on %s"%(dataset_name[dataset]), fontsize=16)

            plt.savefig(ospj(save_full_path,dataset+"_overall.png"), bbox_inches = "tight")
            plt.close()




            x=np.arange(num_sampled_classes)
            indices=np.linspace(0, num_classes-1, num=num_sampled_classes).astype(np.int32)

            fig, ax=plt.subplots(figsize=(plot_width,plot_height))
            ax.xaxis.set_tick_params(labelsize=12)
            plt.bar(x, [hl_ratios[ii][0] for ii in indices])
            plt.xticks(x, tuple([hl_ratios[ii][1] for ii in indices]), rotation="vertical")

            # ax.xaxis.set_tick_params(labelsize=16)
            ax.yaxis.set_tick_params(labelsize=16)
            plt.ylabel("High resolution ratio in all resolutions", fontsize=16)
            plt.title("Relative high resolution usage on %s" % (dataset_name[dataset]), fontsize=16)
            # plt.tight_layout()
            plt.savefig(ospj(save_full_path, dataset + "_rel_high.png"), bbox_inches = "tight")
            plt.close()




            fig, ax = plt.subplots(figsize=(plot_width,plot_height))
            ax.xaxis.set_tick_params(labelsize=12)
            the_data=[rs_ratios[ii][0] for ii in indices]
            plt.bar(x, the_data)
            plt.ylim(min(the_data)-0.05, max(the_data)+0.05)
            plt.xticks(x, tuple([rs_ratios[ii][1] for ii in indices]), rotation="vertical")

            # ax.xaxis.set_tick_params(labelsize=16)
            ax.yaxis.set_tick_params(labelsize=16)
            plt.ylabel("Resolution usage ratio", fontsize=16)
            plt.title("Resolution usage on %s" % (dataset_name[dataset]), fontsize=16)
            # plt.tight_layout()
            plt.savefig(ospj(save_full_path, dataset + "_reso.png"), bbox_inches = "tight")
            plt.close()

        else:
            pred_threshold=0.3
            gap_threshold=0.1
            low_pred_threshold=0.2

            cnt=0

            for i in range(num_samples):
                name = names[i]
                label = targets[i,0]
                pred = np.argmax(results[i])
                if label!=pred: #TODO must be correct
                    continue

                choices=np.argmax(rs[i],axis=1)
                if option not in ["easy","mid","hard"]:
                    if 0 not in choices or 1 not in choices or 2 not in choices or 4 not in choices: #TODO include all
                        continue


                t_check=True
                counter=0
                for t in range(T):
                    if counter>0: #TODO skip mechanism
                        counter-=1
                        continue
                    if choices[t] in [4,5,6] and counter==0:
                        counter = choices[t]-1  #TODO skip mechanism
                        if all_preds[i,0,t,label]>low_pred_threshold: #TODO check "skip because low pred threshold"
                            t_check=False
                            break
                        continue
                    if choices[t]==0:
                        if all_preds[i,0,t,label]<pred_threshold: #TODO check "choose original must have high pred"
                            t_check=False
                            break
                    if choices[t] in [1,2,3]:
                        if all_preds[i,0,t,label] > all_preds[i,choices[t], t, label]+ gap_threshold: #TODO check "choose low reso must be not to bad than original"
                            t_check = False
                            break
                if not t_check:
                    continue

                n_orig=np.sum(choices==0)
                n_skip=np.sum(choices>=4)

                if option=="easy": #TODO 0~1 original, >=4 skip
                    if n_orig>1:
                        continue
                    if n_skip<4:
                        continue
                elif option=="mid": #TODO 2~3 original, 1~3 skip
                    if n_orig<2 or n_orig>3:
                        continue
                    if n_skip<1 or n_skip>3:
                        continue
                elif option=="hard": #TODO >=4 original
                    if n_orig <4:
                        continue
                else:
                    if np.sum(choices==0)>2:
                        continue
                    # if np.sum(choices==1)>2:
                    #     continue
                    if np.sum(choices >= 4) > 2:
                        continue
                print(name, label, results[i][label], choices, indices[i])#all_preds[i,:,:,label])

                canvas=np.ones((224 * 2, 224 * T, 3))*255
                for tt in range(T):
                    im = Image.open(("%s/%s/"+prefix)%(img_full_path, name, indices[i][tt]))
                    width, height = im.size
                    ratio=width/height
                    new_w = 224 if ratio < 1 else int(224 * ratio)
                    new_h = 224 if ratio > 1 else int(224 / ratio)
                    im1 = im.resize((new_w, new_h))
                    im2 = im1.crop(((new_w-224)/2, (new_h-224)/2, (new_w+224)/2, (new_h+224)/2))

                    shrink_d={0:0, 1:28, 2:56, 3:70}
                    size_d={0:224, 1:168, 2:112, 3:84}

                    canvas[:224, 224*tt:224*(tt+1)] = im2
                    if choices[tt] < 4:
                        d = shrink_d[choices[tt]]
                        im3 = im2.resize((size_d[choices[tt]], size_d[choices[tt]]))
                        canvas[224+d:448-d, 224*tt+d:224*(tt+1)-d] = im3

                plt.imsave("%s/%d-%s.png"%(save_full_path, cnt, classes[label]), canvas.astype('uint8'))
                plt.close()

                cnt+=1
                # print("rsync -ave ssh \'%s\' ~/Downloads/tmp_act_net/%s"%(
                #     " ".join(["%s/%s/image_%05d.jpg"%(pstr,name,indices[i][0])]+
                #              ["/gpfs/wscgpfs02/cvpr/datasets/activity-net-v1.3/frames_self/%s/image_%05d.jpg"%(name,x) for x in indices[i][1:]]),name))
                if cnt>cnt_threshold:
                    break