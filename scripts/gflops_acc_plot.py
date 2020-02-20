import matplotlib
import matplotlib.pyplot as plt

matplotlib.rc('xtick', labelsize=15)
matplotlib.rc('ytick', labelsize=15)
matplotlib.rc('axes', labelsize=15)
matplotlib.rc('legend', fontsize= 12)
matplotlib.rc('legend', handlelength=2)
matplotlib.rcParams['axes.spines.right'] = False
matplotlib.rcParams['axes.spines.top'] = False

res_list=[]

color_list=['b', 'g', 'c', 'm',   'y', 'r']
shape_list=['.', 'v',  "+","D", "x", "*"]
line_list=[':', '-.',  ":","-.", "--", "-."]

res_list.append(["LSTM", [(0.4825,50.71), (0.965, 61.76), (1.93,69.31), (3.86, 72.9), (3.86/16*25, 74.04)], 16])

# res_list.append(["SCSampler", [(0.24, 63.5), (0.48, 69.8), (0.96,72.6), (1.45, 73.2), (1.93, 73.7), (2.82, 73.7)], 16])
res_list.append(["FrameGlimpse(CVPR2016)",[(1.32, 60.2)],25])
res_list.append(["LiteEval(NeurIPS2019)", [(3.80, 72.7)], 25])
res_list.append(["AdaFrame(CVPR2019)", [(15.11/25,56.24),(17.85/25,61.14),(26.44/25,64.04), (44.98/25,68.07), (53.56/25,69.10), (78.97/25, 71.5)], 25])
res_list.append(["ListenToLook(Arxiv2019)",[(26.60/16, 69.47),(42.24/16,70.66),(58.67/16,71.47),(5.09, 72.3)],16])
# res_list.append(["Ours", [(2.00, 73.1), (2.91, 74.6)], 16])
res_list.append(["Ours", [(1.85/16*8, 68.29), (1.75, 73.4), (1.75/16*25, 74.8)], 16])

flops_per_video=True

for i,pair in enumerate(res_list):
    factor = pair[2] if flops_per_video else 1
    line, = plt.plot([x[0]*factor for x in pair[1]],[x[1] for x in pair[1]], label=pair[0], linestyle=line_list[i],
                     color=color_list[i], marker=shape_list[i], markersize=15)


plt.xlabel("GFLOPS/video" if flops_per_video else "GFLOPS/frame")
plt.ylabel("mAP")
plt.legend()
plt.show()

