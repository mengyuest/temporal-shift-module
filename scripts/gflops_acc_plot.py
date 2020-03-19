import matplotlib
import matplotlib.pyplot as plt

matplotlib.rc('xtick', labelsize=25)
matplotlib.rc('ytick', labelsize=25)
matplotlib.rc('axes', labelsize=25)
matplotlib.rc('legend', fontsize= 15)
matplotlib.rc('legend', handlelength=2)
matplotlib.rcParams['axes.spines.right'] = False
matplotlib.rcParams['axes.spines.top'] = False

res_list=[]

color_list=['b', 'g', 'c', 'm',   'y', 'r'] * 2
shape_list=['.', 'v',  "+","D", "x", "*"] * 2
line_list=[':', '-.',  ":","-.", "--", "-."] * 2

# res_list.append(["LSTM", [(0.4825,50.71), (0.965, 61.76), (1.93,69.31), (3.86, 72.9), (3.86/16*25, 74.04)], 16])
# # res_list.append(["SCSampler", [(0.24, 63.5), (0.48, 69.8), (0.96,72.6), (1.45, 73.2), (1.93, 73.7), (2.82, 73.7)], 16])
# res_list.append(["FrameGlimpse(CVPR2016)",[(1.32, 60.2)],25])
# res_list.append(["LiteEval(NeurIPS2019)", [(3.80, 72.7)], 25])
# res_list.append(["AdaFrame(CVPR2019)", [(15.11/25,56.24),(17.85/25,61.14),(26.44/25,64.04), (44.98/25,68.07), (53.56/25,69.10), (78.97/25, 71.5)], 25])
# res_list.append(["ListenToLook(Arxiv2019)",[(26.60/16, 69.47),(42.24/16,70.66),(58.67/16,71.47),(5.09, 72.3)],16])
# # res_list.append(["Ours", [(2.00, 73.1), (2.91, 74.6)], 16])
# res_list.append(["Ours", [(1.85/16*8, 68.29), (1.75, 73.4), (1.75/16*25, 74.8)], 16])

# flops_per_video=True
#
# for i,pair in enumerate(res_list):
#     factor = pair[2] if flops_per_video else 1
#     line, = plt.plot([x[0]*factor for x in pair[1]],[x[1] for x in pair[1]], label=pair[0], linestyle=line_list[i],
#                      color=color_list[i], marker=shape_list[i], markersize=15)





#TODO

# res_list.append(["Uniform",
#                  [(16.4412, 62.849), (32.8824, 69.327), (65.7648, 72.49), (102.7575, 73.321),
#                   (131.5296, 74.05), (197.2944, 74.064), (263.0592, 74.066)]])

# res_list.append(["LSTM", 'b', 'P', '-',
#                  [(16.4732, 60.55), (32.9464, 66.935), (65.8928, 71.206), (102.9575, 72.589),
#                   (131.7856, 73.001), (197.6784,73.596), (263.5712,73.785)]])

#TODO this one is tricky
res_list.append(["SCSampler", 'y', 'o', ':',
                 [
                     #(4.5335, 53.052),
                  (16.8644,67.83), (33.7288, 72.443),
                     # (41.9494, 72.851),
                     (51.0164, 73.945)]])

res_list.append(["MultiAgent", 'c', 'X', ':',
                 [
                  # (650.8475, 74.1390), (493.6114, 73.6757),
                  (308.9961,72.9035), (201.8253,71.0888), (163.4941,70.4710), (125.1630,69.1197),
                     (86.8318, 65.5290)
                 ]])

res_list.append(["LiteEval",'g', 's', '-',
                 [(95.1, 72.7)]])

res_list.append(["AdaFrame-10", 'k', 'd', '--',
                 [(79.7914, 71.5907), (53.9765, 69.1969), (45.3716, 68.1931)]])

res_list.append(["AdaFrame-5",'k', 'v', '--',
                 [(34.4198, 69.5830), (32.0730, 68.6950), (24.2503, 66.0309)]])

# res_list.append(["ListenToLook(Image-Audio | Image-Audio)", 'm', '*', '-.',
#                  [(112.6467, 76.6100), (75.0978, 76.3784), (56.3233, 76.1853), (45.3716, 76.0695),
#                   (37.5489, 75.6448), (32.0730, 75.3745), (22.6858, 74.7568), (11.7340,71.6680)]])

res_list.append(["ListenToLook(Image-Audio|ResNet-101)", 'm', 'd', '-.',
                 [(168.9700, 74.7568), (145.5020, 74.1776), (130.6389, 73.4826), (114.2112,72.4402)]])

res_list.append(["ListenToLook(MobileNetv2|ResNet-101)", 'm', 'h', '-.',
                 [(81.3559, 72.2857),(58.6701,71.4749), (42.2425, 70.6641), (26.5971,69.4672)]])

res_list.append(["Ours(ResNet)", 'r', '*', '-.',
                 [
                     # (9.03387, 60.278),
                  (17.2881,69.247), (33.44,73.8), (51.5985,74.858),
                  # (65.49834,75.17),
                     (97.76952, 76.059), (129.8723, 76.079)]])

res_list.append(["Uniform(EffNet)", 'b', 'x', '-.',
                 [(7.2, 70.953),(14.4,76.796), (28.8, 78.764), (45,79.663), (57.6,79.899),
                  (86.4,79.981), (115.2,80.216)]])

res_list.append(["LSTM(EffNet)", 'b', 'h', '-.',
                 [(7.24, 69.396),(14.48,75.15), (28.96, 78.06), (45.25,78.899), (57.92,79.121),
                  (86.88,79.405), (115.84,79.542)]])

# res_list.append(["Ours(EffNet)", 'r', '*', '-.',
#                  [(4.0508, 70.981), (7.86899,77.314), (15.84,79.7), (23.67925, 80.802),
#                   (30.19792,81.174), (45.08146,81.399), (60.00557, 81.548)]])

for i,pair in enumerate(res_list):
    c, m, l = pair[1],pair[2],pair[3]
    line, = plt.plot([x[0] for x in pair[4]],[x[1] for x in pair[4]], label=pair[0], linestyle=l,
                     color=c, marker=m, markersize=15)


plt.grid(True)

plt.xlabel("GFLOPS/video")
plt.ylabel("mAP(%)")
plt.legend()
plt.show()

