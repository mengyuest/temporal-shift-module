import numpy as np
import pickle
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerLine2D

data_path="/Users/meng/Downloads/gate-stat-val.pkl"
with open(data_path, "rb") as f:
    data = pickle.load(f)

#TODO data should have L arrays, with each array in shape (N*T*C) valued from {0,1,2}
# we need 2*3*L stats
# 2: average / always on
# 3: three options
# L: layers

mean_stats=np.zeros((len(data),3))
always_stats=np.zeros((len(data),3))
for l in range(len(data)):
    N = data[l].shape[0]
    T = data[l].shape[1]
    num_channels = data[l].shape[-1]
    for i in range(3):
        mean_stats[l, i] = np.sum(data[l][: ,:, :]==i) / num_channels / T / N
        # always_stats[l, i] = np.sum(np.prod((data[l]==i).reshape((N*T, num_channels)), axis=0)) / num_channels
        always_stats[l, i] = np.sum(np.prod((data[l] == i).reshape((N, T, num_channels)), axis=1)) / num_channels / N

print(mean_stats)
print(always_stats)

mean_stats = mean_stats
always_stats = always_stats

label_font_size = 13
legend_font_size = 13
tick_font_size = 10
bar_rel_width=0.9
legend_text_wspace=0.1
legend_line_width=0.01

from pylab import rcParams
rcParams['figure.figsize'] = 6, 5
from matplotlib.ticker import FormatStrFormatter
import matplotlib.ticker as ticker
# now plot
f, axs = plt.subplots(3, sharex=True)

for i in range(3):
    axs[i].tick_params(labelsize=tick_font_size)
    axs[i].set_ylabel("Ratio", fontsize=label_font_size)
    axs[i].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
axs[2].set_xlabel("Bottleneck Block Index", fontsize=label_font_size)

ind=np.arange(len(data))

axs[1].xaxis.set_major_locator(ticker.FixedLocator(ind[::2]))

placeholder = matplotlib.lines.Line2D([],[],linestyle='')


red = np.array([[250., 124, 112], [249,0,0], [202,0,0]]) / 255.
orange = np.array([[252.,203,105],[251,170,0],[207,138,0]]) / 255.
blue = np.array([[93.,148,211], [18,113,196], [9,83,149]]) / 255.

colors_list = [ red[2], red[0], orange[2], orange[0], blue[2], blue[0]]
ls=[]
for i in range(3):
    l1 = axs[i].bar(ind, always_stats[:,i], color=colors_list[i*2+0], width=bar_rel_width)
    l2 = axs[i].bar(ind, mean_stats[:,i]-always_stats[:,i], bottom=always_stats[:,i], color=colors_list[i*2+1],width=bar_rel_width)
    ls.append(l2)
    ls.append(l1)
    z = np.polyfit(ind, mean_stats[:,i], 3)
    p = np.poly1d(z)
    axs[i].plot(ind, p(ind), "k--")

leg = f.legend(ls,["Skip","Skip\n(Instance)", "Reuse","Reuse\n(Instance)", "Keep", "Keep\n(Instance)"],
           loc="center",  # Position of legend
           bbox_to_anchor=[0.85, 0.52], # x-offset, y-offset
           labelspacing=2.5, fontsize = legend_font_size, handletextpad=legend_text_wspace)
# for legobj in leg.legendHandles:
#     legobj.set_linewidth(legend_line_width)

f.tight_layout()
# plt.subplots_adjust(right=0.71) #0.72 for Skip(Always)
# plt.subplots_adjust(right=0.78)  # for figsize (10, 5)
# plt.subplots_adjust(right=0.75)  # for figsize (8, 5)
plt.subplots_adjust(right=0.72)  # for figsize (6, 5)
# plt.show()
plt.savefig("fig4_bar_v.png", bbox_inches='tight')