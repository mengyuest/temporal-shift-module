import numpy as np
import pickle
import matplotlib
import matplotlib.pyplot as plt

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

label_font_size = 15
legend_font_size = 15
tick_font_size = 15
bar_rel_width=0.9
# now plot
f, axs = plt.subplots(3, sharex=True)

plt.tick_params(axis='both', which='minor', labelsize=tick_font_size)

axs[0].set_ylabel("Ratio", fontsize=label_font_size)
axs[1].set_ylabel("Ratio", fontsize=label_font_size)
axs[2].set_ylabel("Ratio", fontsize=label_font_size)
axs[2].set_xlabel("Bottleneck Block Index", fontsize=label_font_size)

ind=np.arange(len(data))

placeholder = matplotlib.lines.Line2D([],[],linestyle='')


red = np.array([[250., 124, 112], [249,0,0], [202,0,0]]) / 255.
orange = np.array([[252.,203,105],[251,170,0],[207,138,0]]) / 255.
blue = np.array([[93.,148,211], [18,113,196], [9,83,149]]) / 255.




# colors_list=["red", "coral",  "green", "lightgreen",  "blue", "lightblue"]
colors_list = [ red[2], red[0], orange[2], orange[0], blue[2], blue[0]]
ls=[]
for i in range(3):
    l1 = axs[i].bar(ind, always_stats[:,i], color=colors_list[i*2+0], width=bar_rel_width)
    l2 = axs[i].bar(ind, mean_stats[:,i]-always_stats[:,i], bottom=always_stats[:,i], color=colors_list[i*2+1],width=bar_rel_width)
    ls.append(l2)
    ls.append(l1)
    # axs[i].set_ylim(0, max(mean_stats[:,i]))
    z = np.polyfit(ind, mean_stats[:,i], 3)
    p = np.poly1d(z)
    axs[i].plot(ind, p(ind), "k--")

# ls = [ls[0],ls[1],placeholder,placeholder,ls[2],ls[3],placeholder,placeholder,ls[4],ls[5]]

f.legend(ls,["Skip","Skip\n(Always)", "Reuse","Reuse\n(Always)", "Keep", "Keep\n(Always)"],
           loc="center",  # Position of legend
           bbox_to_anchor=[0.85, 0.54], # x-offset, y-offset
           # borderaxespad=0.1, # Small spacing around legend box
           labelspacing=2, fontsize = legend_font_size)

f.tight_layout()
plt.subplots_adjust(right=0.75)
plt.show()