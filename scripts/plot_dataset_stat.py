import numpy as np
import pickle
import matplotlib
import matplotlib.pyplot as plt
# from matplotlib import rc
# rc('text', usetex=True)

# TODO: skip, reuse, keep
#TODO can we also have ada-TSM?
usage_d={
    "Something-V1": [0.3701, 0.2577, 0.3721],
    "Something-V2": [0.5963, 0.1716, 0.2322],
    "Jester": [0.7122, 0.1049, 0.1829],
    "Mini-Kinetics": [0.3484, 0.1235, 0.5331],
}

labels=["Skip", "Reuse", "Keep"]

is_pie=True
# from pylab import rcParams
# rcParams['figure.figsize'] = 5, 5

label_font_size = 15
legend_font_size = 15
tick_font_size = 15
bar_rel_width=0.9

red = np.array([[250., 124, 112], [249,0,0], [202,0,0]]) / 255.
orange = np.array([[252.,203,105],[251,170,0],[207,138,0]]) / 255.
blue = np.array([[93.,148,211], [18,113,196], [9,83,149]]) / 255.

# now plot
for i,key in enumerate(usage_d):
    ax = plt.subplot(2, 2, i+1)
    ax.pie(usage_d[key], labels=labels, colors=[red[0], orange[0], blue[0]], autopct='%1.1f%%',startangle=90)
    ax.axis('equal')
    ax.set_title(key, weight="bold", y=0.92)

# plt.show()
plt.subplots_adjust(hspace=0.12)
plt.subplots_adjust(wspace=-0.2)
plt.savefig("fig3_pie_v.png", bbox_inches='tight')