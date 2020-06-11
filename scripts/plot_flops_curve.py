import numpy as np
import pickle
import matplotlib
import matplotlib.pyplot as plt
# from matplotlib import rc
# rc('text', usetex=True)
params = {'mathtext.default': 'regular' }
plt.rcParams.update(params)

# TODO SomethingV1
# {name: [#params, flops, acc, id]}
# AdaFuse
# AdaFuse-shift
# AdaFuse-acc
data={
    "TSN":[24.3, 33.2, 19.7, 0, 0, 0.8],
    "TSM":[24.3, 33.2, 45.7, 1, 4, -1],
    r"$TRN_{Multiscale}$": [18.3, 16.0, 34.4, 2, 0, 0.5],
    r'$TRN_{RGB+Flow}$':[36.6, 32.0, 42.0,2, 2, 0.6],
    "I3D":[28.0, 306, 41.6, 3, -60, 0.8],
    r"I3D+GCN+NL":[62.2, 606, 46.1, 3, -270, 1.2],
    "ECO":[47.5, 32, 39.6, 4, 2, 0.8],
    r"$ECO_{en}Lite$":[150, 267, 46.4, 4, -130, 1.6],
    r"$AdaFuse_{Inc}$":[14.5, 12.1, 38.5,5, -1, 0.7],
    r"$AdaFuse_{R50}$":[37.7, 22.1, 41.9, 5, -10, 0.88],
    r"$AdaFuse_{R50}^{TSM}$":[37.7, 19.1, 44.9, 5, -8, 0.85],
    r"$AdaFuse_{R50}^{TSM+Last}$":[39.1,31.3, 46.8, 5, -11, 1],
}

log_scale=True

label_font_size = 15
legend_font_size = 15
tick_font_size = 15

red = np.array([[250., 124, 112], [249,0,0], [202,0,0]]) / 255.
orange = np.array([[252.,203,105],[251,170,0],[207,138,0]]) / 255.
blue = np.array([[93.,148,211], [18,113,196], [9,83,149]]) / 255.

flops = [data[key][1] for key in data]
accs = [data[key][2] for key in data]
ss = [(data[key][0])**(1/1)*12 for key in data]  # TODO transform
labels = [key for key in data]

colors=[red[0], red[1], red[2], orange[0], orange[1], blue[0], blue[1], blue[2]]
rs = [data[key][3] for key in data]
cs = [colors[data[key][3]] for key in data]
xos=[data[key][4] for key in data]
yos=[data[key][5] for key in data]

from pylab import rcParams
rcParams['figure.figsize'] = 8, 8


if log_scale:
    plt.scatter(flops, accs, s=ss, c=cs)
    plt.xlabel("GLOPS", fontsize=label_font_size)
    plt.ylabel("Accuracy (%)", fontsize=label_font_size)
    plt.ylim(top=49,bottom=19)
    plt.xlim(right=980, left=10)
    plt.tick_params(labelsize=tick_font_size)
    for i, txt in enumerate(labels):
        # if rs[i]==5:
        #     if flops[i]>19:
        #         x_offset=-10
        #     else:
        #         x_offset=-2
        # else:
        #     x_offset=2
        # y_offset = 0.75 if rs[i] == 5 else -1
        x_offset=xos[i]
        y_offset=yos[i]
        plt.text(flops[i]+x_offset, accs[i]+y_offset, txt, fontsize=legend_font_size)
    plt.xscale("log")
    ax1=plt.gca()
    lam_beta=[10, 20, 40, 80, 160, 320, 640]
    ax1.set_xticks(lam_beta)
    ax1.set_xticklabels(lam_beta)
    ax1.set_xticks([], minor=True)
    ax1.set_xticklabels([], minor=True)
    # plt.show()
    plt.savefig("fig2_curve_v.png", bbox_inches='tight')
# else:
#     f, (ax, ax2) = plt.subplots(2, 1, sharex=True, sharey=True)
#     ax.scatter(flops, accs, s=ss, c=cs)
#     ax2.scatter(flops, accs, s=ss, c=cs)
#
#     ax.set_xlim(250, 606)  # outliers only
#     ax2.set_xlim(0, 60)  # most of the data
#
#     ax.set_ylim(18, 20)  # outliers only
#     ax2.set_ylim(32, 48)  # most of the data
#
#     ax.spines['bottom'].set_visible(False)
#     ax2.spines['top'].set_visible(False)
#     ax.xaxis.tick_top()
#     ax.tick_params(labeltop=False)  # don't put tick labels at the top
#     ax2.xaxis.tick_bottom()
#
#     d = .015  # how big to make the diagonal lines in axes coordinates
#     # arguments to pass to plot, just so we don't keep repeating them
#     kwargs = dict(transform=ax.transAxes, color='k', clip_on=False)
#     ax.plot((-d, +d), (-d, +d), **kwargs)  # top-left diagonal
#     ax.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # top-right diagonal
#
#     kwargs.update(transform=ax2.transAxes)  # switch to the bottom axes
#     ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal
#     ax2.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # bottom-right diagonal
#     plt.savefig("flops_curve.png", bbox_inches='tight')



