import random
import numpy as np
from matplotlib import pyplot as plt


def rand_color():
    return (random.random(), random.random(), random.random())


def plot_stats(metrics_path, names, size_per_row=15, size_per_col=50, ncols=3, fontsize=50, sampling_timestamp=1):
    val_accuracy, val_precision, val_recall, val_f_measure, val_tp, val_tn, val_fp, val_fn = [list()]*8
    train_batch_loss, train_batch_accuracy, train_batch_precision, train_batch_recall, train_batch_f_measure, \
    train_batch_tp, train_batch_tn, train_batch_fp, train_batch_fn = [list()]*9

    num_iters = 0
    metrics = list()

    with open(metrics_path, 'r') as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            if i%sampling_timestamp == 0:
                m = [float(val) for val in line.split(' ')]
                metrics.append(m)
                num_iters += 1

    bottom_lim = 0
    metrics = np.array(metrics).T.tolist()
    num_metrics = len(metrics)
    nrows = 1+num_metrics//ncols if num_metrics%ncols != 0 else num_metrics//ncols
    plt.rcParams["figure.figsize"] = (nrows*size_per_row, ncols*size_per_col)
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols) # create subplot table
    iters_axis = [j+1 for j in range(num_iters)]

    for i in range(nrows):
        for j in range(ncols):
            if i*ncols+j == num_metrics:
                plt.show()
                return

            n, m = names[i*ncols+j], metrics[i*ncols+j]
            ax[i][j].plot(iters_axis, m, color=rand_color())
            ax[i][j].set_xlabel('iteration', fontsize=fontsize)
            ax[i][j].set_ylabel(n, fontsize=fontsize)
            ytick_num = len(ax[i][j].get_yticklabels())
            xtick_num = len(ax[i][j].get_xticklabels())
            y_top_lim = max(metrics[i*ncols+j])
            x_top_lim = max(iters_axis)
            ax[i][j].set_ylim(bottom_lim, y_top_lim)
            ax[i][j].set_xlim(bottom_lim, x_top_lim)
            # ax[i][j].set_yticklabels([min(metrics[i*ncols+j])]+(ytick_num-2)*[""]+[y_top_lim])
            # ax[i][j].set_xticklabels([min(iters_axis)]+(xtick_num-2)*[""]+[x_top_lim])