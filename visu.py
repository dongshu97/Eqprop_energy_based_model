import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import pylab as P


def plot_output(x, out_nb, ylabel, path, prefix):
    fig, ax = plt.subplots()
    label_name = range(out_nb)
    for out in label_name:
        ax.plot(x[:, out].numpy(), label=f'Output{out}')
    ax.set_xlabel('Examples')
    ax.set_ylabel(ylabel)
    ax.set_title('The evolution of ' + ylabel + ' with respect to the input examples')
    ax.legend(label_name, loc='upper right', bbox_to_anchor=(1.12, 1.15), ncol=1, fontsize=6, frameon=False)
    plt.savefig(str(path) + prefix + ylabel + '.png', format='png', dpi=300)


def plot_distribution(x, out_nb, file_name, path, prefix):
    fig = plt.figure()
    colors = cm.rainbow(np.linspace(0, 1, out_nb))
    n, bins, patches = P.hist(x.numpy().transpose(), 10, density=1, histtype='bar',
                              color=colors, label=range(out_nb), stacked=True)
    plt.legend(loc='upper right', bbox_to_anchor=(1.12, 1.15), ncol=1, fontsize=6, frameon=False)
    fig.savefig(str(path) + prefix + file_name + '.png', format='png', dpi=300)


def plot_imshow(x, out_nb, display, imShape, figName, path, prefix):
    fig, axes = plt.subplots(display[0], display[1])
    for i, ax in zip(range(out_nb), axes.flat):
        plot = ax.imshow(x[i, :].reshape(imShape[0], imShape[1]), cmap=cm.coolwarm)
        #ax.axis('off')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    fig.subplots_adjust(hspace=0.4, wspace=0.3)
    cax = plt.axes([0.92, 0.1, 0.02, 0.8])
    cb = fig.colorbar(plot, cax=cax)
    cb.ax.tick_params(labelsize=8)
    fig.suptitle('Imshow of ' + figName + ' neurons', fontsize=10)
    plt.savefig(str(path) + prefix + figName + '.png', format='png', dpi=300)



