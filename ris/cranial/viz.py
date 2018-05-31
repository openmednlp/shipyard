from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import time


def tsne_transform_fit(x_w2v):
    print('doing tsne')
    tic = time.time()
    tsne = TSNE(n_components=2)
    x_tsne = tsne.fit_transform(x_w2v)
    print('Time taken:', time.time() - tic)

    return x_tsne


def tsne_plot(x_w2v):
    print('plotting')
    x_tsne = tsne_transform_fit(x_w2v)
    plt.scatter(x_tsne[:, 0], x_tsne[:, 1])
    plt.show()


def tsne_run(x_w2v):
    tsne_plot(x_w2v)


def box_plot(names, results, header, show_plot=True, persist=False):
    # boxplot algorithm comparison
    fig = plt.figure()
    fig.suptitle(header)
    ax = fig.add_subplot(111)
    plt.boxplot(results)
    ax.set_xticklabels(names)
    plt.xlabel('Algorithms') #TODO: not generalized name
    plt.xticks(rotation=45)
    plt.ylim(0.6, 1.01)
    plt.tight_layout()

    if show_plot:
        plt.show()

    plt.savefig('output/' + header.replace(' ', '_') + '.png')
    plt.close()
