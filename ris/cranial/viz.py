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


def box_plot(names, results, header, show_plot=True, persist=False, x_label='', y_label=''):
    # boxplot algorithm comparison
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.boxplot(results.tolist())
    ax.set_xticklabels(names)
    plt.ylabel(y_label)
    plt.xlabel(x_label) #TODO: not generalized name
    plt.xticks(rotation=0)

    max_val = 1.01 # min(1.01, max(results.max()))
    min_val = min(results.min())
    axis_offset = (max_val - min_val) * 1.05
    y_max = max_val  # 1.01
    y_min = min(results.min()) - axis_offset  # 0.70

    plt.ylim(ymin=y_min, ymax=y_max)
    plt.tight_layout()

    if show_plot:
        plt.show()

    if persist:
        plt.savefig('output/' + header.replace(' ', '_') + '.png')

    plt.close()


def line_plot(names, results, header, show_plot=True, persist=False, x_label='', y_label=''):
    fig = plt.figure()
    ax = fig.add_subplot(111)

    x = range(len(results[0]))
    line_min, = plt.plot(
        x,
        results[0].tolist(),
        linestyle='-.',
        label=y_label + ' min'
    )

    line_max, = plt.plot(
        x,
        results[1].tolist(),
        linestyle='--',
        label=y_label + ' max'
    )

    line_mean, = plt.plot(
        x,
        results[2].tolist(),
        linestyle='-',
        label=y_label + ' mean'
    )

    plt.legend(handles=[line_min, line_max, line_mean])

    plt.xticks(range(0, len(names)), names)

    plt.ylabel(y_label)
    plt.xlabel(x_label)  # TODO: not generalized name
    plt.xticks(rotation=0)

    # max_val = 1.01  # min(1.01, max(results.max()))
    # min_val = min(results[0].min())
    # axis_offset = (max_val - min_val) * 1.05
    # y_max = max_val  # 1.01
    # y_min = min(results[0].min()) - axis_offset  # 0.70
    #
    # plt.ylim(ymin=y_min, ymax=y_max)
    # plt.tight_layout()

    if show_plot:
        plt.show()

    if persist:
        plt.savefig('output/line/' + header.replace(' ', '_') + '.png')

    plt.close()