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


def box_plot(
        names,
        results,
        header,
        show_plot=True,
        persist=False,
        x_label='',
        y_label='',
        save_dir_name='default'):
    # boxplot algorithm comparison
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.boxplot(results.tolist(), showfliers=False)
    ax.set_xticklabels(names)
    plt.ylabel(y_label)
    plt.xlabel(x_label) #TODO: not generalized name
    plt.xticks(rotation=0)

    # max_val = 1.02 # min(1.01, max(results.max()))

    # min_val = min(results.min())
    # axis_offset = (max_val - min_val) * 1.05
    # y_max = max_val  # 1.01
    # y_min = min(results.min()) - axis_offset  # 0.70

    # plt.ylim(ymax=max_val)

    from matplotlib.ticker import FormatStrFormatter
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    # # to make it a square uncoment
    # ratio = 1
    # xleft, xright = ax.get_xlim()
    # ybottom, ytop = ax.get_ylim()
    # # the abs method is used to make sure that all numbers are positive
    # # because x and y axis of an axes maybe inversed.
    # ax.set_aspect(abs((xright - xleft) / (ybottom - ytop)) * ratio)

    # plt.tight_layout()

    if show_plot:
        plt.show()

    if persist:
        from os.path import join, exists
        from os import makedirs

        dir_path = join('output', save_dir_name)
        if not exists(dir_path):
            makedirs(dir_path)

        file_name = header.replace(' ', '_') + '.png'

        save_path = join(dir_path, file_name)

        plt.savefig(save_path)

    plt.close()


def line_plots(
        x,
        ys,
        line_labels,
        header,
        show_plot=True,
        persist=False,
        x_label='',
        y_label='',
        save_dir_name='default'):

    fig = plt.figure()
    ax = fig.add_subplot(111)
    lines = []
    for y, line_label in zip(ys, line_labels):
        line, = plt.plot(
            x,
            y,
            linestyle='-',
            marker='o',
            label=line_label
        )
        lines.append(line)

    plt.xticks(x)

    plt.legend(handles=lines, loc='lower right')
    plt.ylabel(y_label)
    plt.xlabel(x_label)  # TODO: not generalized name
    plt.xticks(rotation=0)

    from matplotlib.ticker import FormatStrFormatter
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    if show_plot:
        plt.show()

    if persist:
        file_name = header.replace(' ', '_') + '.png'
        persist_viz(plt, save_dir_name, file_name)

    plt.close()


def line_plot(
        names,
        results,
        header,
        show_plot=True,
        persist=False,
        x_label='',
        y_label='',
        save_dir_name='default'):
    fig = plt.figure()
    ax = fig.add_subplot(111)

    x = range(len(results[0]))
    line_min, = plt.plot(
        x,
        results[0].tolist(),
        linestyle='-.',
        marker='o',
        label=y_label + ' min'
    )

    line_max, = plt.plot(
        x,
        results[1].tolist(),
        linestyle='--',
        marker='o',
        label=y_label + ' max'
    )

    line_mean, = plt.plot(
        x,
        results[2].tolist(),
        linestyle='-',
        marker='o',
        label=y_label + ' mean'
    )

    plt.legend(handles=[line_min, line_max, line_mean], loc='lower right')

    plt.xticks(range(0, len(names)), names)

    plt.ylabel(y_label)
    plt.xlabel(x_label)  # TODO: not generalized name
    plt.xticks(rotation=0)

    from matplotlib.ticker import FormatStrFormatter
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

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
        file_name = header.replace(' ', '_') + '.png'
        persist_viz(plt, save_dir_name, file_name)

    plt.close()


def persist_viz(plt, save_dir_name, file_name):
    from os.path import join, exists
    from os import makedirs

    dir_path = join('output', save_dir_name)
    if not exists(dir_path):
        makedirs(dir_path)

    save_path = join(dir_path, file_name)
    plt.savefig(save_path)