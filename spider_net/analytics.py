import pandas as pd
import pickle as pkl
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import numpy as np

from spider_net.chroma import color_create

colors = color_create()
plt.style.use('material')


def init_clip(arr):
    for i in range(len(arr)):
        if arr[i] != None:
            break
    return [0] * i + [x if x else 0 for x in arr[i:]]


def both_clip(arr):
    for i in range(len(arr)):
        if arr[i] is not None:
            break

    for j, val in enumerate(arr[::-1]):
        if val is not None:
            break

    if j!=0:
        return np.array(arr[i:-j]), i, len(arr) - j
    else:
        return np.array(arr[i:]), i, len(arr) - j



def net_analytics(investigate='print', interval=None, subset=None):
    with open('pickles/analytics', "rb") as f:
        [super_g, analytics, name_to_key, out_str] = pkl.load(f)

    if investigate == 'network':
        return super_g
    elif investigate == 'print':
        print(out_str)
        return None

    elif 'history' in investigate:
        # == Individual Cell Plots
        weight_area = {}
        grad_area = {}

        for chain_idx, chain in analytics.items():
            for cell_idx, cell in chain.items():
                for name, edge_group in cell.items():
                    if name == 'tower':
                        continue
                    key, edge, lineage = edge_group['key'], edge_group['analytics'], edge_group['lineage']

                    op0 = list(edge.keys())[0]
                    tick_max = len(edge[op0]['pruner'])
                    epochs = tick_max//782 + 1
                    label = False
                    if len([x for x in edge[op0]['pruner'] if x]):
                        if 'cell' in investigate:
                            fig, axes = plt.subplots(3, 1, figsize=(16, 9))
                            print("Cell {} Edge {}".format(cell_idx, name))
                            print("Key:", key)
                            print("Lineage: {}".format(lineage))
                            axes[0].set_title("C{} E{}".format(cell_idx, name))
                            axes[0].set_ylim(-.1, .1)
                            axes[2].set_ylim(-1, 1)
                            axes[1].set_title("Pruner Toggle")
                            axes[2].set_title("Grad")
                            for a in range(len(axes)):
                                axes[a].set_xticks(np.arange(0, tick_max, 782))
                                axes[a].set_xticklabels([int(x) for x in np.arange(0, epochs)])
                                axes[a].set_xlim(0, tick_max)

                        age = len([x for x in edge[op0]['pruner'] if x]) / 782
                        total_weight_area = None

                        for op_name, analytics in edge.items():
                            if subset is not None and op_name not in subset:
                                continue
                            #if name == 1 and cell_idx = 1 and op_name == 'Identity':
                            #    return analytics['grad'], analytics['pruner']
                            # plot individual cell info
                            if 'cell' in investigate:
                                axes[0].plot(analytics['pruner'], c=colors[op_name]['hex'], label=op_name)
                                grad = pd.Series(analytics['grad']).rolling(100).mean()

                                sg, start, end = both_clip(analytics['pruner_sg'])
                                axes[1].fill_between(np.arange(start, end),
                                                     sg+colors[op_name]['pos'],
                                                     np.ones(end-start)*colors[op_name]['pos'],
                                                     color=colors[op_name]['hex'],
                                                     step='pre')
                                if start > 0:
                                    axes[1].text(.9*start,
                                                 len(colors)/2-.1,
                                                 "From E{}".format(lineage[-1]),
                                                 color='#263238')
                                axes[2].plot(analytics['grad'],
                                             c=colors[op_name]['hex'])

                            # retrieve overall plot info
                            if total_weight_area is None:
                                total_weight_area = np.array(init_clip(analytics['pruner']))
                                total_grad = -np.array(init_clip(analytics['grad']))
                            else:
                                total_weight_area += np.array(init_clip(analytics['pruner']))
                                total_grad -= np.array(init_clip(analytics['grad']))

                        if 'cell' in investigate:
                            axes[1].set_yticks(np.arange(0, len(colors))+.5)
                            axes[1].set_yticklabels(list(colors.keys()), rotation=0)
                            minor_locator = AutoMinorLocator(2)
                            axes[1].yaxis.set_minor_locator(minor_locator)
                            axes[1].grid(True, which='minor')
                            axes[1].grid(False, which='major')

                            handles, labels = axes[0].get_legend_handles_labels()
                            fig.legend(handles, labels, loc='center right')
                            plt.subplots_adjust(hspace=.3)

                            plt.savefig('figures/Cell{}_Edge{}.png'.format(cell_idx, name), facecolor='#263238',
                                        edgecolor='none')
                            plt.show()
                            print('=' * 100)


                        weight_area['Cell {}, {} (age: {})'.format(cell_idx, name, age)] = total_weight_area
                        grad_area['Cell {}, {} (age: {})'.format(cell_idx, name, age)] = total_grad


        if 'overall' in investigate:
            fig, axes = plt.subplots(3, 1, figsize=(16, 18))
            axes[0].set_title("Pruner Weight by Edge")
            tup = sorted(list(weight_area.items()), key=lambda x: x[1][-1], reverse=True)

            kcolor = {}
            cmap = plt.get_cmap('plasma')
            for i, (k, v) in enumerate(tup):
                kcolor[k] = cmap(i / len(tup))
                concat = [v[i*782*interval:(i+1)*782*interval] - v[i*782*interval] for i in range(epochs//4)]
                a = np.concatenate(concat)
                axes[0].plot(a, label=k, c=kcolor[k], solid_joinstyle='round')
            axes[0].legend()

            axes[1].set_title("Pruner Grad by Edge")
            for i, (k, v) in enumerate(grad_area.items()):
                a = pd.Series(v).rolling(500).mean()
                axes[1].plot(a, label=k, c=kcolor[k], solid_joinstyle='round')
            axes[1].legend()
            axes[2].set_xlabel("Accumulated Grad")
            axes[2].set_ylabel("Pruner Weight")
            axes[2].set_title("Mean Gradient vs Pruner Weight")

            for a in range(2):
                axes[a].set_xticks(np.arange(0, tick_max, 782))
                axes[a].set_xticklabels([int(x) for x in np.arange(0, epochs)])

            for k, _ in tup:
                plt.scatter(np.mean(grad_area[k]), weight_area[k][-1], c=[kcolor[k]], label=k)
            axes[2].legend()

            plt.savefig("figures/cell_edge_evaluations.png", facecolor='#263238', edgecolor='none')
            plt.show()