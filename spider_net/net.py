import torch
import torch.nn as nn

import numpy as np
from graphviz import Digraph
import random
import matplotlib.pyplot as plt

from spider_net.ops import *
from spider_net.chroma import color_create
from spider_net.pruner import PrunableOperation, PrunableTower
from spider_net.helpers import *
from spider_net.trainers import size_test

arrow_char = "↳"


def encode(i, j, s="->"):
    return "{}{}{}".format(i, s, j)


def decode(edge, s="->"):
    i, j = edge.split(s)
    if s != "->":
        return i, int(j)
    else:
        return int(i), int(j)


class Edge(nn.Module):
    def __init__(self, ops, dim, op_size, name, mut_metric, data_index=0, lineage=None, mutate_flag=False):
        super().__init__()
        self.operation_set = commons if ops is None else ops
        self.dim = dim

        self.op_sizes = op_size
        self.post_splittable = dim[-1] == 1
        self.name = name
        self.data_index = data_index
        self.mutate_flag = mutate_flag
        self.growth_factor = []
        self.lineage = [] if lineage is None else lineage
        self.mut_metric = mut_metric

        self.ops = []
        for i, (key, op) in enumerate(self.operation_set.items()):
            prune_op = PrunableOperation(op_function=op,
                                         name=key,
                                         c_in=self.dim[1],
                                         mem_size=self.op_sizes[dim][key],
                                         stride=self.dim[-1],
                                         pruner_init=.01,
                                         prune=True,
                                         metric=self.mut_metric['metric'],
                                         start_idx=self.data_index)
            self.ops.append(prune_op)
        self.ops = nn.ModuleList(self.ops)
        self.num_ops = len([op for op in self.ops if not op.zero])
        self.used = self.get_edge_size()
        if self.num_ops:
            self.norm = nn.BatchNorm2d(self.dim[1])
        self.zero = None

    def deadhead(self, prune_interval):
        dhs = sum([op.deadhead(prune_interval) for i, op in enumerate(self.ops)])
        self.used = self.get_edge_size()
        self.num_ops -= dhs
        if self.num_ops == 0:
            self.norm = None
            self.zero = Zero(stride=self.stride)
        return dhs

    def get_growth(self):
        if len(self.growth_factor)==0:
            return np.inf
        else:
            return self.mut_metric['agg'](self.growth_factor)


    def get_edge_size(self):
        return sum([op.pruner.mem_size for op in self.ops if not op.zero])

    def __repr__(self):
        return "Edge {}".format(self.name, self.op_sizes)

    def forward(self, x, drop_prob):
        if self.num_ops:
            outs = [op(x) if op.name in ['Identity', 'Zero'] else drop_path(op(x), drop_prob) for op in self.ops]
            return self.norm(sum(outs))
        else:
            return self.zero(x)


class Cell(nn.Module):
    def __init__(self, cell_idx, input_dim, op_sizes, mut_metric, data_index=0):
        super().__init__()

        self.name = cell_idx
        self.edge_counter = Count()
        self.input_dim = input_dim
        self.op_sizes = op_sizes
        self.data_index = data_index
        self.op_keys = list(self.op_sizes.keys())

        self.edges = nn.ModuleDict()
        self.edges[encode(0, 1)] = Edge(None,
                                        list(self.op_sizes.keys())[0],
                                        self.op_sizes,
                                        self.edge_counter(),
                                        mut_metric,
                                        self.data_index)
        self.output_node = 0
        self.forward_iterator = []
        self.set_op_order()

    def get_new_op_sizes(self, orig):
        if orig.dim[-1] == 2:
            return self.op_keys[-1], orig.dim
        else:
            return orig.dim, orig.dim

    def split_edge(self, key, device=torch.device("cpu"), data_index=0):
        i, j = decode(key)
        new_sizes = self.get_new_op_sizes(self.edges[key])
        shifted_edges = []
        edge_ops = [op.name for op in self.edges[key].ops if not op.zero]
        new_ops = {k: v for k, v in commons.items() if k in edge_ops}

        cond_i = lambda x, y: x >= j
        cond_j = lambda x, y: y >= j
        edge_a = Edge(new_ops,
                      new_sizes[1],
                      self.op_sizes,
                      self.edge_counter(),
                      data_index,
                      lineage=self.edges[key].lineage + [self.edges[key].name],
                      mutate_flag=True)
        edge_b = Edge(new_ops,
                      new_sizes[0],
                      self.op_sizes,
                      self.edge_counter(),
                      data_index,
                      lineage=self.edges[key].lineage + [self.edges[key].name],
                      mutate_flag=True)
        edge_c = self.edges[key]
        edge_c.mutate_flag = True
        new_edges = [edge_a, edge_b]

        for tgt_key, tgt_edge in self.edges.items():
            if tgt_key == key:
                continue
            tgt_i, tgt_j = decode(tgt_key)
            if cond_i(tgt_i, tgt_j):
                tgt_i += 1
            if cond_j(tgt_i, tgt_j):
                tgt_j += 1
            shifted_edges.append([encode(tgt_i, tgt_j), tgt_edge])

        shifted_edges.append([encode(i, j), edge_a])
        shifted_edges.append([encode(j, j+1), edge_b])
        shifted_edges.append([encode(i, j + 1), edge_c])
        edge_c.growth_factor = []

        self.edges = nn.ModuleDict()
        for k, edge in shifted_edges:
            self.edges[k] = edge

        self.edges = self.edges.to(device)
        self.set_op_order()
        return new_edges

    def set_op_order(self):
        node_inputs = {}
        for key in self.edges.keys():
            i, j = decode(key)

            if j > self.output_node:
                self.output_node = j

            if node_inputs.get(i) is None:
                node_inputs[i] = []
            node_inputs[i].append(key)

        node_order = sorted(node_inputs.keys())

        self.forward_iterator = []
        for node in node_order:
            for edge in node_inputs[node]:
                i, j = decode(edge)
                last = edge == node_inputs[node][-1]
                self.forward_iterator.append([edge, i, j, last])

    def plot_cell(self, subgraph=None, color_by='growth', **kwargs):
        g = Digraph() if subgraph is None else subgraph
        if color_by == 'op':
            colors = color_create()

        for key, edge in self.edges.items():
            i, j = decode(key)
            i_str = kwargs.get('prefix', "") + encode(self.name, i, "_")
            j_str = kwargs.get('prefix', "") + encode(self.name, j, "_")
            g.node(i_str, label=str(i))
            g.node(j_str, label=str(j))

            if color_by == 'op':
                for op in edge.ops:
                    if not op.zero:
                        g.edge(i_str, j_str, color=colors[op.name]['hex'])
            elif color_by in ['attrition', 'growth']:
                if color_by == 'attrition':
                    cmap = plt.get_cmap('Reds_r')
                    attrition = (len(commons) - edge.num_ops) / len(commons)
                    color = rgb_to_hex(cmap(attrition))
                else:
                    cmap = plt.get_cmap('Reds')
                    color = rgb_to_hex(cmap(kwargs['norm'](edge.get_growth())))
                g.edge(i_str, j_str,
                       color=color,
                       label=str(edge.num_ops),
                       penwidth=str(edge.num_ops**1.3),
                       arrowhead='none')
            else:
                raise ValueError("Invalid 'color_by' specified: {}".format(color_by))
        return g

    def __repr__(self, out_format=None):
        dim_rep = list(self.op_sizes.keys())[-1][1:3]
        dim = '{:^4}x{:^4}'.format(*dim_rep)
        if out_format is not None:
            ops = sum([len(edge.ops) for edge in self.edges.values()])
            layer_name = "Cell {:<2}".format(self.name)
            out = out_format(l=layer_name, d=dim, p=general_num_params(self), c=ops)
            return out
        else:
            return "Cell {:<2}: D: {} P:{}".format(self.name, dim, general_num_params(self))

    def forward(self, x, drop_prob):
        node_storage = {0: x}
        for edge, i, j, last in self.forward_iterator:
            if node_storage.get(j) is None:
                node_storage[j] = self.edges[edge](node_storage[i], drop_prob)
            else:
                node_storage[j] += self.edges[edge](node_storage[i], drop_prob)
            if last:
                del node_storage[i]

        return node_storage[self.output_node]


class Net(nn.Module):
    def __init__(self, hypers):
        super().__init__()
        wipe_output()
        self.input_dim = hypers['input_dim']
        self.out_classes = hypers['dataset']['classes']
        self.reductions = hypers['reductions']
        self.scale = hypers['scale']
        self.prune = True
        self.drop_prob = hypers['drop_prob']
        self.gpu_space = hypers['gpu_space']
        self.chains = hypers['chains']
        self.model_id = hypers.get('model_id', namer())
        self.mut_metric = hypers['mut_metric']
        self.data_index = 0
        self.device = torch.device(hypers['device'])
        self.epoch = 0
        self.hypers = hypers

        self.initializers = nn.ModuleList([initializer(self.input_dim[1], self.scale*2**i) for i in range(self.chains)])

        init_dims = [channel_mod(self.input_dim, self.scale*(2**i)) for i in range(self.chains)]
        self.dims = [[cw_mod(dim, 2**i) for i in range(self.reductions+1)] for dim in init_dims]
        self.dims = [[[d+(1,), channel_mod(d, d[1]*2)+(2,)] if i != len(dim_group)-1 else [d+(1,)]
                     for i, d in enumerate(dim_group)] for dim_group in self.dims]
        self.dims = [[d for dim in dim_group for d in dim] for dim_group in self.dims]
        self.all_dims = [d for dim_group in self.dims for d in dim_group]

        # get all operation sizes
        size_set = compute_sizes()
        op_match = (len(size_set) > 0) and all([op in list(size_set.values())[0].keys() for op in commons])
        if not op_match or not all([dim in size_set for dim in self.all_dims]):
            size_set = compute_sizes(self.all_dims)
        size_set = {dim: {k: v for k, v in ops.items() if k in commons} for dim, ops in size_set.items()}
        self.size_set = size_set

        # build cells
        self.scalers = nn.ModuleDict({str(c): nn.ModuleDict() for c in range(self.chains)})
        self.residual_scalers = nn.ModuleDict({str(c): nn.ModuleDict() for c in range(self.chains)})
        self.towers = nn.ModuleDict({str(c): nn.ModuleDict() for c in range(self.chains)})
        self.cells = nn.ModuleDict({str(c): nn.ModuleList() for c in range(self.chains)})

        for chain in range(self.chains):
            chain_str = str(chain)
            for cell_idx in range(self.reductions+1):
                cell_dims = self.dims[chain][max(0, cell_idx * 2 - 1):cell_idx * 2 + 1]
                cell_sizes = {d: size_set[d] for d in cell_dims}
                dim = cell_dims[0]
                cell_name = "{}_{}".format(chain_str, cell_idx)
                self.cells[chain_str].append(Cell(cell_name, dim, cell_sizes, mut_metric=self.mut_metric))

                if not cell_idx == self.reductions:
                    if cell_idx:
                        self.residual_scalers[chain_str][str(cell_idx)] = MinimumIdentity(dim[1], dim[1], 2)
                    else:
                        self.residual_scalers[chain_str][str(cell_idx)] = nn.Sequential()
                    self.scalers[chain_str][str(cell_idx)] = Scaler(dim[1], dim[1]*2)
                self.towers[chain_str][str(cell_idx)] = PrunableTower(str(cell_idx), dim, self.out_classes)
        self.mut_sizes = {}

    def all_cells(self, enum=True):
        if enum:
            return ((i, chain, cell) for chain in range(self.chains) for i, cell in enumerate(self.cells[str(chain)]))
        else:
            return (cell for chain in range(self.chains) for cell in self.cells[str(chain)])

    def deadhead(self, prune_interval):
        old_params = general_num_params(self)
        deadheads = 0
        deadhead_spots = []
        for i, _,  cell in self.all_cells(enum=True):
            for key in cell.edges.keys():
                dh = cell.edges[key].deadhead(prune_interval)
                deadheads += dh
                if dh:
                    deadhead_spots.append([i, key])

        self.log_print("Deadheaded {} operations".format(deadheads))
        print("Deadheaded", deadhead_spots)
        self.log_print("Param Delta: {:,} -> {:,}".format(old_params, general_num_params(self)))
        clean("Deadhead", verbose=False)

    def update_mut_sizes(self):
        for i, c, cell in self.all_cells():
            self.mut_sizes[encode(c, i, ",")] = {}
            for k, e in cell.edges.items():
                self.mut_sizes[encode(c, i, ",")][k] = 2*e.get_edge_size()/1024

    def compile_growth_factors(self):
        for cell in self.all_cells(enum=False):
            for key, edge in cell.edges.items():
                [op.log_analytics() for op in edge.ops]
                edge.growth_factor += [op.get_growth_factor() for op in edge.ops if not op.zero]

    def compile_pruner_stats(self):
        for cell in self.all_cells(enum=False):
            for key, edge in cell.edges.items():
                [op.pruner.track_gates() for op in edge.ops]
                [op.pruner.clamp() for op in edge.ops]
        for chain in range(self.chains):
            for cell_idx, tower in self.towers[str(chain)].items():
                tower.track_gates()

    def get_growth_factors(self):
        factors = {}
        for i, c, cell in self.all_cells():
            for k, edge in cell.edges.items():
                if not edge.mutate_flag:
                    factors[(encode(c, i, ","), k)] = edge.get_growth()
        return factors

    def clear_grad(self):
        for cell in self.all_cells(enum=False):
            for e in cell.edges.values():
                e.growth_factor = []
                e.mutate_flag = False

    def mutate(self, n=1):
        mutations = []
        new_edges = []
        mutation_in_chain = set()
        for i in range(n):
            growth_factors = self.get_growth_factors()
            growth_sum = sum(list(growth_factors.values()))
            if growth_sum != np.inf:
                edges = sorted(list(growth_factors.items()), key=lambda x: x[1], reverse=mut_metric['which']=='max')
                edges = [x[0] for x in edges]
                loc = 0
            else:
                edges = [edge for edge in growth_factors.keys() if edge[0] not in mutation_in_chain]
                loc = np.random.choice(np.arange(len(edges)), replace=False)
                mutation_in_chain.add(edges[loc][0])
                print(edges[loc][0])
            chain_cell, edge = edges[loc]
            chain, cell = decode(chain_cell, ",")

            size, overfill = size_test(self)
            self.update_mut_sizes()
            mut_size = self.mut_sizes[chain_cell][edge]
            print(overfill, size, mut_size, size + mut_size)
            if not overfill and (size + mut_size) < self.gpu_space:
                edges = self.cells[chain][cell].split_edge(edge, self.device, self.data_index)
                new_edges += edges
                mutations.append((chain_cell, edge))

        if len(new_edges):
            self.clear_grad()
        return mutations, new_edges

    def plot_network(self, color_by='growth'):
        super_g = Digraph()
        for chain in range(self.chains):
            with super_g.subgraph(name='chain_{}'.format(chain)) as g:
                chain_str = str(chain)
                g.attr(style='filled', color='red')
                g.attr(label='Chain {}'.format(chain))
                prefix = "C" + chain_str + " "

                if color_by == 'growth':
                    max_grad = max([edge.get_growth() for cell in self.cells[chain_str] for k, edge in cell.edges.items()])
                    norm = lambda x: x/max_grad if max_grad != 0 else 1
                else:
                    norm = None

                for i, cell in enumerate(self.cells[chain_str]):
                    with g.subgraph(name='cluster_{}'.format(i)) as c:
                        c.attr(style='filled', color='grey')
                        c.attr(label='Cell {}'.format(i))
                        cell.plot_cell(subgraph=c, color_by=color_by, norm=norm, prefix=prefix)
                        c.node_attr.update(style='filled', color='white')
        return super_g

    def creation_string(self):
        return "ID: '{}', Dim: {}, Classes: {}, Scale: {}, Patterns: {}".format(
            self.model_id,
            self.input_dim,
            self.out_classes,
            self.scale,
            len(self.cells)
        )

    def save_analytics(self):
        super_g = self.plot_network()
        analytics = {}
        name_to_key = {}
        for chain in range(self.chains):
            if not analytics.get(chain):
                analytics[chain] = {}
                name_to_key[chain] = {}
            for cell_idx, cell in enumerate(self.cells[str(chain)]):
                if not analytics[chain].get(cell_idx):
                    analytics[chain][cell_idx] = {'tower': self.towers[str(chain)][str(cell_idx)].analytics}
                    name_to_key[chain][cell_idx] = {}

                for k, edge in sorted(cell.edges.items(), key=lambda x: x[1].name):
                    analytics[chain][cell_idx][edge.name] = {'key': k,
                                                             'analytics': {op.name: op.analytics for op in edge.ops},
                                                             'lineage': edge.lineage}

        out_str = self.__str__()

        with open('pickles/analytics', "wb") as f:
            pkl.dump([super_g, analytics, name_to_key, out_str], f)

    def reset_parameters(self):
        self.data_index = 0
        self.epoch = 0
        for module in self.modules():
            if 'reset_parameters' in dir(module) and type(module) != type(self):
                module.reset_parameters()

    def __str__(self):
        def out_format(l="", p="", d="", c=""):
            sep = ' : '
            out_fmt = '{l:}{s}{d}{s}{p}{s}{c}\n'.format(l='{l:<20}', d='{d:^12}', p='{p:^12}', c='{c:^9}', s=sep)
            try:
                p = "{:,}".format(p)
            except ValueError:
                pass
            c = "" if c is None else c
            return out_fmt.format(l=l, p=p, d=d, c=c)

        spacer = '{{:=^{w}}}\n'.format(w=len(out_format()))
        out = spacer.format(" NETWORK ")
        out += spacer.format(" "+self.model_id+" ")
        out += out_format(l='Epoch {}'.format(self.epoch), d='Dim', p='Params', c='Ops:')

        for chain in range(self.chains):
            out += spacer.format(" Chain {} ".format(chain))
            out += out_format(l="Initializer", p=general_num_params(self.initializers[chain]))
            for i, cell in enumerate(self.cells[str(chain)]):
                out += cell.__repr__(out_format)
                if str(i) in self.towers[str(chain)].keys():
                    out += out_format(l=" {} Aux Tower".format(arrow_char),
                                      p=general_num_params(self.towers[str(chain)][str(i)]))
            if 'Classifier' in self.towers:
                out += out_format(l=" {} Classifier".format(arrow_char),
                                  p=general_num_params(self.towers['Classifier']))
        out += spacer.format("")
        out += out_format(l="Total", p=general_num_params(self))
        out += spacer.format("")
        return out

    def forward(self, x, drop_prob=0., verbose=False):
        outs = []
        orig_x = x
        for chain in range(self.chains):
            chain_str = str(chain)
            x = self.initializers[chain](orig_x)
            for i, cell in enumerate(self.cells[chain_str]):
                cell_out = cell(x, drop_prob)
                cell_idx = str(i)
                if i != len(self.cells[chain_str]) - 1:
                    x = self.scalers[chain_str][cell_idx](self.residual_scalers[chain_str][cell_idx](x) + cell_out)
                outs.append(self.towers[chain_str][cell_idx](cell_out))
        return outs

