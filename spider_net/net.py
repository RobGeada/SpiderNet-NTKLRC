import torch
import torch.nn as nn

import numpy as np
from graphviz import Digraph
import random
import matplotlib.pyplot as plt

from spider_net.ops import *
from spider_net.chroma import color_create
from spider_net.pruner import PrunableOperation, PrunableTower, Pruner
from spider_net.helpers import *
from spider_net.trainers import size_test

arrow_char = "↳"


class Edge(nn.Module):
    def __init__(self, ops, dim, op_size, name):
        super().__init__()
        self.operation_set = commons if ops is None else ops
        self.dim = dim
        self.op_sizes = op_size
        self.post_splittable = dim[-1] == 1
        self.name = name
        self.accumulated_grad = []

        self.ops = []
        for i, (key, op) in enumerate(self.operation_set.items()):
            prune_op = PrunableOperation(op_function=op,
                                         name=key,
                                         c_in=self.dim[1],
                                         mem_size=self.op_sizes[dim][key],
                                         stride=self.dim[-1],
                                         pruner_init=.01,
                                         prune=True)
            self.ops.append(prune_op)
        self.ops = nn.ModuleList(self.ops)
        self.num_ops = len([op for op in self.ops if not op.zero])

        if self.num_ops:
            self.norm = nn.BatchNorm2d(self.dim[1])
        self.zero = None

    def get_grad(self):
        return np.mean(self.accumulated_grad) if len(self.accumulated_grad) else 0

    def deadhead(self, prune_interval):
        dhs = sum([op.deadhead(prune_interval) for i, op in enumerate(self.ops)])
        self.used = sum([op.pruner.mem_size for op in self.ops if not op.zero])
        self.num_ops -= dhs
        if self.num_ops == 0:
            self.norm = None
            self.zero = Zero(stride=self.stride)
        return dhs

    def __repr__(self):
        return "Edge {}".format(self.name, self.op_sizes)

    def forward(self, x, drop_prob):
        if self.num_ops:
            outs = [op(x) if op.name in ['Identity', 'Zero'] else drop_path(op(x), drop_prob) for op in self.ops]
            return self.norm(sum(outs))
        else:
            return self.zero(x)


class Cell(nn.Module):
    def __init__(self, cell_idx, input_dim, op_sizes):
        super().__init__()

        self.name = cell_idx
        self.edge_counter = Count()
        self.input_dim = input_dim
        self.op_sizes = op_sizes
        self.op_keys = list(self.op_sizes.keys())
        self.op_size_est = np.mean([sum(v.values()) for v in op_sizes.values()])

        self.edges = nn.ModuleDict()
        self.edges[self.encode(0, 1)] = Edge(None, list(self.op_sizes.keys())[0], self.op_sizes, self.edge_counter())
        self.output_node = 0
        self.forward_iterator = []
        self.set_op_order()

    def get_new_op_sizes(self, orig):
        if orig.dim[-1] == 2:
            return self.op_keys[-1], orig.dim
        else:
            return orig.dim, orig.dim

    def split_edge(self, key, where):
        orig_device = self.edges[list(self.edges.keys())[0]].ops[0].pruner.weight.device
        i, j = self.decode(key)
        new_sizes = self.get_new_op_sizes(self.edges[key])
        shifted_edges = []
        edge_ops = [op.name for op in self.edges[key].ops]
        new_ops = {k: v for k, v in commons.items() if k in edge_ops}

        if where == 'pre':
            cond_i = lambda x, y: x >= j
            cond_j = lambda x, y: y > j
            edge_a = self.edges[key]
            edge_b = Edge(new_ops, new_sizes[0], self.op_sizes, self.edge_counter())
            edge_c = Edge(new_ops, new_sizes[1], self.op_sizes, self.edge_counter())
            new_edges = [edge_b, edge_c]
        elif where == 'post':
            if not self.edges[key].post_splittable:
                where = np.random.choice(['pre', 'hypot'])[0]
                return self.split_edge(key, where)
            cond_i = lambda x, y: x >= j
            cond_j = lambda x, y: y >= j
            edge_a = Edge(new_ops, new_sizes[0], self.op_sizes, self.edge_counter())
            edge_b = self.edges[key]
            edge_c = Edge(new_ops, new_sizes[0], self.op_sizes, self.edge_counter())
            new_edges = [edge_a, edge_c]
        else:
            cond_i = lambda x, y:  x >= j
            cond_j = lambda x, y: y >= j
            edge_a = Edge(new_ops, new_sizes[1], self.op_sizes, self.edge_counter())
            edge_b = Edge(new_ops, new_sizes[0], self.op_sizes, self.edge_counter())
            edge_c = self.edges[key]
            new_edges = [edge_a, edge_b]

        for tgt_key, tgt_edge in self.edges.items():
            if tgt_key == key:
                continue
            tgt_i, tgt_j = self.decode(tgt_key)
            if cond_i(tgt_i, tgt_j):
                tgt_i += 1
            if cond_j(tgt_i, tgt_j):
                tgt_j += 1
            shifted_edges.append([self.encode(tgt_i, tgt_j), tgt_edge])

        shifted_edges.append([self.encode(i, j), edge_a])
        shifted_edges.append([self.encode(j, j+1), edge_b])
        shifted_edges.append([self.encode(i, j + 1), edge_c])

        self.edges = nn.ModuleDict()
        for k, edge in shifted_edges:
            self.edges[k] = edge
        self.edges.to(orig_device)

        self.set_op_order()
        return new_edges

    def encode(self, i, j):
        return "{}->{}".format(i,j)

    def decode(self, edge):
        i, j = edge.split("->")
        return int(i), int(j)

    def set_op_order(self):
        node_inputs = {}
        for key in self.edges.keys():
            i, j = self.decode(key)

            if j > self.output_node:
                self.output_node = j

            if node_inputs.get(i) is None:
                node_inputs[i] = []
            node_inputs[i].append(key)

        node_order = sorted(node_inputs.keys())

        self.forward_iterator = []
        for node in node_order:
            for edge in node_inputs[node]:
                i, j = self.decode(edge)
                last = edge == node_inputs[node][-1]
                self.forward_iterator.append([edge, i, j, last])

    def plot_cell(self, subgraph=None, color_by='grad', **kwargs):
        g = Digraph() if subgraph is None else subgraph
        if color_by == 'op':
            colors = color_create()

        for key, edge in self.edges.items():
            i, j = self.decode(key)
            i_str = "{}_{}".format(self.name, i)
            j_str = "{}_{}".format(self.name, j)
            g.node(i_str, label=str(i))
            g.node(j_str, label=str(j))

            if color_by == 'op':
                for op in edge.ops:
                    if not op.zero:
                        g.edge(i_str, j_str, color=colors[op.name]['hex'])
            elif color_by in ['attrition', 'grad']:
                if color_by == 'attrition':
                    cmap = plt.get_cmap('Reds_r')
                    attrition = (len(commons) - edge.num_ops) / len(commons)
                    color = rgb_to_hex(cmap(attrition))
                else:
                    cmap = plt.get_cmap('Reds')
                    color = rgb_to_hex(cmap(kwargs['norm'](edge.get_grad())))
                g.edge(i_str, j_str,
                       color=color,
                       label=str(edge.num_ops),
                       penwidth=str(edge.num_ops**1.3),
                       arrowhead='none')
            else:
                raise ValueError("Invalid 'color_by' specified: {}".format(color_by))
        return g

    def __repr__(self, out_format=None):
        dim_rep = self.input_dim[1:3]
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
        self.input_dim = hypers['input_dim']
        self.out_classes = hypers['dataset']['classes']
        self.reductions = hypers['reductions']
        self.scale = hypers['scale']
        self.prune = True
        self.drop_prob = hypers['drop_prob']
        self.gpu_space = hypers['gpu_space']
        self.model_id = hypers.get('model_id', namer())
        self.hypers = hypers

        self.initializer = initializer(self.input_dim[1], self.scale)

        init_dim = channel_mod(self.input_dim, self.scale)
        self.dims = [cw_mod(init_dim, 2**i) for i in range(self.reductions+1)]
        self.dims = [[d+(1,), channel_mod(d, d[1]*2)+(2,)] if i != len(self.dims)-1 else [d+(1,)]
                     for i, d in enumerate(self.dims)]
        self.dims = [d for dlist in self.dims for d in dlist]

        # get all operation sizes
        size_set = compute_sizes()
        op_match = (len(size_set) > 0) and all([op in list(size_set.values())[0].keys() for op in commons])
        if not op_match or not all([dim in size_set for dim in self.dims]):
            size_set = compute_sizes(self.dims)
        size_set = {dim: {k: v for k, v in ops.items() if k in commons} for dim, ops in size_set.items()}
        self.size_set = size_set

        # build cells
        self.scalers = nn.ModuleDict()
        self.residual_scalers = nn.ModuleDict()
        self.towers = nn.ModuleDict()
        cells = []
        for cell_idx in range(self.reductions+1):
            cell_dims = self.dims[max(0, cell_idx * 2 - 1):cell_idx * 2 + 1]
            cell_sizes = {d: size_set[d] for d in cell_dims}
            dim = cell_dims[0]
            cells.append(Cell(cell_idx, dim, cell_sizes))

            if not cell_idx == self.reductions:
                if cell_idx:
                    self.residual_scalers[str(cell_idx)] = MinimumIdentity(dim[1],dim[1], 2)
                else:
                    self.residual_scalers[str(cell_idx)] = nn.Sequential()
                self.scalers[str(cell_idx)] = Scaler(dim[1], dim[1]*2)
            self.towers[str(cell_idx)] = PrunableTower(str(cell_idx), dim, self.out_classes)

        self.cells = nn.ModuleList(cells)
        self.mut_sizes = [(cell.op_size_est*2/1024) for cell in self.cells]

    def deadhead(self, prune_interval):
        old_params = general_num_params(self)
        deadheads = 0
        deadhead_spots = []
        for i, cell in enumerate(self.cells):
            for key in cell.edges.keys():
                dh = cell.edges[key].deadhead(prune_interval)
                deadheads += dh
                if dh:
                    deadhead_spots.append([i, key])

        self.log_print("Deadheaded {} operations".format(deadheads))
        print("Deadheaded", deadhead_spots)
        self.log_print("Param Delta: {:,} -> {:,}".format(old_params, general_num_params(self)))
        clean("Deadhead", verbose=False)

    def compile_grads(self):
        for i, cell in enumerate(self.cells):
            for key, edge in cell.edges.items():
                edge.accumulated_grad += [op.get_grad() for op in edge.ops]

    def get_grads(self):
        return {(i, k): e.get_grad() for i, cell in enumerate(self.cells) for k, e in cell.edges.items()}

    def clear_grad(self):
        for cell in self.cells:
            for e in cell.edges.values():
                e.accumulated_grad = []

    def mutate(self, n=1):
        mutations = []
        new_edges = []
        for i in range(n):
            grads = self.get_grads()
            grad_sum = sum(np.array(list(grads.values()))**2)
            if grad_sum != 0:
                adj_grads = [(k, v**2/grad_sum) for k, v in grads.items()]
                edges, p = zip(*adj_grads)
            else:
                edges = list(grads.keys())
                p = None

            loc = np.random.choice(np.arange(len(edges)), p=p)
            cell, edge = edges[loc]
            print(cell, edge)
            if size_test(self)[0] + self.mut_sizes[cell] < self.gpu_space - 1:
                new_edges += self.cells[cell].split_edge(edge, 'hypot')
                mutations.append((cell, edge))
        self.clear_grad()
        return mutations, new_edges

    def plot_network(self, color_by='grad'):
        from graphviz import Digraph
        g = Digraph()
        in_names = ['Initializer'] + ['{}_0'.format(i) for i in range(len(self.cells))]
        out_names = ['Initializer'] + ['{}_{}'.format(i,cell.output_node) for i, cell in enumerate(self.cells)]
        scalers = ['Initializer'] + ['Scaler {}'.format(i) for i in range(len(self.cells))]

        if color_by == 'grad':
            max_grad = max([edge.get_grad() for cell in self.cells for k, edge in cell.edges.items()])
            norm = lambda x: x/max_grad if max_grad != 0 else 1
        else:
            norm = None

        for i, cell in enumerate(self.cells):
            with g.subgraph(name='cluster_{}'.format(i)) as c:
                c.attr(style='filled', color='grey')
                c.attr(label='Cell {}'.format(i))
                cell.plot_cell(subgraph=c, color_by=color_by, norm=norm)
                c.node_attr.update(style='filled', color='white')
            g.edge(scalers[i], in_names[i+1])
            g.edge(out_names[i+1], 'Tower {}'.format(i))
            if i < len(self.cells)-1:
                g.edge(out_names[i+1], 'Scaler {}'.format(i))
                g.edge(scalers[i], 'ResScaler {}'.format(i))
                g.edge('ResScaler {}'.format(i), scalers[i+1])
        return g

    def creation_string(self):
        return "ID: '{}', Dim: {}, Classes: {}, Scale: {}, Patterns: {}".format(
            self.model_id,
            self.input_dim,
            self.out_classes,
            self.scale,
            len(self.cells)
        )

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
        out += out_format(l='', d='Dim', p='Params', c='Ops:')

        out += out_format(l="Initializer", p=general_num_params(self.initializer))
        for i,cell in enumerate(self.cells):
            out += cell.__repr__(out_format)
            if str(i) in self.towers.keys():
                out += out_format(l=" {} Aux Tower".format(arrow_char),
                                  p=general_num_params(self.towers[str(i)]))
        if 'Classifier' in self.towers:
            out += out_format(l=" {} Classifier".format(arrow_char),
                              p=general_num_params(self.towers['Classifier']))
        out += spacer.format("")
        out += out_format(l="Total", p=general_num_params(self))
        out += spacer.format("")
        return out

    def forward(self, x, drop_prob=None, verbose=False):
        outs = []
        x = self.initializer(x)
        for i, cell in enumerate(self.cells):
            cell_out = cell(x, drop_prob)
            cell_idx = str(i)
            if i != len(self.cells)-1:
                x = self.scalers[cell_idx](self.residual_scalers[cell_idx](x) + cell_out)
            outs.append(self.towers[cell_idx](cell_out))
        return torch.mean(torch.stack(outs), 0)
