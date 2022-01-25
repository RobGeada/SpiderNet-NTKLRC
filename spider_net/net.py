import torch
import torch.nn as nn

import numpy as np
from graphviz import Digraph
import random
import matplotlib.pyplot as plt
import shap
import uuid

from spider_net.ops import *
from spider_net.chroma import color_create
from spider_net.pruner import PrunableOperation, PrunableTower
from spider_net.helpers import *
from spider_net.trainers import size_test, top_k_accuracy

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
    def __init__(self, ops, dim, stride, op_size, name, data_index=0, prune=True, lineage=None, original=True):
        super().__init__()
        self.operation_set = commons if ops is None else ops
        self.dim = dim

        self.op_sizes = op_size
        self.name = name
        self.data_index = data_index
        self.growth_factor = {'weight': [], 'grad': []}
        self.shap = None
        self.lineage = [] if lineage is None else lineage

        self.ops = []

        for i, (key, op) in enumerate(self.operation_set.items()):
            prune_op = PrunableOperation(op_function=op,
                                         name=key,
                                         c_in=self.dim[1],
                                         mem_size=self.op_sizes[dim][key],
                                         stride=stride,
                                         pruner_init=.01,
                                         prune=prune,
                                         start_idx=self.data_index)
            self.ops.append(prune_op)
        self.ops = nn.ModuleList(self.ops)
        self.num_ops = len([op for op in self.ops if not op.zero])
        self.prev_num_ops = self.num_ops 
        self.used = self.get_edge_size()
        self.original = original
        self.hash = uuid.uuid4().hex
        self.zero = None

    def deadhead(self, prune_epochs, prune_interval):
        self.prev_num_ops = self.num_ops
        dhs = [op.orig_name for i, op in enumerate(self.ops) if op.deadhead(prune_epochs, prune_interval)]
        self.used = self.get_edge_size()
        self.num_ops -= len(dhs)
        if self.num_ops == 0:
            self.zero = Zero(stride=self.dim[-1])
        return dhs
    
    def update_num_ops(self):
        self.num_ops = len([op for op in self.ops if not op.zero])
        if self.num_ops == 0:
            self.zero = Zero(stride=self.dim[-1])

    def get_growth(self):
        all_none_grad = all([x is None for x in self.growth_factor['grad']])
        all_none_weight = all([x is None for x in self.growth_factor['weight']])
   
        
        if 1: #len(self.growth_factor['weight']) == 0 or all_none_grad or all_none_weight:
            return {
                'std_weight': None,
                'std_grad': None,
                'mean_weight':None,
                'mean_grad': None,
                'abs_std_weight': None,
                'abs_std_grad': None,
                'abs_mean_weight': None,
                'abs_mean_grad': None,
                'shap': None
            }
        else:
            adj = 1
            
            weights = np.array(self.growth_factor['weight'])
            out = {
                'random': None,
                'mean_sum': np.mean(np.sum(weights.reshape(-1, self.prev_num_ops), 1)[-100:]),
                'mean_mean': np.mean(np.mean(weights.reshape(-1, self.prev_num_ops), 1)[-100:]),
                'std_weight': np.std(self.growth_factor['weight'])/adj,
                'std_grad': np.std(self.growth_factor['grad'])/adj,
                'mean_weight': np.mean(self.growth_factor['weight'])/adj,
                'mean_grad': np.mean(self.growth_factor['grad'])/adj,
                'abs_std_weight': np.std(np.abs(self.growth_factor['weight'])),
                'abs_std_grad': np.std(np.abs(self.growth_factor['grad'])),
                'abs_mean_weight': np.mean(np.abs(self.growth_factor['weight'])),
                'abs_mean_grad': np.mean(np.abs(self.growth_factor['grad'])),
                'shap': self.shap
            }
            return {k:(v,None) if v is not None else v for k,v in out.items()}

    def reset_growth_factor(self):
        self.growth_factor = {'weight': [], 'grad': []}

    def get_edge_size(self):
        return sum([op.pruner.mem_size for op in self.ops])

    def __repr__(self):
        return "Edge {}".format(self.name, self.op_sizes)

    def forward(self, x, drop_prob):
        if self.num_ops == 0:
            return [self.zero(x)]
        else:
            out = []
            for op in self.ops:
                if op.name == 'Zero':
                    continue
                elif op.name == 'Identity':
                    out.append(op(x))
                else:
                    out.append(drop_path(op(x), drop_prob))
            return out


class Cell(nn.Module):
    def __init__(self, cell_idx, input_dims, op_sizes,  prune=True, data_index=0):
        super().__init__()
        self.cell_idx = cell_idx
        self.name = str(cell_idx)
        self.edge_counter = Count()
        self.input_dims = input_dims[::]
        self.op_sizes = op_sizes
        self.data_index = data_index
        self.prune = prune
        self.n_nodes = len(self.input_dims)+2
        self.op_keys = list(self.op_sizes.keys())

        self.edges = nn.ModuleDict()
        self.bns = nn.ModuleDict()
        self.scalers = nn.ModuleDict()
        target_dim = list(self.op_sizes.keys())[0]
        self.inputs = list(range(len(self.input_dims)))
        
        for i in range(self.n_nodes):
            #j = self.n_nodes
            for j in range(i+1, self.n_nodes):
                if i in self.inputs and j in self.inputs:
                    continue
                if i in self.inputs and j != len(self.inputs):
                    continue
                self.edges[encode(i, j)] = Edge(ops=None,
                                                dim=target_dim,
                                                stride=2 if i in self.inputs and cell_idx else 1,
                                                op_size=self.op_sizes,
                                                name=self.edge_counter(),
                                                prune=prune,
                                                data_index=self.data_index)
                self.bns[str(i)] = nn.BatchNorm2d(list(self.op_sizes.keys())[0][1])
                self.bns[str(j)] = nn.BatchNorm2d(list(self.op_sizes.keys())[0][1])
        
        for i in range(len(input_dims)):
            stride = input_dims[i][-2]//target_dim[-2]
            self.scalers[str(i)] = MinimumIdentity(input_dims[i][1], target_dim[1], stride=stride)

        self.output_node = 0
        self.forward_iterator = []
        self.set_op_order()

    def get_new_op_sizes(self, orig):
        if orig.dim[-1] == 2:
            return self.op_keys[-1], orig.dim
        else:
            return orig.dim, orig.dim
        
    def split_edge(self, key, device=torch.device("cpu"), data_index=0, verbose=True):       
        i, j = decode(key)
        if verbose:
            print("Mutating",self.name, key)
        new_sizes = self.get_new_op_sizes(self.edges[key])
        shifted_edges = []
        shifted_bns = []
        edge_ops = [op.name for op in self.edges[key].ops if not op.zero]
        new_ops = {k: v for k, v in commons.items()}

        cond_i = lambda x, y: x >= j
        cond_j = lambda x, y: y >= j
        edge_a = Edge(ops=new_ops,
                      dim=new_sizes[1],
                      stride=2 if i in self.inputs and self.cell_idx else 1, 
                      op_size=self.op_sizes,
                      name=self.edge_counter(),
                      data_index=data_index,
                      prune=self.prune,
                      lineage=self.edges[key].lineage + [self.edges[key].name],
                      original=False)
        edge_b = Edge(ops=new_ops,
                      dim=new_sizes[0],
                      stride=1,
                      op_size=self.op_sizes,
                      name=self.edge_counter(),
                      data_index=data_index,
                      prune=self.prune,
                      lineage=self.edges[key].lineage + [self.edges[key].name],
                     original=False)
        edge_c = self.edges[key]
        new_edges = [edge_a, edge_b]
        new_keys = [encode(i, j), encode(j, j+1)]
        shifted_edges.append([encode(i, j), edge_a])
        shifted_edges.append([encode(j, j+1), edge_b])
        shifted_edges.append([encode(i, j+1), edge_c])
        
#         for tgt_key, tgt_edge in self.edges.items():
#             tgt_i, tgt_j = decode(tgt_key)
#             if tgt_j == i and j - tgt_i <= 3:
#                 new_sizes = self.get_new_op_sizes(self.edges[tgt_key])
#                 edge = Edge(ops=new_ops,
#                             dim=new_sizes[1],
#                             stride=2 if tgt_i in self.inputs and self.cell_idx else 1,
#                             op_size=self.op_sizes,
#                             name=self.edge_counter(),
#                             data_index=data_index,
#                             prune=self.prune,
#                             lineage=self.edges[tgt_key].lineage + [self.edges[tgt_key].name])
#                 shifted_edges.append([encode(tgt_i, j), edge])
#                 new_edges.append(edge)
                
        for tgt_key, tgt_edge in self.edges.items():
            tgt_i, tgt_j = decode(tgt_key)
            prei, prej = tgt_i, tgt_j
            if cond_i(tgt_i, tgt_j):
                tgt_i += 1
            if cond_j(tgt_i, tgt_j):
                tgt_j += 1
            shifted_bns.append([str(tgt_i), self.bns[str(prei)], prei])
            shifted_bns.append([str(tgt_j), self.bns[str(prej)], prej])
            
            if tgt_key == key:
                continue
            shifted_edges.append([encode(tgt_i, tgt_j), tgt_edge])
           
        shifted_bns.append([str(j), nn.BatchNorm2d(new_sizes[0][1]), None])
            
        edge_c.reset_growth_factor()
        self.edges = nn.ModuleDict()
        for k, edge in shifted_edges:
            self.edges[k] = edge
        for k, bn, _ in shifted_bns:
            self.bns[k] = bn

        self.edges = self.edges.to(device)
        self.bns = self.bns.to(device)
        self.set_op_order()
        return new_keys

    def set_op_order(self):
        node_outputs = {}
        for key in self.edges.keys():
            i, j = decode(key)

            if j > self.output_node:
                self.output_node = j

            if node_outputs.get(i) is None:
                node_outputs[i] = []
            node_outputs[i].append([key, j])
        node_order = sorted(node_outputs.keys())

        self.forward_iterator = []
        for node in node_order:
            self.forward_iterator.append([node, str(node), node_outputs[node], False])
        self.forward_iterator.append([self.output_node, str(self.output_node), None, True])

    def plot_cell(self, subgraph=None, color_by='growth', **kwargs):
        g = Digraph() if subgraph is None else subgraph
        if color_by == 'op':
            colors = color_create()

        for key, edge in self.edges.items():
            i, j = decode(key)
            i_str = kwargs.get('prefix', "") + encode(self.name, i, "_")
            j_str = kwargs.get('prefix', "") + encode(self.name, j, "_")
            g.node(i_str, label=str(i), color='#0000ffa0' if i<len(self.inputs) else 'white')
            g.node(j_str, label=str(j), color='#0000ffa0' if j<len(self.inputs) else 'white')

            if color_by == 'op':
                for op in edge.ops:
                    if not op.zero:
                        g.edge(i_str, j_str, color=colors[op.name]['hex'])
            elif color_by in ['attrition', 'growth', 'apl']:
                if color_by == 'attrition':
                    cmap = plt.get_cmap('Reds_r')
                    attrition = (len(commons) - edge.num_ops) / len(commons)
                    color = rgb_to_hex(cmap(attrition))
                    label = str(edge.num_ops)
                elif color_by == 'growth':
                    cmap = plt.get_cmap('Reds_r')
                    color = rgb_to_hex(cmap(kwargs['norm'](edge.get_growth())))
                    label = str(edge.get_growth())
                else:
                    cmap = plt.get_cmap('Reds')
                    label = str(edge.avg_path_length)
                    color = rgb_to_hex(cmap(kwargs['norm'](edge.avg_path_length)))
                
                    
                g.edge(i_str, j_str,
                       color=color,
                       label=label,
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

    def forward(self, xs, drop_prob, edge_toggles):
        raw_node_ins = {i:[] for i,_,_,_ in self.forward_iterator}
        #raw_node_ins[0] = [self.scalers[str(len(xs)-1)](xs[-1])]
        
        for i,x in enumerate(xs):
            raw_node_ins[i] = [self.scalers[str(i)](x)]
        #raw_node_ins[1] = [self.scalers[str(i)](x) for i,x in enumerate(xs)]
        
        for i, i_str, js, last in self.forward_iterator:
            rnis = raw_node_ins[i]
            
            if not len(rnis):
                continue
            node_ins = sum(rnis[1:], rnis[0]) if len(rnis) > 1 else rnis[0]
            node_in = self.bns[i_str](node_ins)

            #del raw_node_ins[i]
            if last:
                return node_in
            
            for edge, j in js:
                if edge_toggles[edge]:
                    raw_node_ins[j] += self.edges[edge](node_in, drop_prob)
            

class Net(nn.Module):
    def __init__(self, hypers):
        super().__init__()
        wipe_output()
        self.input_dim = hypers['input_dim']
        self.out_classes = hypers['dataset']['classes']
        self.reductions = hypers['reductions']
        self.scale = hypers['scale']
        self.prune = hypers.get('prune', True)
        self.drop_prob = hypers['drop_prob']
        self.gpu_space = hypers['gpu_space']
        self.n_nodes = hypers.get('nodes', 1)
        self.model_id = hypers.get('model_id', namer())
        self.data_index = 0
        self.device = torch.device(hypers['device'])
        self.epoch = 0

        self.hypers = hypers
        self.initializer = initializer(self.input_dim[1], self.scale, ksize=3) 
        
        init_dim = channel_mod(self.input_dim, self.scale)
        self.dims = [cw_mod(init_dim, 2**i) for i in range(self.reductions+1)]
        self.dims = [[d+(1,), channel_mod(d, d[1]*2)+(2,)] if i != len(self.dims)-1 else [d+(1,)]
                     for i, d in enumerate(self.dims)]
        self.dims = [d for dim in self.dims for d in dim]

        # get all operation sizes
        size_set = compute_sizes()
        op_match = (len(size_set) > 0) and all([op in list(size_set.values())[0].keys() for op in commons])
        if not op_match or not all([dim in size_set for dim in self.dims]):
            size_set = compute_sizes(self.dims)
        size_set = {dim: {k: v for k, v in ops.items() if k in commons} for dim, ops in size_set.items()}
        self.size_set = size_set

        # build cells
        self.scalers = nn.ModuleDict()
        self.towers = nn.ModuleDict()
        self.cells = nn.ModuleList()
        
        input_dims = [list(init_dim)]
        for cell_idx in range(self.reductions+1):
            cell_dims = self.dims[max(0, cell_idx * 2 - 1):cell_idx * 2 + 1]
            cell_sizes = {d: size_set[d] for d in cell_dims}
            self.cells.append(Cell(cell_idx, input_dims, cell_sizes, prune=self.prune))
            input_dims.append(cw_mod(input_dims[-1],2) if cell_idx else input_dims[-1])
            self.towers[str(cell_idx)] = Classifier(str(cell_idx), True, cell_dims[0], self.out_classes)
        self.mut_sizes = {}
        self.set_cell_node_lengths()
        self.update_edge_toggles()
    
    def deadhead(self, prune_epochs, prune_interval):
        old_params = general_num_params(self)
        deadheads = 0
        deadhead_spots = []
        for i, cell in enumerate(self.cells):
            for key in cell.edges.keys():
                dhs = cell.edges[key].deadhead(prune_epochs, prune_interval)
                deadheads += len(dhs)
                for dh in dhs:
                    deadhead_spots.append([i, key, dh])

        self.log_print("Deadheaded {} operations".format(deadheads))
        self.jn_print("Deadheaded {}".format(deadhead_spots))
        self.log_print("Param Delta: {:,} -> {:,}".format(old_params, general_num_params(self)))
        self.clear_deadends()
        clean("Deadhead", verbose=False)

    def update_mut_sizes(self):
        for i, cell in enumerate(self.cells):
            self.mut_sizes[i] = {}
            edges_to = {}
            for k, e in cell.edges.items():
                self.mut_sizes[i][k] = e.get_edge_size()/1024       
                tgt_i, tgt_j = decode(k)
                if edges_to.get(tgt_j) is None:
                    edges_to[tgt_j] = 0
                edges_to[tgt_j] += 1
            self.mut_sizes[i] = {k: v * (2 + edges_to.get(decode(k)[0], 0)) for k,v in self.mut_sizes[i].items()}
            #self.mut_sizes[i] = {k: v * 2 for k,v in self.mut_sizes[i].items()}

    def update_edge_toggles(self):
        edge_toggles = []
        for cell in self.cells:
            for k, edge in cell.edges.items():
                edge_toggles.append(1)
        edge_toggles = np.array(edge_toggles)
        self.edge_toggles = edge_toggles
        return edge_toggles
         
    def get_edge_toggle_labels(self):
        i = 0
        labels = {}
        for cell_idx, cell in enumerate(self.cells):
            labels[cell_idx] = {}
            for k, edge in cell.edges.items():
                labels[cell_idx][k] = self.edge_toggles[i]
                i += 1
        return labels
        
    def compile_growth_factors(self):
        for cell in self.cells:
            for key, edge in cell.edges.items():
                [op.log_analytics() for op in edge.ops]
                gfs = [op.get_growth_factor() for op in edge.ops if not op.zero]
                edge.growth_factor['grad'] += [gf['grad'] for gf in gfs]
                edge.growth_factor['weight'] += [gf['weight'] for gf in gfs]

    def compile_pruner_stats(self):
        for cell in self.cells:
            for key, edge in cell.edges.items():
                [op.pruner.track_gates() for op in edge.ops]
                [op.pruner.clamp() for op in edge.ops]

    def get_n_edges(self):
        idx = 0
        for cell in self.cells:
            for k, edge in cell.edges.items():
                idx += 1
        return idx

    def get_growth_factors(self):
        factors = {}
        for i, cell in enumerate(self.cells):
            for k, edge in cell.edges.items():
                factors[(str(i), k)] = edge.get_growth()
        return factors

    def clear_grad(self):
        for cell in self.cells:
            for e in cell.edges.values():
                e.reset_growth_factor()

    def check_mutation(self, cell, edge):
        size, overfill = size_test(self)
        self.update_mut_sizes()
        mut_size = self.mut_sizes[cell][edge]
        return not overfill and (size + mut_size) < self.gpu_space
                
    def mutate(self, n=1, verbose=True, execute=True):
        mutations = []
        new_edges = []
        mutation_in_cell = set()

        for i in range(n):
            growth_factors = self.get_growth_factors()
            growth_factors = {k:v for k,v in growth_factors.items() if k[0] not in mutation_in_cell}
            
            pre = {}
            if verbose:
                 for k, v in growth_factors.items():
                     pre[k] = v[self.mut_metric['metric']]
            if self.mut_metric['metric'] == 'random' or all(v[self.mut_metric['metric']] is None for v in growth_factors.values()):
                edges = [edge for edge in growth_factors.keys() if edge not in mutation_in_cell]
                loc = np.random.choice(np.arange(len(edges)), replace=False)
            else:
                growth_factors = {k: v[self.mut_metric['metric']] for k, v in growth_factors.items()}
                nonnull_growth_factors = [(k, (v,a)) for k, (v,a) in growth_factors.items() if v is not None]
                rank1 = rank([x[1][0] for x in nonnull_growth_factors])
                #rank2 = rank([x[1][1] for x in nonnull_growth_factors])
                rank1 = rank(np.array(rank1), max_at_rank0=self.mut_metric['sort_dir'] == 'max')
                #rank2 = rank(np.array(rank2), max_at_rank0=False)
                
                
                nonnull_ranks = [(k, rank1[i]) for i, (k,_) in enumerate(nonnull_growth_factors)]

                if verbose:
                    for k, v in nonnull_ranks:
                        print(k, pre[k], v)
                edges = sorted(nonnull_ranks,
                               key=lambda x: x[1],
                               reverse=False)
                edges += [(k, v) for k, v in growth_factors.items() if v is None]
                edges = [x[0] for x in edges]
                loc = 0
            cell, edge = edges[loc]
            mutation_in_cell.add(cell)
            cell = int(cell)

            if not execute:
                print('choosing',edges[loc])
                return None
            
            if self.check_mutation(cell, edge):
                edges = self.cells[cell].split_edge(edge, self.device, self.data_index)
                self.cells[cell].edges[edge].reset_growth_factor()
                new_edges += edges
                mutations.append((cell, edge))

        if len(new_edges):
            self.clear_grad()
            self.set_cell_node_lengths()
        return mutations, new_edges

    def plot_network(self, color_by='growth'):
        super_g = Digraph(graph_attr={'nodesep': '.09'})
        if color_by == 'growth':
            max_grad = max([edge.get_growth()[self.mut_metric['metric']] for cell in self.cells for k, edge in cell.edges.items()])
            norm = lambda x: x/max_grad if max_grad != 0 else 1
        elif color_by == 'apl':
            max_apl = max([edge.avg_path_length for cell in self.cells for k, edge in cell.edges.items()])
            norm = lambda x: x/max_apl if max_apl != 0 else 1
        else:
            norm = None

        for i, cell in enumerate(self.cells):
            with super_g.subgraph(name='cluster_{}'.format(i)) as c:
                c.attr(style='filled', color='grey')
                c.attr(label='Cell {}'.format(i))
                cell.plot_cell(subgraph=c, color_by=color_by, norm=norm, prefix='')
                c.node_attr.update(style='filled', color='white')
#             if i==0:
#                 break
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
        if 0:
            super_g = self.plot_network()
        else:
            super_g = None
        analytics = {}
        name_to_key = {}
        for cell_idx, cell in enumerate(self.cells):
            if not analytics.get(cell_idx):
                analytics[cell_idx] = {}

            for k, edge in sorted(cell.edges.items(), key=lambda x: x[1].name):
                analytics[cell_idx][edge.name] = {'key': k,
                                                  'analytics': {op.name: op.analytics for op in edge.ops},
                                                  'lineage': edge.lineage}

        out_str = self.__str__()

        with open('pickles/analytics_{}'.format(self.model_id), "wb") as f:
            pkl.dump([super_g, analytics, name_to_key, out_str], f)

    def set_pruners(self, state):
        for i, cell in enumerate(self.cells):
            self.mut_sizes[str(i)] = {}
            for k, e in cell.edges.items():
                for op in e.ops:
                    op.pruner.prune = state

    def reset_parameters(self):
        self.data_index = 0
        self.epoch = 0
        warn_non_resets = []
        diff_non_resets = []
        self.clear_grad()
        for module in self.modules():
            if type(module) != type(self):
                if 'reset_parameters' in dir(module):
                    module.reset_parameters()
                else:
                    if 'parameters' in dir(module):
                        n_params = general_num_params(module)
                        child_params = sum([general_num_params(m) for m in module.children()])

                        if n_params != 0 and n_params != child_params:
                            diff_non_resets.append([type(module).__name__, n_params])
                    else:
                        warn_non_resets.append(type(module).__name__)

        if len(diff_non_resets):
            error_string = "\n".join(["\t* {}: {:,} parameter(s)".format(m, p) for m, p in diff_non_resets])
            raise AttributeError(
                "{} module(s) have differentiable parameters without a 'reset_parameters' function: \n {}".format(
                    len(diff_non_resets),
                    error_string))
        if len(warn_non_resets):
            warning_msg = "Model contains modules without 'reset_parameters' attribute: "
            warnings.warn(warning_msg + str(set(warn_non_resets)))

    def set_cell_node_lengths(self):
        cell_node_lengths = []
        for cell in self.cells:
            node_lengths = {i: [] for i, _, _, _ in cell.forward_iterator}

            for i,cnl in enumerate(cell_node_lengths):
                node_lengths[i+1] += cnl[max(cnl.keys())]
        
            for i, _, js, _ in cell.forward_iterator:
                if js:
                    for _, j in js:
                        if node_lengths[i]:
                            node_lengths[j] += [nl+1 for nl in node_lengths[i]]
                        else:
                            node_lengths[j] += [1]                         
                            
            for i, _, js, _ in cell.forward_iterator:
                if js:
                    for k, _ in js:
                        cell.edges[k].path_lengths = np.array(node_lengths[i])
                        cell.edges[k].avg_path_length = np.mean(cell.edges[k].path_lengths+1) if len(cell.edges[k].path_lengths) else 1
            cell_node_lengths.append(node_lengths)
     
    def clear_deadends(self):
        for cell in self.cells:
            # find nodes that have no path to the end
            node_end_path = {i: False for i, _, _, _ in cell.forward_iterator}
            for i, _, js, last in cell.forward_iterator[::-1]:
                if last:
                    node_end_path[i] = True
                elif js:
                    for edge, j in js:
                        if node_end_path[j] and edge in cell.edges and cell.edges[edge].num_ops:
                            node_end_path[i] = True
            orphaned_nodes = [i for i, connected in node_end_path.items() if not connected]

            # find nodes with no path to any input
            node_start_path = {i: False for i, _, _, _ in cell.forward_iterator}
            for i, _, js, last in cell.forward_iterator:
                if i<len(cell.inputs):
                    node_start_path[i] = True
                if js:
                    for edge, j in js:
                        if node_start_path[i] and edge in cell.edges and cell.edges[edge].num_ops:
                            node_start_path[j] = True            
            orphaned_nodes += [i for i, connected in node_start_path.items() if not connected]

            # delete orphaned nodes and all edges leading to/from them
            for on in orphaned_nodes:
                if str(on) in cell.bns:
                    del cell.bns[str(on)]
            for k, edge in [(k,v) for k,v in cell.edges.items()]:
                i,j = decode(k)
                if k in cell.edges and (i in orphaned_nodes or j in orphaned_nodes):
                    del cell.edges[k]

            # update cell iterators
            new_fw_it = []
            for i, i_str, js, last in cell.forward_iterator:
                new_js = []
                if js:
                    for edge, j in js:
                        if j not in orphaned_nodes and edge in cell.edges and cell.edges[edge].num_ops:
                            new_js.append([edge, j])
                if i not in orphaned_nodes:
                    new_fw_it.append([i, i_str, new_js, last])
            cell.forward_iterator = new_fw_it
    
    def upsize(self, new_batch, new_scale):
        self.reset_parameters()
        scaling_factor = new_scale/self.scale
        new_batch = 64
        self.initializer = initializer(3, new_scale, ksize=3) 
        for cell in self.cells:
            for k, edge in cell.edges.items():
                i,j = decode(k)
                for op in edge.ops:
                    if not op.zero:
                        op.op = op.op_function(new_scale, op.stride)
                cell.bns[str(i)] = nn.BatchNorm2d(new_scale)
                cell.bns[str(j)] = nn.BatchNorm2d(new_scale)

            target_dim = list(cell.op_sizes.keys())[0]
            for i in range(len(cell.input_dims)): 
                stride = cell.input_dims[i][-2]//target_dim[-2]
                cell.scalers[str(i)] = MinimumIdentity(int(cell.input_dims[i][1]*scaling_factor),
                                                       int(target_dim[1]*scaling_factor),
                                                       stride=stride)

            cell_dim = list(self.dims[max(0, cell.cell_idx * 2 - 1):cell.cell_idx * 2 + 1][0])
            cell_dim[0] = new_batch
            cell_dim[1] = new_scale
            self.towers[str(cell.cell_idx)] = Classifier(cell.cell_idx, True, cell_dim, self.out_classes)
            new_scale *= 2
    
    
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

        out += out_format(l="Initializer", p=general_num_params(self.initializer))
        for i, cell in enumerate(self.cells):
            out += cell.__repr__(out_format)
            if str(i) in self.towers.keys():
                name = 'Classifier' if i == max([int(x) for x in self.towers.keys()]) else 'Aux Tower'
                out += out_format(l=" {} {}".format(arrow_char, name),
                                  p=general_num_params(self.towers[str(i)]))
        if 'Classifier' in self.towers:
            out += out_format(l=" {} Classifier".format(arrow_char),
                              p=general_num_params(self.towers['Classifier']))
        out += spacer.format("")
        out += out_format(l="Total", p=general_num_params(self))
        out += spacer.format("")
        return out

    def forward(self, x, drop_prob=0., aux=True, verbose=False):
        outputs = []
        
        if x.shape[1] != 1:
            xs = [self.initializer(x)]
        else:
            xs = [x]
        
        for i, cell in enumerate(self.cells):
            x = cell(xs, drop_prob, self.get_edge_toggle_labels()[i])
            outputs.append(self.towers[str(i)](x))
            if i != len(self.cells) - 1:
                xs.append(x)
        
        return outputs if aux else outputs[-1]