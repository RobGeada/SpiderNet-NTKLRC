import datetime
from IPython.display import display
import time

import torch.nn as nn
import torch.optim as optim

from spider_net.helpers import *

# set up logging
setup_logger("training_logger", filename='logs/trainer.log', mode='a', terminator="\n")
setup_logger("jn_out", filename='logs/jn_out.log', mode='a', terminator="")
training_logger = logging.getLogger('training_logger')
jn_out = logging.getLogger('jn_out')

log_print = log_print_curry([training_logger, jn_out])
jn_print  = log_print_curry([jn_out])


# === EPOCH LEVEL FUNCTIONS ============================================================================================
def set_lr(optimizer, annealers):
    new_lrs = []
    for param_group in optimizer.param_groups:
        new_lr = annealers[param_group['key']].step()
        param_group['lr'] = new_lr
        new_lrs.append(new_lr)
    log_print("\n\x1b[31mAdjusting lrs to {}\x1b[0m".format(new_lrs))


def extract_curr_lambdas(comp_lambdas, epoch, tries):
    out = None
    if comp_lambdas:
        if epoch >= comp_lambdas['transition']:
            out = {k: v*2**tries for k, v in comp_lambdas['lambdas'].items()}
    return out


# === CUSTOM LOSS FUNCTIONS ============================================================================================
dummy_zero = torch.tensor([0.], device='cuda')


def compression_loss(model, comp_lambda, comp_ratio, item_output=False):
    # edge pruning
    if comp_lambda['edge']>0:
        prune_sizes = []
        for cell in model.cells:
            edge_pruners = [op.pruner.mem_size*torch.clamp(op.pruner.sg(),0.,1.) if op.pruner else dummy_zero\
                                for key, edge in cell.edges.items() for op in edge.ops]
            prune_sizes += [torch.sum(torch.cat(edge_pruners)).view(-1)]
        edge_comp_ratio = torch.div(torch.cat(prune_sizes), model.edge_sizes)
        edge_comp = torch.norm(comp_ratio - edge_comp_ratio)
        edge_loss = comp_lambda['edge'] * edge_comp
    else:
        edge_loss = 0

    # input pruning
    if comp_lambda['input']>0:
        input_sizes = []
        for cell in model.cells:
            input_pruners = [torch.clamp(pruner.sg(),0,1) if pruner else dummy_zero for pruner in cell.input_handler.pruners]
            input_sizes += [torch.sum(torch.cat(input_pruners)).view(-1)]
        input_comp_ratio = torch.div(torch.cat(input_sizes), model.input_p_tot)
        input_comp = torch.norm(1/model.input_p_tot - input_comp_ratio)
        input_loss = comp_lambda['input']*input_comp
    else:
        input_loss = 0

    loss = edge_loss+input_loss
    if item_output:
        ecr = 0 if edge_loss == 0 else torch.mean(edge_comp_ratio).item()
        icr = 0 if input_loss == 0 else torch.mean(input_comp_ratio).item()
        return loss, [ecr,icr], [edge_loss,input_loss]
    else:
        return loss, [None,None], [None,None]


# === PERFORMANCE METRICS ==============================================================================================
def top_k_accuracy(output, target, top_k):
    if len(output.shape)==2:
        output = output.reshape(list(output.shape)+[1,1])
        target = target.reshape(list(target.shape)+[1,1])
    correct = np.zeros(len(top_k))
    _, pred = output.topk(max(top_k), 1, True, True)
    for i,k in enumerate(top_k):
        target_expand = target.unsqueeze(1).repeat(1,k,1,1)
        equal = torch.max(pred[:,:k,:,:].eq(target_expand),1)[0]
        correct[i] = torch.sum(equal)
    return correct, len(target.view(-1))


def accuracy_string(prefix, corrects, divisor, t_start, top_k, comp_ratio=None, return_str=False):
    corrects = 100. * corrects / float(divisor)
    out_string = "{} Corrects: ".format(prefix)
    for i, k in enumerate(top_k):
        out_string += 'Top-{}: {:.2f}%, '.format(k, corrects[i])
    if comp_ratio is not None:
        out_string += 'Comp: {:.2f}, {:.2f} '.format(*comp_ratio)
    out_string += show_time(time.time() - t_start)
    
    if return_str:
        return out_string
    else:
        log_print(out_string)


# === BASE LEVEL TRAIN AND TEST FUNCTIONS===============================================================================
def train(model, device, **kwargs):
    # === tracking stats ======================
    top_k = kwargs.get('top_k', [1])
    corrects = np.zeros(len(top_k), dtype=float)
    divisor = 0
    div = 0

    # === train epoch =========================
    model.train()
    train_loader = model.data[0]
    epoch_start = time.time()
    multiplier = kwargs.get('multiplier', 1)
    jn_print(datetime.datetime.now().strftime("%m/%d/%Y %I:%M %p"))

    t_data_start = None
    t_cumul_data, t_cumul_ops = 0,0
    for batch_idx, data in enumerate(train_loader):
        if len(data)==3:
            data, _, target = data
        else:
            data, target = data
        t_data_end = time.time()
        if t_data_start is not None:
            t_cumul_data += (t_data_end-t_data_start)
        t_op_start = time.time()

        print_or_end = (not batch_idx % 10) or (batch_idx == len(train_loader)-1)
        batch_start = time.time()

        # pass data ===========================
        data, target = data.to(device), target.to(device)
        if (batch_idx % multiplier == 0) or (batch_idx == len(train_loader)-1):
            kwargs['optimizer'].zero_grad()

        verbose = kwargs['epoch'] == 0 and batch_idx == 0
        output = model.forward(data, model.drop_prob, verbose=verbose)
        loss = kwargs['criterion'](output, target)

        # end train step ======================
        loss = loss/multiplier
        loss.backward()
        model.compile_grads()
        if (batch_idx % multiplier == 0) or (batch_idx == len(train_loader) - 1):
            kwargs['optimizer'].step()
        corr, div = top_k_accuracy(output, target, top_k=kwargs.get('top_k', [1]))
        corrects = corrects + corr
        divisor += div

        # mid epoch updates ===================
        if print_or_end:
            prog_str = 'Train Epoch: {:<3} [{:<6}/{:<6} ({:.0f}%)]\t'.format(
                kwargs['epoch'],
                (batch_idx + 1) * len(data),
                len(train_loader.dataset),
                100. * (batch_idx + 1) / len(train_loader))
            prog_str += 'Per Epoch: {:<7}, '.format(show_time((time.time() - batch_start) * len(train_loader)))
            prog_str += 'Alloc: {}, '.format(cache_stats(spacing=False))
            prog_str += 'Data T: {:<6.3f}, Op T: {:<6.3f}'.format(t_cumul_data,t_cumul_ops)
            jn_print(prog_str, end="\r", flush=True)

        if batch_idx > kwargs.get('kill_at', np.inf):
            break
        t_cumul_ops += (time.time()-t_op_start)
        t_data_start = time.time()

    # === output ===============
    jn_print(prog_str)
    accuracy_string("Train", corrects, divisor, epoch_start, top_k, comp_ratio=None)


def test(model, device, top_k=[1]):
    # === tracking stats =====================
    test_loader = model.data[1]
    corrects, e_corrects = np.zeros(len(top_k)), np.zeros(len(top_k))
    divisor = 0
    t_start = time.time()

    # === test epoch =========================
    model.eval()
    with torch.no_grad():
        for batch_idx, data in enumerate(test_loader):
            if len(data)==3:
                data, _, target = data
            else:
                data, target = data
            data, target = data.to(device), target.to(device)
            output = model.forward(data, drop_prob=0)
            corr, div = top_k_accuracy(output, target, top_k=top_k)
            corrects = corrects + corr
            divisor += div

    # === format results =====================
    return accuracy_string("Last Tower Test ", corrects, divisor, t_start, top_k, return_str=True)


def size_test(model, verbose=False):
    # run a few batches through the model to get an estimate of its GPU size
    try:
        start_size = cache_stats(False)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        criterion = nn.CrossEntropyLoss()

        model.train()
        for batch_idx, (data, target) in enumerate(model.data[0]):
            data, target = data.to(device), target.to(device)
            out = model.forward(data, drop_prob=.3, verbose=(verbose and batch_idx == 0))
            loss = criterion(out, target)
            loss.backward()
            if batch_idx > 2:
                break
        overflow = False
        model.zero_grad()
        size = (cache_stats(False) - start_size) / 1024 / 1024 / 1024
        clean(verbose=False)

    except RuntimeError as e:
        if 'CUDA out of memory' in str(e):
            overflow = True
            model = model.to(torch.device('cpu'))
            model.zero_grad()
            size = cache_stats(False)/(1024**3)
            clean(verbose=False)
            del model
            try:
                del data, target
            except:
                pass
            clean(verbose=False)
        else:
            raise e
    return size, overflow


def sp_size_test(n, e_c, add_pattern, prune=True,**kwargs):
    with open("pickles/size_test_in.pkl","wb") as f:
        pkl.dump([n, e_c, add_pattern, prune, kwargs],f)
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sizer.py')
    python = get_python3()
    try:
        s = subprocess.check_output([python, path], stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as e:
        print("Failed")
        print(python, path)
        print(e.output.decode('utf8'))
        raise e
    if kwargs.get('print_model',False):
        print(s.decode('utf8'))
    with open("pickles/size_test_out.pkl","rb") as f:
        return pkl.load(f)


# === FULL TRAINING HANDLER=============================================================================================
def full_train(model, kwargs):
    # === learning handlers ==================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    optimizer = optim.SGD([{'params': model.parameters(), 'lr':kwargs['lr_schedule']['lr_max'], 'key': 0}],
                          momentum=.9,
                          weight_decay=3e-4)
    lr_schedulers = {0: LRScheduler(kwargs['lr_schedule']['T'], kwargs['lr_schedule']['lr_max'])}


    model.jn_print, model.log_print = jn_print, log_print
    print("=== Training {} ===".format(model.model_id))

    # === init logging =======================
    training_logger.info("=== NEW FULL TRAIN ===")
    log_print("Starting at {}".format(datetime.datetime.now()))
    training_logger.info(model.creation_string())
    criterion = nn.CrossEntropyLoss()

    if torch.cuda.is_available():
        model.cuda()
        criterion.cuda()

    # === run n epochs =======================
    epochs = kwargs['lr_schedule']['T']
    met_thresh = False
    for epoch in range(0, epochs):
        training_logger.info("=== EPOCH {} ===".format(epoch))

        # train =========================
        train(model, device, criterion=criterion, optimizer=optimizer, epoch=epoch, **kwargs)

        # prune ==============================
        if model.prune:
            model.eval()
            edge_pruners = [op.pruner for cell in model.cells \
                            for key, edge in cell.edges.items() for op in edge.ops if op.pruner]
            [pruner.track_gates() for pruner in edge_pruners]
            model.deadhead(kwargs['nas_schedule']['prune_interval'])

        # mutate
        if (epoch+1) % kwargs['mutate_schedule']['mutate_interval'] == 0 and epoch>0:
            #display(model.plot_network(color_by='grad'))
            mutations, new_edges = model.mutate(n=kwargs['mutate_schedule']['n_mutations'])
            print("Performed {} mutations: {}".format(len(mutations), mutations))
            lr_schedulers[epoch] = LRScheduler(epochs - epoch, kwargs['lr_schedule']['lr_max'])
            new_params = [param for edge in new_edges for param in edge.parameters()]
            optimizer.add_param_group({'params': new_params,
                                       'lr': kwargs['lr_schedule']['lr_max'],
                                       'key': epoch})

        # test ===============================
        log_print(test(model, device, top_k=kwargs.get('top_k', [1])))
        print()

        # anneal =============================
        set_lr(optimizer, lr_schedulers)

        if met_thresh and epochs is None:
            break 
    return met_thresh
