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
        outputs = model.forward(data, model.drop_prob, verbose=verbose)
        model.data_index += 1

        def loss_f(x): return kwargs['criterion'](x, target)
        losses = [loss_f(output) for output in outputs[:-1]]
        final_loss = loss_f(outputs[-1])
        loss = final_loss + .2 * sum(losses)

        # end train step ======================
        loss = loss/multiplier
        loss.backward()

        model.compile_growth_factors()
        model.compile_pruner_stats()
        if (batch_idx % multiplier == 0) or (batch_idx == len(train_loader) - 1):
            kwargs['optimizer'].step()
        corr, div = top_k_accuracy(outputs[-1], target, top_k=kwargs.get('top_k', [1]))
        corrects = corrects + corr
        divisor += div

        # mid epoch updates ===================
        if print_or_end:
            cache = cache_stats(human_readable=False)
            prog_str = 'Train Epoch: {:<3} [{:<6}/{:<6} ({:.0f}%)]\t'.format(
                kwargs['epoch'],
                (batch_idx + 1) * len(data),
                len(train_loader.dataset),
                100. * (batch_idx + 1) / len(train_loader))
            prog_str += 'Per Epoch: {:<7}, '.format(show_time((time.time() - batch_start) * len(train_loader)))
            prog_str += 'Alloc: {}, '.format(sizeof_fmt(cache, spacing=True))
            prog_str += 'Data T: {:<6.3f}, Op T: {:<6.3f}'.format(t_cumul_data, t_cumul_ops)
            jn_print(prog_str, end="\r", flush=True)

        if batch_idx > kwargs.get('kill_at', np.inf):
            break
        t_cumul_ops += (time.time()-t_op_start)
        t_data_start = time.time()

    # === output ===============
    jn_print(prog_str)
    for i, c, cell in model.all_cells():
        print("== {},{} ==".format(c, i))
        for key, edge in cell.edges.items():
            print(" ", key, edge.get_growth())

    accuracy_string("Train", corrects, divisor, epoch_start, top_k, comp_ratio=None)


def test(model, device, top_k=[1]):
    # === tracking stats =====================
    test_loader = model.data[1]
    corrects, e_corrects = np.zeros(len(top_k)), np.zeros(len(top_k))
    divisor = 0
    t_start = time.time()

    # === test epoch =========================
    model.eval()
    outputs, targets, metas = [], [], []
    with torch.no_grad():
        for batch_idx, data in enumerate(test_loader):
            if len(data)==3:
                data, metadata, target = data
            else:
                data, target = data
                metadata = None
            data, target = data.to(device), target.to(device)
            output = model.forward(data, drop_prob=0)[-1]
            corr, div = top_k_accuracy(output, target, top_k=top_k)
            corrects = corrects + corr
            outputs.append(torch.argmax(output, 1).tolist())
            targets.append(target)
            metas.append(metadata)
            divisor += div

    # === format results =====================
    return accuracy_string("Last Tower Test ", corrects, divisor, t_start, top_k, return_str=True), outputs, targets, metas


def size_test(model, verbose=False):
    # run a few batches through the model to get an estimate of its GPU size
    try:
        start_size = cache_stats(False)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        criterion = nn.CrossEntropyLoss()

        model.train()
        for batch_idx, data in enumerate(model.data[0]):
            if len(data)==3:
                data, _, target = data
            else:
                data, target = data
            data, target = data.to(device), target.to(device)
            out = model.forward(data, drop_prob=.3, verbose=(verbose and batch_idx == 0))
            loss = criterion(out[-1], target)
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
            #model = model.to(torch.device('cpu'))
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


def sp_size_test(model):
    torch.save(model, "pickles/sp_size_test.pt")
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sizer.py')
    python = get_python3()
    try:
        s = subprocess.check_output([python, path], stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as e:
        print("Failed")
        print(python, path)
        print(e.output.decode('utf8'))
        raise e
    with open("pickles/size_test_out.pkl", "rb") as f:
        return pkl.load(f)


# === FULL TRAINING HANDLER=============================================================================================
def full_train(model, kwargs):
    # === learning handlers ==================
    model = model.to(kwargs['device'])

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

    # === run n epochs =======================
    epochs = kwargs['lr_schedule']['T']
    model.data_index = 0
    last_mutation = -1

    for epoch in range(0, epochs):
        training_logger.info("=== EPOCH {} ===".format(epoch))

        # train =========================
        train(model, criterion=criterion, optimizer=optimizer, epoch=epoch, **kwargs)
        out_str, outputs, targets, metas = test(model, kwargs['device'], top_k=kwargs.get('top_k', [1]))
        log_print(out_str)
        model.epoch += 1
        model.save_analytics()

        # prune ==============================
        if model.prune:
            model.deadhead(kwargs['mod_interval']*len(model.data[0]))

        # mutate
        if epoch and (epoch-last_mutation) % kwargs['mod_interval'] == 0 and kwargs.get('mutate', True):
            mutations, new_edges = model.mutate(n=kwargs['n_mutations'])
            if len(mutations):
                last_mutation = epoch
            print("Performed {} mutations: {}".format(len(mutations), mutations))
            lr_schedulers[epoch] = LRScheduler(epochs - epoch, kwargs['lr_schedule']['lr_max'])
            new_params = [param for edge in new_edges for param in edge.parameters()]
            optimizer.add_param_group({'params': new_params,
                                       'lr': kwargs['lr_schedule']['lr_max'],
                                       'key': epoch})
            print()

        # anneal =============================
        set_lr(optimizer, lr_schedulers)

    return outputs, targets, metas
