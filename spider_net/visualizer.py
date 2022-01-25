from IPython.display import clear_output
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import gridspec
import numpy as np
import time
import re
from spider_net.helpers import show_time
from scipy.optimize import curve_fit
import sys

# === SOME OPTIONAL CONFIGS ============================================================================================
try:
    from pyautogui import hotkey

    resize_default = True
except (ImportError, KeyError) as e:
    resize_default = False

plt.style.reload_library()
try:
    plt.style.use('material')
except OSError:
    pass


# === HELPERS ==========================================================================================================
def time_parse(t):
    if "m" in t:
        return int(t.split("m")[0]) * 60 + int(t.split(",")[1].split('s')[0])
    else:
        return float(t.split('s')[0])


# === DATA LOADING =====================================================================================================
path = ""


def get_prune_logs():
    with open(path + "logs/jn_out.log", "r") as f:
        out = f.read().replace('\x00', '')
    return out.split("\n")


def get_raw_runs():
    with open(path + "logs/trainer.log") as f:
        data = f.read()
    runs = data.split('=== NEW FULL TRAIN ===')
    return [run for run in runs if 'Starting at' in run]


# === PRUNE VISUALIZATION ==============================================================================================
def scrape(prog=False):
    rows = []
    row = {}
    aim, target = 0, 0
    logs = get_prune_logs()
    curr_prog = int([log for log in logs if '%' in log and 'Train Epoch' in log][-1].split("(")[1].split("%")[0])
    if prog:
        return curr_prog

    for line in logs:
        if line == "" and 'Test LT Acc' in row:
            row['Aim Comp'] = aim
            row['Target Comp'] = target
            if row:
                rows.append(row)
            row = {}
        if 'Adjusting lr' in line:
            #lrs = line.split("to [")[1].split("]\x1b[0m")[0]
            row['Learning Rate'] = None#[float(x.strip()) for x in lrs.split(",")]
        if 'Target Comp' in line:
            target = float(line.split(":")[1].split(",")[0])
            if 'Aim' in line:
                aim = float(line.split(":")[-1])
        if 'Train Epoch' in line:
            if 'Loss:' in line:
                row['C Loss'] = float(line.split(":")[2].split(",")[0])
            row['Alloc'] = float(line.split(":")[-1].split("GiB")[0].replace("|carr_ret|", ""))
        if 'Train Corrects' in line:
            row['Train Acc'] = float(line.split(":")[2].split(",")[0][:-1])
            if 'Comp' in line:
                row['Edge Comp'] = float(line.split(":")[3].split(",")[0])
                row['Input Comp'] = float(line.split(":")[3].split(" ")[2])
            row['Runtime'] = time_parse(line.split(" ")[-1].strip())
        if 'Hard Comp' in line:
            row['Soft Comp'] = float(line.split(':')[1].split(",")[0].strip())
            row['Hard Comp'] = float(line.split(':')[2].strip())
        if 'Train Loss Components' in line:
            row['C Loss'] = float(line.split(":")[2].split(",")[0])
            row['E Loss'] = float(line.split(":")[3].split(",")[0])
            row['I Loss'] = float(line.split(":")[4].split(",")[0])
        if 'Test' in line and 'Corrects' in line:
            if 'All Towers' in line:
                row['Test AT Acc'] = float(line.split(":")[2].split("%")[0])
            elif 'Last Tower' in line:
                row['Test LT Acc'] = float(line.split(":")[2].split("%")[0])
    return pd.DataFrame(rows), curr_prog


class PruneAnimator:
    def __init__(self, axes, col_sets):
        self.axes = axes
        self.col_sets = col_sets
        self.prog = 0

    def animate(self, i):
        df, prog = scrape()
        cycles = []
        if 'E_Loss' in list(df):
            e_losses = df[df['E_Loss'].isnull()].index
            cycles = [x for i, x in enumerate(e_losses) if x != e_losses[i - 1] + 1]

        for i, cols in enumerate(self.col_sets):
            dmin, dmax = 1000, 0
            self.axes[i].clear()
            labeled = True
            for label in cols:
                if label not in list(df):
                    labeled = False
                    continue
                self.axes[i].plot(df[label], label=label)
                if df[label].min(skipna=True) < dmin:
                    dmin = df[label].min(skipna=True)
                if df[label].max(skipna=True) > dmax:
                    dmax = df[label].max(skipna=True)
            for cycle in cycles:
                self.axes[i].plot([cycle] * 100, np.linspace(dmin, dmax, 100), c='k', alpha=.25)
            if labeled:
                self.axes[i].legend(fontsize=7)
            if any(['Acc' in c for c in cols]):
                self.axes[i].set_yticks(np.arange(np.floor(dmin / 10) * 10, np.ceil(dmax / 10) * 10, 10))
            self.axes[i].set_xticks(np.arange(0, np.ceil(len(df) / 10) * 10, 10))
            self.axes[i].set_xlim(-1, np.ceil(len(df) / 10) * 10)
            self.axes[i].set_title(", ".join(cols), fontsize=7)

        if prog != self.prog:
            self.axes[-1].clear()
            self.axes[-1].barh(1, prog, align='center', color='#FFCB6B')
            self.axes[-1].set_title("Epoch {} Progress ({:.0f}%)".format(len(df), prog), fontsize=7)
            self.axes[-1].set_yticks([])
            self.axes[-1].set_xticks(range(0, 110, 10))
            self.prog = prog
        return self.axes


# === TRAIN SCRAPING ===================================================================================================
def accuracy(raw_line, prefix, run_details):
    top1 = "{} Top-1".format(prefix)

    if run_details.get(top1) is None:
        run_details[top1] = []
    if 'Top-1' in raw_line:
        run_details[top1].append(float(raw_line.split(':', 2)[-1].split("%")[0]))
    if 'Top' not in raw_line:
        run_details[top1].append(float(raw_line.split(':', 1)[-1].split("%")[0]))
    return run_details


def time_proc(raw_line):
    try:
        raw_line = raw_line.replace("min", "m")
        raw_line = raw_line.replace(" m","m")
        raw_line = raw_line.replace("m, ", "m")
        raw_line = raw_line.replace("m,", "m")
        raw_line = raw_line.replace(" s", "s")
        if 'Time' in raw_line:
            run_time = raw_line.split("Time: ")[1]
        elif 'Comp' in raw_line:
            run_time = raw_line.split("Comp")[1]
            if len(run_time.split(" ")) == 5:
                run_time = "".join(raw_line.split(" ")[-2:])
            else:
                run_time = "".join([x for x in raw_line.split(" ")[-2:] if 'm' in x or 's' in x])
            run_time = run_time.replace(',', "")
        else:
            run_time = raw_line.split(", ")[-1]

        secs = 0
        if 'h' in run_time:
            h = int(run_time.split("h")[0]) * 60 * 60
            m = int(run_time.split("h")[1].split('m')[0]) * 60
            s = int(run_time.split("m")[1].split('s')[0])
            secs = h+m+s
        elif 'm' in run_time:
            if 'Comp' in run_time:
                if ' ' not in run_time:
                    run_time = run_time[-6:].replace(",", "")
                else:
                    run_time = run_time.split(" ")[-1]
            secs += 60 * int(run_time.split("m")[0])
            if 's' in run_time:
                secs += int(run_time.split("m")[1].split("s")[0])
        else:
            secs += float(run_time.split("s")[0])
        if '7m,52s' in raw_line:
            print(raw_line,run_time, secs)
        return secs
    except Exception as e:
        print(raw_line)
        print(run_time)
        raise e


def proc_run(run):
    run_details = {}
    curr_epoch = -1
    deadhead_history = []
    param_history = []
    epochs,epoch_times = [],[]
    a_loss, e_loss, i_loss = [], [], []

    prev_line = ""
    for raw_line in run.split("\n"):
        if raw_line == prev_line:
            prev_line = raw_line
            continue

        # add new values to histories
        if 'EPOCH' in raw_line:
            deadhead_history.append(0)
            param_history.append(0)
            epochs.append(int(raw_line.split("EPOCH")[1].split(" ===")[0]))

        # track model stats
        if 'Starting at' in raw_line:
            run_details['Start Time'] = raw_line.split("Starting at")[1]
        if 'Dim' in raw_line:
            if 'torch' in raw_line:
                raw_line = raw_line.replace(")", "").replace("torch.Size(", "")
            detail_str = re.split(',(?=\s[A-Za-z])', raw_line)
            new_str = ''
            for detail in detail_str:
                try:
                    k, v = detail.split(":")
                except Exception as e:
                    pass
                new_str += "'" + k.strip() + "':" + v
                if detail != detail_str[-1]:
                    new_str += ", "
            locals_ = locals()
            if '<' in new_str and '>' in new_str:
                new_str = new_str.split("<")[0] + 'None' + new_str.split(">")[-1]
            exec('details={' + new_str + "}", None, locals_)
            run_details.update(locals_['details'])

        # add accuracies
        if 'Train Corrects:' in raw_line:
            run_details = accuracy(raw_line, 'Train', run_details)
            if "," in raw_line and ('s'==raw_line[-1] or 'm'==raw_line[-1]):
                epoch_times.append(time_proc(raw_line))

        elif ('Last Towers Test' in raw_line and 'Corrects' in raw_line) or (
                'Test' in raw_line and 'Towers' not in raw_line and 'Corrects' in raw_line) or 'test acc' in raw_line:
            run_details = accuracy(raw_line, 'LT Test', run_details)
        elif ('All Towers Test' in raw_line and 'Corrects' in raw_line):
            run_details = accuracy(raw_line, 'AT Test', run_details)

        # track deadheading
        if 'Deadheaded' in raw_line:
            deadhead_history[-1] = -int(raw_line.split('Deadheaded')[-1].split("operations")[0])
        if 'Param Delta' in raw_line:
            param_history[-1] = [int(raw_line.split('Param Delta:')[-1].split("->")[0].replace(",", "")),
                                 int(raw_line.split('->')[-1].replace(",", ""))]

        # track loss_comps
        if 'Train Loss Components' in raw_line:
            loss_comps = raw_line.split(':')
            a, e, i = [float(loss.split(",")[0]) for loss in loss_comps[2:]]
            a_loss, e_loss, i_loss = a_loss + [a], e_loss + [e], i_loss + [i]
        if raw_line != "":
            prev_line = raw_line
    run_details['Loss Accuracy'] = a_loss
    run_details['Epoch_Times'] = epoch_times
    run_details['Total_Train_Time']= show_time(sum(epoch_times))
    run_details['Loss Edge'] = e_loss
    run_details['Loss Input'] = i_loss
    run_details['Epochs'] = epochs

    # flatten param hist
    if [x for x in param_history if type(x) is not int]:
        prev_val = [hist[0] for hist in param_history if hist != 0][0]
        new_param_hist = []
        for val in param_history:
            if val == 0:
                new_param_hist.append(prev_val)
            else:
                new_param_hist.append(val[1])
                prev_val = val[1]
        run_details['Params'] = new_param_hist
    run_details['Deadhead'] = deadhead_history
    return run_details


def proc_all_runs():
    runs = [proc_run(run) for run in get_raw_runs()]
    runs = pd.DataFrame(runs)
    for col in [col for col in list(runs) if 'Top' in col]:
        runs[col] = runs[col].apply(lambda x: x if type(x) == list else [])
    runs['LT Test Top-1 Max'] = runs['LT Test Top-1'].apply(lambda x: max(x, default=0))
    runs['Epoch'] = runs['Epochs'].apply(lambda x: x[-1] if len(x) else 0)
    return runs


# === TRAIN VISUALIZATION ==============================================================================================
class TrainAnimator:
    def __init__(self, axes, marker):
        self.axes = axes
        self.marker = marker
        self.prog = 0
        self.curr_epoch = 0

    def animate(self, i):
        runs = proc_all_runs()
        full_runs = runs[runs['Epochs'].apply(lambda x: len(x)>1 and max(x)>300)].sort_values(by='LT Test Top-1 Max', ascending=False)
        #full_runs = pd.concat([runs[852:858], runs[868:]])

        full_runs = full_runs.sort_values(by='LT Test Top-1 Max', ascending=False)
        #display(full_runs)
        compare = full_runs['LT Test Top-1'].values[0]
        compare_str = 'PR'

        lt = max(runs.iloc[-1]['LT Test Top-1'])
        lt_last = runs.iloc[-1]['LT Test Top-1'][-1]
        curr_run = runs.iloc[-1]
        epoch = runs.iloc[-1]['Epoch']
        cm = plt.cm.Spectral

        def smooth_max(l):
            out = [max(l[:i + 1]) for i, x in enumerate(l)]
            return out

        def fit_curve(xs, ys):
            start = 50

            if len(xs) > start:
                def func(x, a, b, c): return a + b * np.log(c * x)

                def func2(x, args): return func(x, args[0], args[1], args[2])

                curve = curve_fit(func,
                                  xs[start:],
                                  smooth_max(ys[start:]),
                                  bounds=[[-np.inf, -np.inf, .0001], np.inf])
                new_x = np.append(np.array(xs), np.arange(max(xs), 600))[start:]

                return new_x, func2(new_x, curve[0])
            return None, None

        if self.marker is not None:
            marker = full_runs[full_runs['ID'] == self.marker].iloc[0]['Start Time']
            labeled = full_runs.sort_values(by='Start Time',ascending=False)
            labeled = list(labeled[labeled['Start Time'].apply(lambda x: x>=marker)]['ID'])
        else:
            labeled = list(full_runs.sort_values(by='Start Time',ascending=False)[:3]['ID'])

        # plot previous runs
        w = 5, 15
        if epoch != self.curr_epoch:
            self.axes[0].clear()
            self.axes[1].clear()
            for i, (idx, run) in enumerate(full_runs.iterrows()):
                # print(runs.iloc[-1])
                if run['ID'] == curr_run['ID'] and run['Epoch']==curr_run['Epoch']:
                    continue
                ys = run['LT Test Top-1']
                if run['ID'] in labeled:
                    label = "{}, {}: {}".format(run['ID'],
                                                run['Start Time'],\
                                                max(run['LT Test Top-1']))
                else:
                    label = None
                xs = np.array(list(run['Epochs'])[:len(ys)])
                if min(xs) == 0 and max(xs) == 518:
                    xs += 82
                color = "#68099c" if run['ID'] == self.marker else cm(i / len(full_runs))
                self.axes[0].plot(xs,
                                  smooth_max(ys),
                                  color=color,
                                  alpha=(.75 if i == 0 else .5) + .05,
                                  label=label)
                    
        # plot current run
        if epoch < 0:
            print("No log yet...")
        else:
            if epoch != self.curr_epoch:
                curr = curr_run['LT Test Top-1'][-1]
                curr_max = max(curr_run['LT Test Top-1'])
                curr_arg_max = np.argmax(curr_run['LT Test Top-1'])
                if max(compare)<=curr_max:
                    rec, rec_max, rec_arg_max = curr, max(compare), np.argmax(compare)
                else:
                    rec, rec_max, rec_arg_max = compare[epoch], max(compare[:epoch + 1]), np.argmax(compare[:epoch + 1])

                text = "==== EPOCH {} ======================================\n".format(epoch)
                text += "LT Max: {}\n".format(lt)
                text += "LT Last: {}\n".format(lt_last)
                text += "Current Delta to {}:     {:> 2.2f}% ({}% vs {}%)\n".format(compare_str, curr - rec, curr, rec)
                text += "Current Delta to {} Max: {:> 2.2f}% ({}% @{} vs {}% @{})".format(compare_str,
                                                                                          curr_max - rec_max,
                                                                                          curr_max,
                                                                                          curr_arg_max,
                                                                                          rec_max,
                                                                                          rec_arg_max)
                yrange = 100 - min(curr_run['LT Test Top-1'][-10:])
                #self.axes[0].text(-30, 101 + yrange / 15, text, fontsize=8, fontfamily='monospace')
                self.axes[0].text(-1, 101 + yrange / 15, text, fontsize=8, fontfamily='monospace')
                ys = curr_run['LT Test Top-1']
                xs = list(curr_run['Epochs'])[:len(ys)]
                label = "{} {}: {}".format(curr_run['ID'],curr_run['Start Time'],max(curr_run['LT Test Top-1'])) 
                self.axes[0].plot(xs, ys, color='k', linewidth=1.5,label=label)
                x_f, y_f = fit_curve(xs, ys)
                if 0: #x_f is not None:
                    self.axes[0].plot(x_f, y_f, color='k', alpha=.5, linewidth=1.5, linestyle='--')
                
                # self.axes[0].plot(xs, ys, color='k', linewidth=1.5)
                self.axes[0].set_ylim(min(curr_run['LT Test Top-1'][-10:]) - 1, 100)
                self.axes[0].set_ylim(90, 100)
                self.axes[0].set_title("CIFAR-10 Loss History, SpiderNet")
                self.axes[0].set_xlabel("Epoch", fontsize=8)
                self.axes[0].set_ylabel("Accuracy", fontsize=8)
                self.axes[0].legend()
                ax_min, ax_max = min(50, int(min(curr_run['LT Test Top-1'][-10:]))), 100
                if ax_max - ax_min > 20:
                    div = 5
                    ax_min = (ax_min // div) * div
                elif ax_max - ax_min > 15:
                    div = 2.5
                    ax_min = (ax_min // div) * div
                else:
                    div = 1
                self.axes[0].set_ylim(ax_min, ax_max)
                self.axes[0].set_yticks(np.arange(ax_min, ax_max, div))
                self.axes[0].tick_params(axis='both', which='major', labelsize=8)
                self.curr_epoch = epoch
#                 plt.savefig('/home/campus.ncl.ac.uk/b6070424/Dropbox/PhD/monitoring.png',
#                             facecolor="#263238",
#                             bbox_inches="tight")

        prog = scrape(prog=True)
        if prog != self.prog:
            self.axes[-1].clear()
            self.axes[-1].barh(1, prog, align='center', color='#FFCB6B')
            self.axes[-1].set_xlim(0, 100)
            self.axes[-1].set_title("Epoch {} Progress ({}%)".format(epoch, prog), fontsize=7)
            self.axes[-1].set_yticks([])
            self.axes[-1].set_xticks(range(0, 110, 10))
            self.prog = prog
        return self.axes


# === PLOTTERS =========================================================================================================
def plot_monitor(fn):
    def monitor_loop(self, cleaner=clear_output):
        curr_progress = self.update_check()
        while 1:
            fn(self)
            new_prog = self.update_check()

            while new_prog == curr_progress:
                time.sleep(15)
                new_prog = self.update_check()
            cleaner()
            curr_progress = self.update_check()

    return monitor_loop


class PrunePlot:
    def __init__(self, figsize=(10, 4)):
        self.figsize = figsize
        self.col_sets = [
            ['Train Acc', 'Test LT Acc'],
            ['C Loss', 'E Loss', 'I Loss'],
            ['Hard Comp', 'Soft Comp', 'Aim Comp', 'Target Comp'],
            ['Input Comp']
        ]
        self.update_check = lambda: scrape(prog=True)

    def plot(self, resize=False):
        fig = plt.figure(figsize=self.figsize, dpi=125)
        gs = gridspec.GridSpec(len(self.col_sets) + 1, 1, height_ratios=[1] * len(self.col_sets) + [.25])
        axes = [plt.subplot(g) for g in gs]
        plt.subplots_adjust(left=.05, right=.95, top=.95, bottom=.05, hspace=.35)
        animator = PruneAnimator(axes, self.col_sets)
        ani = animation.FuncAnimation(fig, animator.animate, interval=1000)
        plt.draw()
        plt.pause(.001)
        if resize:
            # this is a dumb thing for my specific computer
            hotkey('winleft', 'ctrl', '6')
        plt.show()

    @plot_monitor
    def monitor(self):
        self.plot()


class TrainPlot:
    def __init__(self, marker=None, figsize=(10, 4)):
        self.figsize = figsize
        self.marker = marker
        self.update_check = lambda: len(scrape()[0])

    def plot(self, resize=False):
        fig = plt.figure(figsize=self.figsize, dpi=125)
        gs = gridspec.GridSpec(2, 1, height_ratios=[1, 1 / 15])
        axes = [plt.subplot(g) for g in gs]

        plt.subplots_adjust(left=.05, right=.95, top=.88, bottom=.05, hspace=.5)
        animator = TrainAnimator(axes=axes, marker=self.marker)
        ani = animation.FuncAnimation(fig, animator.animate, interval=1000)

        plt.draw()
        plt.pause(.001)

        if resize:
            # this is a dumb thing for my specific computer
            hotkey('winleft', 'ctrl', '6')
        plt.show()

    @plot_monitor
    def monitor(self):
        self.plot()


# === MAIN =============================================================================================================
if __name__ == '__main__':
    if sys.argv[1] == 'p':
        PrunePlot().plot(resize=resize_default)
    else:
        TrainPlot().plot(resize=resize_default)
