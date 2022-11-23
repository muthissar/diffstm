from dstm.util import add_centered
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from dstm.util.constants import Constants
import torch
from pathlib import Path
from dstm.model.baseline import confidence_interval
import matplotlib
from dstm.util.constants import Constants
import tqdm
plt.rcParams.update(Constants.matplotlib_rcparams)

from dstm.util import add_centered
#fontsize = 16
#titlefontsize = 20

def plot_piano_roll(piano_roll, colors = ["w","g"],time_sig="4/4"):
    bounds = np.arange(len(colors)+1)
    bounds = np.array(bounds) -0.5
    cmap = ListedColormap(colors)
    norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)
    fig, ax = plt.subplots(figsize=(Constants.text_height-0.1, Constants.text_height/4))
    ax.set_yticks(np.arange(0.5, 48, 12))
    ax.set_yticklabels(np.arange(48, 96, 12))
    if time_sig == "4/4":
        ticks_pr_2 = 16*2
    elif time_sig == "3/4":
        ticks_pr_2 = 12*2
    else:
        raise NotImplementedError("time signature not implimented.")
    x_ticks = np.arange(-0.5, len(piano_roll), ticks_pr_2)
    ax.set_xticks(x_ticks)
    
    ax.set_xticklabels((x_ticks + 0.5).astype(int))
    ax.xaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator(n=ticks_pr_2))
    ax.yaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator(n=12))
    ax.grid(which='both', linewidth=0.1)

    plt.imshow(piano_roll.T, cmap=cmap, norm=norm, interpolation="none", origin="lower", aspect="equal")
    plt.xlabel("time ($t$)")
    plt.ylabel("\\textit{pitch}")
    fig.tight_layout()
    return fig, ax
def plot_prediction(label, prediction, d, window=(None, None), time_sig="4/4"):
    eye = np.eye(d)
    preds_one_hot = eye[:, prediction]
    summed = preds_one_hot + 2*eye[:, label]
    colors = ['w', 'r', 'g', 'g']
    piano_roll = summed[:, slice(window[0], window[1])].T
    fig, ax = plot_piano_roll(piano_roll, colors,time_sig=time_sig)
    plt.colorbar(ticks=[1., 2.], shrink=0.2, boundaries=[.5, 1.5, 2.5], format=matplotlib.ticker.FuncFormatter(lambda val, pos: "Error" if val == 1. else "True"), orientation="horizontal",pad=0.3)
    return fig, ax


def plot_prediction_dmc(mc, sample, window=(None,
 None), time_sig="4/4"):
    with torch.no_grad():
        sample = sample.unsqueeze(0)
        probs = mc.eval().probs(sample)
        preds = probs[0].argmax(-1).cpu().squeeze()
        d = mc.d
        return plot_prediction(sample.argmax(-1)[0].detach().cpu().numpy(), preds, d, window, time_sig=time_sig)

def plot_saliency_map(mc, piano_roll, time, output):
    # freeze weights
    piano_roll = piano_roll.unsqueeze(0)
    for param in mc.parameters():
        param.requires_grad = False
    piano_roll.requires_grad = True
    out1 = mc.probs(piano_roll, hard=False)[0, time, output]
    piano_roll.grad = None
    out1.backward()
    plt.matshow(piano_roll.grad.data[0][:time].T)

def compute_aggregate_saliency_map(mc, data_loader, path="out/session/large/results", device="cpu", method="log", model="Continuous", relative_pitch=True):
    # freeze weights
    for param in mc.parameters():
        param.requires_grad = False
    r = 2**(len(mc.hparams.hparams.filters) + 1)
    if relative_pitch:
        height_saliency_map = mc.d * 2 -1
    else:
        height_saliency_map = mc.d
    
    saliency_map = torch.zeros(r, height_saliency_map).to(device)
    for batch_id, (batch, lengths) in enumerate(tqdm.tqdm(data_loader, desc=f'Computing aggregate saliency map {model}')):
        if batch.shape[1] == 0:
            continue
        batch = batch.to(device)
        batch.requires_grad = True
        if method == "log":
            probs, _ = torch.log(mc.probs(batch))
        elif method == "normal":
            probs, _  = mc.probs(batch)
        else:
            raise NotImplementedError("method {} is not defined")
        n_grads_pr_piece = 10
        # def jac(batch):
        #     probs, _ = mc.probs(batch)
        #     max = probs.max(-1)
        #     time_indexes = torch.randperm(batch.shape[1])
        #     #ziped_indexes = torch.cat([torch.arange(batch.).unsqueeze(0), indexes.unsqueeze(0)]).tolist()
        #     return max[:,time_indexes[:n_grads_pr_piece]]
        # jac = torch.autograd.functional.jacobian(jac, batch)
        # if relative_pitch:
        #     pitch_index = slice(mc.d-output-1, 2*mc.d - output-1)
        # else:
        #     pitch_index = slice(None)
        # saliency_map[max((r-time), 0):, pitch_index] += jac.sum((0,1))[max(0, time-r):time, :]
        for prob, length in zip(probs, lengths):

           
            #time_pitch = set([])
            # for i in range(n_grads_pr_piece):
            #     while True :
            #         time = np.random.random_integers(0, length-1)
            #         #output = np.random.random_integers(0, mc.d-1)
            #         #if (time, output) not in time_pitch:
            #         if time not in time_pitch:
            #             break
            #     #time_pitch.add((time,output))
            #     time_pitch.add((time))
            for time in np.random.choice(np.arange(0, length, 1), min(n_grads_pr_piece, length.item()), replace=False):
                #pick always the note which is predicted
                output = prob[time].argmax(-1)
                out1 = prob[time, output]
                batch.grad = None
                out1.backward(retain_graph=True)
                summed = torch.abs(batch.grad.data).sum(axis=0)
                add_centered(saliency_map, summed, pitch_relative_to=output,  time_relative_to=time, relative_pitch=relative_pitch)
            torch.save({"saliency_map": saliency_map.cpu(), "r":r, "d": mc.d}, "{}/saliency_map_{}_{}{}.p".format(path, method, model,"_rp" if relative_pitch else ""))
def plot_aggregate_saliency_map(path="fig",method="log",model="Continuous", relative_pitch=True):
    data = torch.load("{}/saliency_map_{}_{}{}.p".format(path, method, model,"_rp" if relative_pitch else ""))
    r = data["r"]
    d = data["d"]
    saliency_map = data["saliency_map"]
    # for t_name, t, pdf_path in [("Sensitivity salency map (log transformed)",  torch.log, "{}/saliency_map_{}_t_{}_{}.pdf".format(path, method, "log", model)), 
    #     ("Sensitivity salency map",lambda x: x, "{}/saliency_map_{}_{}.pdf".format(path, method, model))]:
    #     fig, ax = plt.subplots(1, 1)
    #     plt.imshow(t(saliency_map.T).cpu()) 
    #     #plt.imshow(saliency_map.T.cpu())
    #     ax.set_xticks(list(range(0, r-1, int(r/(16*1)))))
    #     ax.set_xticklabels(list(range(r,1, int(-r/(16*1)))))
    #     #plt.title(t_name, fontsize=titlefontsize)
    #     #plt.xlabel("\\textit{time lag}", fontsize=fontsize)
    #     #plt.ylabel("\\textit{relative pitch}", fontsize=fontsize)
    #     plt.title(t_name)
    #     plt.xlabel("\\textit{time lag}")
    #     plt.ylabel("\\textit{relative pitch}")
    #     y_ticks = np.arange(0, 2*(d) - 1, 16)
    #     ax.set_yticks(y_ticks)
    #     ax.set_yticklabels(y_ticks - d + 1)
    #     #fig.savefig(pdf_path)
    #fig, ax = plt.subplots(figsize=(5.78, 4 ))
    #fig, ax = plt.subplots(figsize=(6, 2.5 ))
    fig, ax = plt.subplots(figsize=(Constants.text_width, Constants.text_width/2))
    #ax.xaxis.tick_top()
    # #plt.imshow(saliency_map.T.cpu())
    ax.set_xticks(list(range(0, r-1, int(r/(16*1)))))
    ax.set_xticklabels(list(range(r,1, int(-r/(16*1)))))
    #ax.xaxis.grid(True, which='minor')
    ax.xaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())
    ax.yaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())
    #ax.tick_params(which='both', width=2)
    ax.tick_params(which='major', length=7)
    ax.tick_params(which='minor', length=4)
    
    # ax.set_xticks( np.arange(int(r/16-1), r, int(r/(16*1))) )
    # ax.set_xticklabels(np.arange(int(r - r/16) + 1, 0, int(-r/(16*1))))
    
    #plt.title(t_name, fontsize=titlefontsize)
    #plt.xlabel("\\textit{time lag}", fontsize=fontsize)
    #plt.ylabel("\\textit{relative pitch}", fontsize=fontsize)
    plt.title("\\textbf{{{} sensitivity}}".format(Constants.styles[model]['name'][:-6]))
    #plt.title("DCSTM sensitivity".format(model))
    plt.xlabel("time lag")
    if relative_pitch:
        plt.ylabel("relative pitch")
        y_ticks = np.arange(0, 2*(d) - 1, 24)
        ax.set_yticks(y_ticks)
        ax.set_yticklabels(y_ticks - d + 1)
    else:
        plt.ylabel("\\textit{absolute pitch}")
        y_ticks = np.arange(1, d, 12)
        ax.set_yticks(y_ticks)
        ax.set_yticklabels(y_ticks+ 1)
    im = ax.imshow(saliency_map.T.cpu() + 1e-5, cmap="jet", interpolation=None, norm=matplotlib.colors.LogNorm(),origin="lower" )
    fig.tight_layout()
    
    fig.colorbar(im, orientation="horizontal",pad=0.25)
    #ax.set_xlim(511-290,511-280)
    fig.show()
    pdf_path = "fig/saliency_map_{}_{}{}.pdf".format(method, model, "_rp" if relative_pitch else "")
    fig.savefig(pdf_path)

def p_confidence_interval(ps, n):
    bounds = list(map(lambda p: confidence_interval(p, n)[1] - p, ps))
    plt.errorbar(x = list(range(len(ps))),
        y=np.array([ps]).T,
        yerr=np.array(bounds),
        capsize=5.0,
        marker='o',
        linestyle=""
    )