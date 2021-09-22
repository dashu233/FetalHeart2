import numpy as np
import matplotlib.pyplot as plt
import torch
import matplotlib
VERBOSE = False
if not VERBOSE:
    matplotlib.use('Agg')

def draw_pic(pic_name,data:torch.Tensor,mean=0,var=1,target=None,output=None):
    if isinstance(data, torch.Tensor):
        ys = data.view(-1).detach().cpu().numpy()
    else:
        ys = data
    ys = mean + var * ys
    xs = np.array(range(len(ys)))
    VT = 30

    plt.figure(figsize=(20, 5))

    yticks = [i * 30 + 60 for i in range(6)]
    dense_yticks = [i * 10 + 60 for i in range(16)]
    ylabels = []
    for dy in dense_yticks:
        if dy in yticks:
            ylabels.append(str(dy))
        else:
            ylabels.append("")
    plt.yticks(dense_yticks, ylabels)

    xticks = [i * 240 for i in range(21)]
    dense_xticks = [i * 120 for i in range(41)]
    xlabels = []
    for dx in dense_xticks:
        if dx in xticks:
            xlabels.append(str(dx))
        else:
            xlabels.append("")
    plt.xticks(dense_xticks, xlabels)
    plt.ylabel("BPM")
    plt.xlabel("Time")
    plt.ylim(bottom=60, top=210)
    plt.xlim(left=0, right=4800)
    plt.plot(xs, ys, 'b')
    if target is not None:
        plt.text(4400, 6 * VT, 'target:{:.2f}'.format(target.item()))
    if output is not None:
        plt.text(4400, 5 * VT, 'output:{:.2f}'.format(output.item()))
    plt.savefig(pic_name, bbox_inches='tight', dpi=200)
    plt.close('all')

def draw_pic_clip(pic_name,data:torch.Tensor,mean=0,var=1,target=None,output=None):
    if isinstance(data, torch.Tensor):
        ys = data.view(-1).detach().cpu().numpy()
    else:
        ys = data
    ys = mean + var * ys
    xs = np.array(range(len(ys)))
    VT = 30

    plt.figure(figsize=(5, 5))

    yticks = [i * 30 + 60 for i in range(6)]
    dense_yticks = [i * 10 + 60 for i in range(16)]
    ylabels = []
    for dy in dense_yticks:
        if dy in yticks:
            ylabels.append(str(dy))
        else:
            ylabels.append("")
    plt.yticks(dense_yticks, ylabels)

    xticks = [i * 10 for i in range(11)]
    dense_xticks = [i * 2 for i in range(51)]
    xlabels = []
    for dx in dense_xticks:
        if dx in xticks:
            xlabels.append(str(dx))
        else:
            xlabels.append("")
    plt.xticks(dense_xticks, xlabels)
    plt.ylabel("BPM")
    plt.xlabel("Time")
    plt.ylim(bottom=60, top=210)
    plt.xlim(left=0, right=100)
    plt.plot(xs, ys, 'b')
    if target is not None:
        plt.text(80, 6 * VT, 'target:{:.2f}'.format(target.item()))
    if output is not None:
        plt.text(80, 5 * VT, 'output:{:.2f}'.format(output.item()))
    plt.savefig(pic_name, bbox_inches='tight', dpi=200)
    plt.close('all')

def draw_cam(pic_name,data:torch.Tensor,region:torch.Tensor,
             mean=0,var=1,target=None,output=None,verbose=False):
    if isinstance(data, torch.Tensor):
        ys = data.view(-1).detach().cpu().numpy()
    else:
        ys = data
    ys = mean + var * ys
    xs = np.array(range(len(ys)))
    VT = 30

    plt.figure(figsize=(20, 5))
    ax1 = plt.subplot(211)
    yticks = [i * 30 + 60 for i in range(6)]
    dense_yticks = [i * 10 + 60 for i in range(16)]
    ylabels = []
    for dy in dense_yticks:
        if dy in yticks:
            ylabels.append(str(dy))
        else:
            ylabels.append("")

    ax1.set_yticks(dense_yticks)
    ax1.set_yticklabels(ylabels)

    xticks = [i * 240 for i in range(21)]
    dense_xticks = [i * 120 for i in range(41)]
    xlabels = []
    for dx in dense_xticks:
        if dx in xticks:
            xlabels.append(str(dx))
        else:
            xlabels.append("")

    ax1.set_xticks(dense_xticks)
    ax1.set_xticklabels(xlabels)

    ax1.set_ylabel("BPM")
    ax1.set_xlabel("Time")
    ax1.set_ylim(bottom=60, top=210)
    ax1.set_xlim(left=0, right=4800)
    ax1.plot(xs, ys, 'b')
    if target is not None:
        ax1.text(4400, 6 * VT, 'target:{:.2f}'.format(target.item()))
    if output is not None:
        ax1.text(4400, 5 * VT, 'output:{:.2f}'.format(output.item()))

    ax2 = plt.subplot(212, sharex=ax1)

    ax2.set_ylabel("Attr")
    ax2.set_xlabel("Time")

    if isinstance(region,torch.Tensor):
        ys = region.view(-1).detach().cpu().numpy()
    else:
        ys = region
    xs = np.array(range(len(ys)))
    ax2.plot(xs, ys, 'b')

    plt.savefig(pic_name, bbox_inches='tight', dpi=200)
    if verbose:
        plt.show()
    plt.close('all')

def draw_cam_double(pic_name,data,region:torch.Tensor,region2:torch.Tensor,
             mean=0,var=1,target=None,output=None):
    assert data.size(0) == 1, 'please make sure input data batch is 1'

    # draw data
    if isinstance(data,torch.Tensor):
        ys = data.view(-1).detach().cpu().numpy()
    else:
        ys = data
    ys = mean + var * ys
    xs = np.array(range(len(ys)))
    VT = 30

    plt.figure(figsize=(20, 5))
    ax1 = plt.subplot(311)
    yticks = [i * 30 + 60 for i in range(6)]
    dense_yticks = [i * 10 + 60 for i in range(16)]
    ylabels = []
    for dy in dense_yticks:
        if dy in yticks:
            ylabels.append(str(dy))
        else:
            ylabels.append("")

    ax1.set_yticks(dense_yticks)
    ax1.set_yticklabels(ylabels)

    xticks = [i * 240 for i in range(21)]
    dense_xticks = [i * 120 for i in range(41)]
    xlabels = []
    for dx in dense_xticks:
        if dx in xticks:
            xlabels.append(str(dx))
        else:
            xlabels.append("")

    ax1.set_xticks(dense_xticks)
    ax1.set_xticklabels(xlabels)

    ax1.set_ylabel("BPM")
    ax1.set_xlabel("Time")
    ax1.set_ylim(bottom=60, top=210)
    ax1.set_xlim(left=0, right=4800)
    ax1.plot(xs, ys, 'b')
    if target is not None:
        ax1.text(4400, 6 * VT, 'target:{:.2f}'.format(target.item()))
    if output is not None:
        ax1.text(4400, 5 * VT, 'output:{:.2f}'.format(output.item()))

    ax2 = plt.subplot(312, sharex=ax1)

    ax2.set_ylabel("Attr")
    ax2.set_xlabel("Time")
    if isinstance(region,torch.Tensor):
        ys = region.view(-1).detach().cpu().numpy()
    else:
        ys = region
    xs = np.array(range(len(ys)))
    ax2.plot(xs, ys, 'b')

    ax3 = plt.subplot(313, sharex=ax1)

    ax3.set_ylabel("fmap")
    ax3.set_xlabel("Time")


    if isinstance(region2,torch.Tensor):
        ys = region2.view(-1).detach().cpu().numpy()
    else:
        ys = region2
    xs = np.array(range(len(ys)))
    ax3.plot(xs, ys, 'b')

    plt.savefig(pic_name, bbox_inches='tight', dpi=200)
    plt.close('all')