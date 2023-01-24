import numpy as np
import torch
from tqdm import tqdm

def eval_acc(net, loader, BatchLim=np.inf, Domain='Tgt', Text=""):
    net.eval()
    accs = []
    progress_bar = tqdm(enumerate(loader),desc=Text, bar_format = "{desc:20}{percentage:2.0f}{r_bar}")
    for ind, (src_img, src_label, tgt_img, tgt_label) in progress_bar:
        if ind > BatchLim:
            break
        if Domain=='Tgt':
            img = tgt_img
            label = tgt_label
        else:
            img = src_img
            label = src_label
        img = img.to(net.device, dtype=torch.float)
        label = label.to(net.device, dtype=torch.long)
        pred = net(img)
        _, idx = pred.max(dim=1)
        acc = (idx == label).sum().cpu().item() / len(idx)
        accs.append(acc)
    return accs

