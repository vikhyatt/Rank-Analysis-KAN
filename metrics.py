import matplotlib.pyplot as plt
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
from tqdm import tqdm
from torch import nn
from fastkan import FastKAN as KAN
from fastkan import FastKANLayer as KANLinear
from functools import partial
from einops.layers.torch import Rearrange, Reduce
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


def generate_plots(model, dataloader, exp_name):
    model.eval()
    classes = [f"Class {i}" for i in range(100)]
    preds = []
    actual = []
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    with torch.no_grad():
        for images, labels in dataloader:
            output = model(images.to(device))
            preds.append(output.argmax(dim=1))
            actual.append(labels.to(device))
    
    preds = torch.cat(preds).cpu()
    actual = torch.cat(actual).cpu()
    
    cm = confusion_matrix(actual, preds , normalize = 'true')
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=classes)
    fig, ax = plt.subplots(figsize=(10,10))
    disp.plot(ax=ax)
    plt.title(f'{exp_name}')
    plt.savefig(f'confusion_matrix_{exp_name}.jpg', bbox_inches = 'tight', dpi = 200)
    