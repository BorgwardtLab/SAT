# -*- coding: utf-8 -*-
import numpy as np
import torch
from sklearn.metrics import confusion_matrix


def accuracy_SBM(scores, targets):
    targets = targets.cpu().numpy()
    scores = scores.argmax(dim=-1).cpu().numpy()
    return torch.from_numpy(confusion_matrix(targets, scores).astype('float32'))
