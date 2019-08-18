from copy import deepcopy

import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
import torch.utils.data


def variable(t: torch.Tensor, use_cuda=True, **kwargs):
    if torch.cuda.is_available() and use_cuda:
        t = t.cuda()
    return Variable(t, **kwargs)


class Regularization(object):
    def __init__(self, model: nn.Module, dataset, mode):

        self.model = model
        self.dataset = dataset
        self.mode = mode

        if self.mode == "EWC":
            self.params = {n: p for n, p in self.model.named_parameters() if p.requires_grad}
            self._means = {}
            self._precision_matrices = self._diag_fisher()

            for n, p in deepcopy(self.params).items():
                self._means[n] = p.data

        if self.mode == "SI":
            NotImplementedError
        if self.mode == "MAS":
            NotImplementedError

    def _diag_fisher(self):
        precision_matrices = {}
        for n, p in deepcopy(self.params).items():
            p.data.zero_()
            precision_matrices[n] = p.data

        self.model.eval()
        for step,batch in enumerate(self.dataset):
            self.model.zero_grad()
            batch = tuple(t.to(device) for t in batch)
            input_ids, lm_labels= batch
            self.model(batch.input_ids,labels=lm_labels)
            loss.backward()

            for n, p in self.model.named_parameters():
                if p.grad is None:
                    continue
                grad = p.grad.data
                precision_matrices[n].data += p.grad.data ** 2 / len(self.dataset)

        precision_matrices = {n: p for n, p in precision_matrices.items()}
        return precision_matrices

    def penalty(self, model: nn.Module):
        loss = 0
        for n, p in model.named_parameters():
            _loss = self._precision_matrices[n] * (p - self._means[n]) ** 2
            loss += _loss.sum()
        return loss

