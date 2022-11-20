import torch
import torch.nn as nn
import torch.nn.functional as F

class conLossv3(nn.Module):
    def __init__(self, temperature, log_of_sum=False, use_l2=False):
        super(conLossv3, self).__init__()
        self.tau = temperature
        self.log_of_sum = log_of_sum

    def forward(self, x, y, labels):
        """
        The embeddings should be normalised
        """
        tgt = self.lbl_to_tgt(labels)
        sim_target = tgt['sim_target'].type_as(x)
        loss = self.loss(x, y, sim_target)
        return loss

    def loss(self, x, y, target):
        x, y = F.normalize(x), F.normalize(y)

        sim_mtrx = (x @ y.T) / self.tau

        sim = torch.exp(sim_mtrx)

        N = target * sim
        D = torch.sum((1 - target) * sim, dim=-1, keepdim=True)

        loss = N / D

        if self.log_of_sum:
            return -torch.log(torch.sum(loss))

        else:
            # sum_of_log
            loss[torch.where(loss == 0)] = 1  # so taking log doesn't throw error

            return -torch.sum(torch.log(loss))

    def lbl_to_tgt(self, labels):
        batch_size = len(labels)
        sim_target = torch.zeros(batch_size, batch_size)

        for i in range(batch_size):
            sim_target[i][torch.where(labels == labels[i])] = 1

        return {'sim_target': sim_target}
