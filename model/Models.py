''' Define the Transformer model '''
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from model.Layers import EncoderLayer, DecoderLayer
import torch.nn.functional as F
import math
from model.TCN import TemporalConvNet
from entmax import entmax15,sparsemax
from torch.nn.functional import softmax
def get_pad_mask(seq, pad_idx):
    # print(seq.shape)
    return (seq != pad_idx)

def get_subsequent_mask(seq):
    ''' For masking out the subsequent info. '''
    sz_b, len_s, _ = seq.size()
    subsequent_mask = (1 - torch.triu(
        torch.ones((1, len_s, len_s), device=seq.device), diagonal=1)).bool()
    return subsequent_mask

class Transformer(nn.Module):
    ''' A sequence to sequence model with attention mechanism. '''

    def __init__(
            self, params, dataset):
        super().__init__()

        self.params = params
        self.dataset = dataset
        self.id_enc = None
        if self.dataset != 'Sanyo' and self.dataset != 'Hanergy':
            self.id_enc = torch.nn.Embedding(params.n_id, params.d_embedding, padding_idx=0)
        self.device = params.device

        self.tcn = TemporalConvNet(params.d_model, params.n_channel, kernel_size=params.d_kernel, dropout=params.dropout)
        self.linear = torch.nn.Linear(params.n_channel[-1]*(params.predict_start+1),params.n_channel[-1],bias=True)
        self.linear_1 = torch.nn.Linear(params.n_channel[-1]*2,1,bias=True)
        self.linear_2 = torch.nn.Linear(params.n_channel[-1]*2,1,bias=True)
        self.distribution_sigma = nn.Softplus()
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src_seq, trg_seq):
        if (self.dataset == 'Sanyo') or (self.dataset == 'Hanergy'):
            enc_id = None
            dec_id = None
        else:
            enc_id = src_seq[:,:,-1].type(torch.long)
            dec_id = trg_seq[:, :, -1].type(torch.long)
            src_seq = torch.cat((src_seq[:,:,:-1],self.id_enc(enc_id)),dim=-1)
            trg_seq = torch.cat((trg_seq[:,:,:-1],self.id_enc(dec_id)),dim=-1)
        srctrg_seq = torch.cat((src_seq, trg_seq), dim=1)
        in_conv = torch.zeros((srctrg_seq.shape[0]*self.params.predict_steps,srctrg_seq.shape[2],self.params.predict_start+1),device=self.device)
        for t in range(self.params.predict_steps):
            in_conv[t*srctrg_seq.shape[0]:(t+1)*srctrg_seq.shape[0]] = srctrg_seq[:,t:self.params.predict_start+t+1,:].transpose(-1, -2)
        output = self.tcn(in_conv).transpose(-1, -2)
        score = entmax15(torch.bmm(output[:,-1,:].unsqueeze_(1),output[:,:-1,:].transpose(-1,-2)))

        c = torch.bmm(score, output[:,:-1,:] ).squeeze_(1)
        concat_c = torch.cat([c, output[:,-1,:]], 1)
        y_hat = self.linear_1(concat_c).t().reshape(self.params.predict_steps,srctrg_seq.shape[0]).t()
        Sigma = self.linear_2(concat_c).t().reshape(self.params.predict_steps,srctrg_seq.shape[0]).t()
        Sigma = self.distribution_sigma(Sigma)

        return y_hat,Sigma

    def score(self,a,b):
        x = torch.bmm(a, b)
        x_exp = torch.exp(x).squeeze(1)
        x_exp_sum = torch.sum(x_exp, axis=1).unsqueeze(1)
        return torch.div(x_exp, x_exp_sum)

    def test(self, src_seq, trg_seq):

        src_mask = get_pad_mask(src_seq, -1)
        src_mask = None
        trg_mask = get_subsequent_mask(trg_seq)

        if (self.dataset == 'Sanyo') or (self.dataset == 'Hanergy'):
            enc_id = None
            dec_id = None
        else:
            enc_id = src_seq[:,:,-1].type(torch.long)
            dec_id = trg_seq[:, :, -1].type(torch.long)
            src_seq = torch.cat((src_seq[:,:,:-1],self.id_enc(enc_id)),dim=-1)
            trg_seq = torch.cat((trg_seq[:,:,:-1],self.id_enc(dec_id)),dim=-1)

        srctrg_seq = torch.cat((src_seq, trg_seq), dim=1)

        y_hat = torch.zeros((srctrg_seq.shape[0],self.params.predict_steps),device=self.device)
        Sigma = torch.zeros((srctrg_seq.shape[0],self.params.predict_steps),device=self.device)
        attn = torch.zeros((srctrg_seq.shape[0],self.params.predict_start,self.params.predict_steps),device=self.device)
        for t in range(self.params.predict_steps):
            in_conv = srctrg_seq[:,t:self.params.predict_start+t+1,:].transpose(-1, -2)
            output = self.tcn(in_conv).transpose(-1,-2)
            score = entmax15(torch.bmm(output[:,-1,:].unsqueeze_(1),output[:,:-1,:].transpose(-1,-2)))
            attn[:,t,:] = score.squeeze(1)
            c = torch.bmm(score, output[:,:-1,:]).squeeze_(1)
            concat_c = torch.cat([c, output[:,-1,:]], 1)
            y_hat[:,t] = self.linear_1(concat_c).squeeze_(-1)
            Sigma[:,t] = self.linear_2(concat_c).squeeze_(-1)
            if t < (self.params.predict_steps - 1):
                srctrg_seq[:,self.params.predict_start+t+1,0] = y_hat[:,t]

        Sigma = self.distribution_sigma(Sigma)
        return y_hat,Sigma,attn


def loss_fn(mu: Variable, sigma: Variable, labels: Variable, predict_start):
    labels = labels[:,predict_start:]
    zero_index = (labels != 0)
    mask = sigma == 0
    sigma_index = zero_index * (~mask)
    distribution = torch.distributions.normal.Normal(mu[sigma_index], sigma[sigma_index])
    likelihood = distribution.log_prob(labels[sigma_index])
    difference = torch.abs(mu - labels)
    return -0.5*torch.mean(likelihood) + torch.mean(difference)


# if relative is set to True, metrics are not normalized by the scale of labels
def accuracy_ND(mu: torch.Tensor, labels: torch.Tensor, relative = False):
    zero_index = (labels != 0)
    if relative:
        diff = torch.mean(torch.abs(mu[zero_index] - labels[zero_index])).item()
        return [diff, 1]
    else:
        diff = torch.sum(torch.abs(mu[zero_index] - labels[zero_index])).item()
        summation = torch.sum(torch.abs(labels[zero_index])).item()
        return[diff, summation]


def accuracy_RMSE(mu: torch.Tensor, labels: torch.Tensor, relative = False):
    zero_index = (labels != 0)
    diff = torch.sum(torch.mul((mu[zero_index] - labels[zero_index]), (mu[zero_index] - labels[zero_index]))).item()
    if relative:
        return [diff, torch.sum(zero_index).item(), torch.sum(zero_index).item()]
    else:
        summation = torch.sum(torch.abs(labels[zero_index])).item()
        if summation == 0:
            logger.error('summation denominator error! ')
        return [diff, summation, torch.sum(zero_index).item()]


def accuracy_ROU(mu: torch.Tensor, labels: torch.Tensor, rou: float,relative = False):
    zero_index = (labels != 0)
    if relative:
        diff = 2*torch.sum(torch.mul(rou-(labels[zero_index]<mu[zero_index]).type(mu.dtype),
                      labels[zero_index] - mu[zero_index])).item()
        return [diff, 1]
    else:
        # diff = torch.sum(torch.abs(mu[zero_index] - labels[zero_index])).item()
        diff = 2*torch.sum(torch.mul(rou-(labels[zero_index]<mu[zero_index]).type(mu.dtype),
                      labels[zero_index] - mu[zero_index])).item()
        summation = torch.sum(torch.abs(labels[zero_index])).item()
        return[diff, summation]

def accuracy_ROU(rou: float, mu: torch.Tensor, labels: torch.Tensor, relative = False):
    zero_index = (labels != 0)
    labels = labels[zero_index]
    mu = mu[zero_index]
    if relative:
        diff = 2*torch.mean(torch.mul(rou-(labels<mu).type(mu.dtype),
                      labels - mu)).item()
        return [diff, 1]
    else:
        diff = 2*torch.sum(torch.mul(rou-(labels<mu).type(mu.dtype),
                      labels - mu))
        summation = torch.sum(torch.abs(labels)).item()
        return[diff, summation]
    numerator = 0
    denominator = 0
    samples = mu
    pred_samples = mu.shape[0]
    for t in range(labels.shape[1]):
        zero_index = (labels[:, t] != 0)
        if zero_index.numel() > 0:
            rou_th = math.ceil(pred_samples * (1 - rou))
            rou_pred = torch.topk(samples[:, zero_index, t], dim=0, k=rou_th)[0][-1, :]
            abs_diff = labels[:, t][zero_index] - rou_pred
            numerator += 2 * (torch.sum(rou * abs_diff[labels[:, t][zero_index] > rou_pred]) - torch.sum(
                (1 - rou) * abs_diff[labels[:, t][zero_index] <= rou_pred])).item()
            denominator += torch.sum(labels[:, t][zero_index]).item()
    if relative:
        return [numerator, torch.sum(labels != 0).item()]
    else:
        return [numerator, denominator]

def accuracy_ROU_(rou: float, mu: torch.Tensor, labels: torch.Tensor, relative = False):
    zero_index = (labels == 0)
    mu[zero_index] = 0
    if relative:
        diff = 2*torch.mean(torch.mul(rou-(labels<mu).type(mu.dtype),
                      labels - mu), axis=1)
        return diff/1
    else:
        diff = 2*torch.sum(torch.mul(rou-(labels<mu).type(mu.dtype),
                      labels - mu), axis=1)
        summation = torch.sum(torch.abs(labels), axis=1)

        return diff/summation


def accuracy_ND_(mu: torch.Tensor, labels: torch.Tensor, relative = False):

    mu = mu.cpu().detach().numpy()
    labels = labels.cpu().detach().numpy()

    mu[labels == 0] = 0.

    diff = np.sum(np.abs(mu - labels), axis=1)
    if relative:
        summation = np.sum((labels != np.inf), axis=1)
        mask = (summation == 0)
        summation[mask] = 1
        result = diff / summation
        result[mask] = -1
        return result
    else:
        summation = np.sum(np.abs(labels), axis=1)
        mask = (summation == 0)
        summation[mask] = 1
        result = diff / summation
        result[mask] = -1
        return result


def accuracy_RMSE_(mu: torch.Tensor, labels: torch.Tensor, relative = False):
    mu = mu.cpu().detach().numpy()
    labels = labels.cpu().detach().numpy()

    mask = labels == 0
    mu[labels == 0] = 0.

    diff = np.sum((mu - labels) ** 2, axis=1)
    summation = np.sum(np.abs(labels), axis=1)
    mask2 = (summation == 0)
    if relative:
        div = np.sum(~mask, axis=1)
        div[mask2] = 1
        result = np.sqrt(diff / div)
        result[mask2] = -1
        return result
    else:
        summation[mask2] = 1
        result = (np.sqrt(diff) / summation) * np.sqrt(np.sum(~mask, axis=1))
        result[mask2] = -1
        return result
