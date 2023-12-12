"""
Adapted from https://github.com/Linear95/CLUB/blob/master/mi_estimators.py
"""

import numpy as np
import math

import torch 
import torch.nn as nn

from src.utils.ddp_utils import SyncFunction


class CLUBVec2Seq(nn.Module):
    """ The CLUB estimator for vector-to-sequence pairs.
    """
    def __init__(
        self,
        seq_dim: int,
        vec_dim: int,
        hidden_size: int,
        is_sampled_version: bool = False,
    ):
        super().__init__()
        self.is_sampled_version = is_sampled_version

        self.seq_prenet = nn.Sequential(
            nn.Conv1d(seq_dim, hidden_size, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(hidden_size, hidden_size, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(hidden_size, hidden_size, kernel_size=5, padding=2),
        )
        # mu net
        self.p_mu = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, vec_dim)
        )
        # variance net
        self.p_logvar = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, vec_dim),
            nn.Tanh()
        )

    def temporal_avg_pool(self, x, mask=None):
        """
        Args:
            x (tensor): shape [B, T, D]
            mask (bool tensor): padding parts with ones
        """
        if mask is None:
            out = torch.mean(x, dim=1)
        else:
            len_ = (~mask).sum(dim=1).unsqueeze(1)
            x = x.masked_fill(mask.unsqueeze(-1), 0)
            x = x.sum(dim=1)
            out = torch.div(x, len_)
        return out

    def get_mu_logvar(self, seq, mask):
        # [B, T, D]
        h = self.seq_prenet(seq.transpose(1, 2)).transpose(1, 2)
        # [B, D]
        h = self.temporal_avg_pool(h, mask)
        mu = self.p_mu(h)
        logvar = self.p_logvar(h)
        return mu, logvar
        
    def loglikeli(self, seq, vec, mask=None):
        """ Compute un-normalized log-likelihood
        Args:
            seq (tensor): sequence feature, shape [B, T, D].
            vec (tensor): vector feature, shape [B, D].
            mask (tensor): padding parts with ones, [B, T].
        """
        # mu/logvar: (bs, vec_dim)
        mu, logvar = self.get_mu_logvar(seq, mask)
        return (-(mu - vec)**2 /logvar.exp() - logvar).sum(dim=1).mean(dim=0) 

    def forward(self, seq, vec, mask=None):
        """ Estimate mutual information CLUB upper bound.
        Args:
            seq (tensor): sequence feature, shape [B, T, D].
            vec (tensor): vector feature, shape [B, D].
            mask (tensor): padding parts with ones, [B, T].
        """

        mu, logvar = self.get_mu_logvar(seq, mask)

        if self.is_sampled_version:
            sample_size = seq.shape[0]
            #random_index = torch.randint(sample_size, (sample_size,)).long()
            random_index = torch.randperm(sample_size).long()

            positive = - (mu - vec)**2 / logvar.exp()
            negative = - (mu - vec[random_index])**2 / logvar.exp()
            upper_bound = (positive.sum(dim = -1) - negative.sum(dim = -1)).mean()
            
            mi_upper = upper_bound / 2.
        else:
            # log of conditional probability of positive sample pairs, [B, D]
            positive = - (mu - vec)**2 /2./logvar.exp()
            # [B, 1, D]
            prediction_1 = mu.unsqueeze(1)
            ## gather representations in case of distributed training
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                # [B * world_size, D]
                y_samples_1_dist = SyncFunction.apply(vec)
                # [1, B * world_size, D]
                y_samples_1 = y_samples_1_dist.unsqueeze(0)
            else:
                # [1, B, D]
                y_samples_1 = vec.unsqueeze(0)

            # log of conditional probability of negative sample pairs, [B, D]
            negative = - ((y_samples_1 - prediction_1)**2).mean(dim=1)/2./logvar.exp() 

            mi_upper = (positive.sum(dim = -1) - negative.sum(dim = -1)).mean()
        # print(mi_upper)
        
        return mi_upper

    def learning_loss(self, seq, vec, mask=None):
        return - self.loglikeli(seq, vec, mask)


class CLUBForCategorical(nn.Module):
    """
    This class provide a CLUB estimator to calculate MI upper bound between 
    vector-like embeddings and categorical labels.
    
    Estimate I(X,Y), where X is continuous vector and Y is discrete label.

    """
    def __init__(self, input_dim, label_num, hidden_size=None):
        '''
        input_dim : the dimension of input embeddings
        label_num : the number of categorical labels 
        '''
        super().__init__()
        
        if hidden_size is None:
            self.variational_net = nn.Linear(input_dim, label_num)
        else:
            self.variational_net = nn.Sequential(
                nn.Linear(input_dim, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, label_num)
            )
            
    def forward(self, inputs, labels):
        """
        Args:
            inputs : shape [batch_size, input_dim], a batch of embeddings
            labels : shape [batch_size], a batch of label index
        """
        logits = self.variational_net(inputs)  #[sample_size, label_num]
        
        # log of conditional probability of positive sample pairs
        #positive = - nn.functional.cross_entropy(logits, labels, reduction='none')    
        sample_size, label_num = logits.shape
        
        # shape [B, B, label_num]
        logits_extend = logits.unsqueeze(1).repeat(1, sample_size, 1)
        # shape [B, B]
        labels_extend = labels.unsqueeze(0).repeat(sample_size, 1)
        # log of conditional probability of negative sample pairs
        log_mat = - nn.functional.cross_entropy(
            logits_extend.reshape(-1, label_num),
            labels_extend.reshape(-1, ),
            reduction='none'
        )
        log_mat = log_mat.reshape(sample_size, sample_size)
        positive = torch.diag(log_mat).mean()
        ## gather representations in case of distributed training
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            # [B * world_size, label_num]
            logits_dist = SyncFunction.apply(logits)
            world_size = torch.distributed.get_world_size()
            # [B * world_size, B, label_num]
            logits_dist_exp = logits_dist.unsqueeze(1).repeat(1, sample_size, 1)
            # [B * world_size]
            labels_dist = labels.unsqueeze(0).repeat(world_size, 1).reshape(-1)
            # [B, B * world_size]
            labels_dist_exp = labels_dist.unsqueeze(0).repeat(sample_size, 1)
            log_mat_dist = - nn.functional.cross_entropy(
                logits_dist_exp.reshape(-1, label_num),
                labels_dist_exp.reshape(-1, ),
                reduction='none'
            )
            # [B, B * world_size]
            log_mat_dist = log_mat_dist.reshape(sample_size, -1)
            negative = log_mat_dist.mean()
        else:
            negative = log_mat.mean()
        return positive - negative

    def loglikeli(self, inputs, labels):
        logits = self.variational_net(inputs)
        return - nn.functional.cross_entropy(logits, labels)
    
    def learning_loss(self, inputs, labels):
        return - self.loglikeli(inputs, labels)
    

class CLUB(nn.Module):  # CLUB: Mutual Information Contrastive Learning Upper Bound
    '''
        This class provides the CLUB estimation to I(X,Y)
        Method:
            forward() :      provides the estimation with input samples  
            loglikeli() :   provides the log-likelihood of the approximation q(Y|X) with input samples
        Arguments:
            x_dim, y_dim :         the dimensions of samples from X, Y respectively
            hidden_size :          the dimension of the hidden layer of the approximation network q(Y|X)
            x_samples, y_samples : samples from X and Y, having shape [sample_size, x_dim/y_dim] 
    '''
    def __init__(self, x_dim, y_dim, hidden_size, is_sampled_version=False):
        super(CLUB, self).__init__()
        self.is_sampled_version = is_sampled_version
        # p_mu outputs mean of q(Y|X)
        #print("create CLUB with dim {}, {}, hiddensize {}".format(x_dim, y_dim, hidden_size))
        self.p_mu = nn.Sequential(nn.Linear(x_dim, hidden_size//2),
                                       nn.ReLU(),
                                       nn.Linear(hidden_size//2, y_dim))
        # p_logvar outputs log of variance of q(Y|X)
        self.p_logvar = nn.Sequential(nn.Linear(x_dim, hidden_size//2),
                                       nn.ReLU(),
                                       nn.Linear(hidden_size//2, y_dim),
                                       nn.Tanh())

    def get_mu_logvar(self, x_samples):
        mu = self.p_mu(x_samples)
        logvar = self.p_logvar(x_samples)
        return mu, logvar
    
    def forward(self, x_samples, y_samples): 
        mu, logvar = self.get_mu_logvar(x_samples)

        if self.is_sampled_version:
            sample_size = x_samples.shape[0]
            #random_index = torch.randint(sample_size, (sample_size,)).long()
            random_index = torch.randperm(sample_size).long()
            
            positive = - (mu - y_samples)**2 / logvar.exp()
            negative = - (mu - y_samples[random_index])**2 / logvar.exp()
            upper_bound = (positive.sum(dim = -1) - negative.sum(dim = -1)).mean()

            mi_upper = upper_bound / 2.
        else:
            
            # log of conditional probability of positive sample pairs
            positive = - (mu - y_samples)**2 /2./logvar.exp()  
            
            prediction_1 = mu.unsqueeze(1)          # shape [nsample,1,dim]
            # y_samples_1 = y_samples.unsqueeze(0)    # shape [1,nsample,dim]
            ## gather representations in case of distributed training
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                # [B * world_size, D]
                y_samples_1_dist = SyncFunction.apply(y_samples)
                # [1, B * world_size, D]
                y_samples_1 = y_samples_1_dist.unsqueeze(0)
            else:
                # [1, B, D]
                y_samples_1 = y_samples.unsqueeze(0)

            # log of conditional probability of negative sample pairs
            negative = - ((y_samples_1 - prediction_1)**2).mean(dim=1)/2./logvar.exp() 

            mi_upper = (positive.sum(dim = -1) - negative.sum(dim = -1)).mean()
        return mi_upper   

    def loglikeli(self, x_samples, y_samples): # unnormalized loglikelihood 
        mu, logvar = self.get_mu_logvar(x_samples)
        return (-(mu - y_samples)**2 /logvar.exp()-logvar).sum(dim=1).mean(dim=0)
    
    def learning_loss(self, x_samples, y_samples):
        return - self.loglikeli(x_samples, y_samples)
