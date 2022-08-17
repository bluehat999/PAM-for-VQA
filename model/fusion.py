"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

Relation-aware Graph Attention Network for Visual Question Answering
Linjie Li, Zhe Gan, Yu Cheng, Jingjing Liu
https://arxiv.org/abs/1903.12314

This code is written by Linjie Li.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.Attention import  PathAttention, CoAttention, SelfAttention, SublayerConnection
from model.net_utils import AttFlat,MLP,LayerNorm
from model.fc import FCN
from torch.nn.utils.weight_norm import weight_norm


def q_expand_v_cat(q, v, mask=True):
    """
        Args:
            v: [batch_size, num_rois, out_dim]
            q: [batch_size, q_dim]
        Returns:
            output: [batch_size, num_rois, out_dim+q_dim]
    """
    q = q.view(q.size(0), 1, q.size(1))
    repeat_vals = (-1, v.shape[1], -1)
    q_expand = q.expand(*repeat_vals)
    if mask:
        v_sum = v.sum(-1)
        mask_index = torch.nonzero(v_sum == 0)
        if mask_index.dim() > 1:
            q_expand[mask_index[:, 0].clone(), mask_index[:, 1].clone()] = 0
    v_cat_q = torch.cat((v, q_expand), dim=-1)
    return v_cat_q

def l2norm(X, dim, eps=1e-8):
    """L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
    X = torch.div(X, norm)
    return X

class SelfCrossGraphLayer(nn.Module):
    def __init__(self, v_dim, q_dim, num_heads=16,dropout=0.2):
        super(SelfCrossGraphLayer, self).__init__()
        self.v_dim = v_dim
        self.q_dim = q_dim
        self.h_num = num_heads
        self.intra_v = SelfAttention(v_dim, num_heads)
        self.intra_q = SelfAttention(q_dim, num_heads) 
        # self.intra_v1 = SelfAttention(v_dim, num_heads)
        # self.intra_q1 = SelfAttention(q_dim, num_heads) 
        self.atten = CoAttention(v_dim, q_dim, num_heads)
        self.align = PathAttention(v_dim, q_dim, num_heads)  
        self.norm_v = SublayerConnection(v_dim, 0.1)
        self.norm_q = SublayerConnection(q_dim, 0.1)
        self.alpha = 0.95
        self.beta = 0.95
        # self.norm = LayerNorm(q_dim)
    def forward(self, v_emb, q_emb, v_mask, q_mask,scale=1,v_imp = None, q_imp = None):
        """
        Args:
            v: [batch_size, num_rois, v_dim]
            q_objs: [batch_size, objs_num, q_dim]
            v_adj: [batch_size, num_rois, num_rois, num_labels]
            q_adj: [batch_size, objs_num, objs_num, 1 ]
            q_relations: [batch_size, relas_num, q_dim]

        Returns:
            output: [batch_size, num_rois, out_dim]
        """
        B = v_emb.size(0)
        N1 = v_emb.size(1)
        N2 = q_emb.size(1)

        v_emb1c, q_emb1c , v2q , q2v = self.atten(v_emb, q_emb, v_mask, q_mask)
        v_imp = torch.sigmoid((1-self.alpha) * v_imp + self.alpha * q2v.sum(-1).sum(1))
        q_imp = torch.sigmoid((1-self.alpha) * q_imp + self.alpha * v2q.sum(-1).sum(1))
        v_emb1c = v_emb1c * v_imp.unsqueeze(-1)
        q_emb1c = q_emb1c * q_imp.unsqueeze(-1)

        v_emb1s, v2v = self.intra_v(v_emb1c, q_emb1c, None, None, v_mask)
        q_emb1s, q2q = self.intra_q(q_emb1c, v_emb1c, None, None, q_mask)
        v_imp = torch.sigmoid((1-self.beta) * v_imp + self.beta * vv.sum(-2).sum(1))
        q_imp = torch.sigmoid((1-self.beta) * q_imp + self.beta * qq.sum(-2).sum(1))
        v_emb1s = v_emb1s * v_imp.unsqueeze(-1)
        q_emb1s = q_emb1s * q_imp.unsqueeze(-1)

        # v_emb1s, vv = self.intra_v1(v_emb1s, q_emb1s, None, None, v_mask)
        # q_emb1s, qq = self.intra_q1(q_emb1s, v_emb1s, None, None, q_mask)
        # v_imp = torch.sigmoid((1-self.beta) * v_imp + self.beta * vv.sum(-2).sum(1))
        # q_imp = torch.sigmoid((1-self.beta) * q_imp + self.beta * qq.sum(-2).sum(1))
        # v_emb1s = v_emb1s * v_imp.unsqueeze(-1)
        # q_emb1s = q_emb1s * q_imp.unsqueeze(-1)

        # v2v = F.softmax(vv*v2v)
        # q2q = F.softmax(qq*q2q) 

        v_emb1a, q_emb1a, l1, l2= self.align(v_emb1s, q_emb1s, v2v, q2q, v2q, q2v, v_mask, q_mask)
        vo = self.norm_v(v_emb1a)
        qo = self.norm_q(q_emb1a)

        return vo, qo, v2q, q2v, v2v, q2q, v_imp, q_imp
class SelfCrossModalFusion(nn.Module):
    def __init__(self, v_dim, q_dim, out_dim, dir_num, label_num0,label_num1, num_heads=16,  dropout=0.2, device=None):
        super(SelfCrossModalFusion, self).__init__()
        self.dir_num = 1
        self.out_dim = out_dim
        v_dim = out_dim
        q_dim = out_dim
        self.q_dim = out_dim
        self.v_dim = out_dim
        self.device = device
        self.l = 1
        layers = []
        for i in range(self.l):
            g_att_layer = SelfCrossGraphLayer(v_dim,q_dim,num_heads=num_heads)
            layers.append(g_att_layer)
        self.layers = nn.ModuleList(layers)
        self.flat_v = AttFlat(v_dim,v_dim,out_dim)
        self.flat_q = AttFlat(q_dim,q_dim,out_dim)
        self.linear_out = FCNet([v_dim+q_dim, out_dim], 'ReLU', 0.1, True)
        # self.out_norm = LayerNorm(out_dim)
    def forward(self, v_emb, q_emb, v_mask, q_mask):
        """
        Returns:
            output: [batch_size, out_dim]
        """
        B = v_emb.size(0)
        v_mask = None
        q_mask = None

        qs = []
        vs = []
        v2q, q2v, v2v, q2q = 0, 0, 0, 0
        # madgap = 0
        N1 = v_emb.size(1)
        N2 = q_emb.size(1)
        v_imp = torch.sigmoid(torch.ones((B,N1)).cuda())
        q_imp = torch.sigmoid(torch.ones((B,N2)).cuda())
        for i in range(self.l):
            v_out, q_out, v2q, q2v, v2v, q2q, v_imp, q_imp = self.layers[i](v_emb,q_emb,v_mask,q_mask,v_imp=v_imp,q_imp=q_imp)
            vs += [v_out.view(B,1,-1,self.v_dim)]
            qs += [q_out.view(B,1,-1,self.q_dim)]
            ve = self.flat_v(v_out,v_mask)
            qe = self.flat_q(q_out,q_mask) 
            f = torch.cat([ve,qe],1)
            if i == 0:
                out = self.linear_out(f) + ve + qe
            else: 
                out = out +  (self.linear_out(f) + ve + qe)
        return out, v2q, q2v, v2v, q2q

def mad_gap_regularizer(intensor,neb_mask,rmt_mask,target_idx=None):
    B,N,H = intensor.size()
    simi_tensor = cosine_similarity(intensor,intensor,-1).view(B,N,N)
    dist_tensor = 1 - simi_tensor

    neb_dist = torch.mul(dist_tensor,neb_mask)
    rmt_dist = torch.mul(dist_tensor,rmt_mask)
    
    divide_neb = (neb_dist!=0).sum(2).type(torch.FloatTensor).cuda() + 1e-8
    divide_rmt = (rmt_dist!=0).sum(2).type(torch.FloatTensor).cuda() + 1e-8

    neb_mean_list = neb_dist.sum(2) / divide_neb
    rmt_mean_list = rmt_dist.sum(2) / divide_rmt

    # neb_mad = torch.mean(neb_mean_list[:,target_idx])
    # rmt_mad = torch.mean(rmt_mean_list[:,target_idx])
    neb_mad = torch.mean(neb_mean_list)
    rmt_mad = torch.mean(rmt_mean_list)

    mad_gap = rmt_mad - neb_mad
    return mad_gap
def cosine_similarity(x1, x2, dim=2, eps=1e-8):
    #x1: (B, n, d) x2: (B, m, d)
    w12 = torch.matmul(x1, x2.transpose(-1,-2))
    #w12: (B, n, m)
    w1 = torch.norm(x1, 2, dim).unsqueeze(-1)
    w2 = torch.norm(x2, 2, dim).unsqueeze(-2)
    #w1: (B, n, 1) w2: (B, 1, m)
    w12_norm = torch.matmul(w1, w2).clamp(min=eps)
    return w12 / w12_norm

def mad(intensor):
    (B,N,H) = intensor.size()
    simi_tensor = cosine_similarity(intensor,intensor,-1).view(B,N,N)
    dist_tensor = 1 - simi_tensor
    neb_dist = dist_tensor
    divide_dist = (neb_dist!=0).sum(2).type(torch.FloatTensor).cuda() + 1e-8
    neb_mean_list = neb_dist.sum(2) / divide_dist
    neb_mad = torch.mean(neb_mean_list)
    return neb_mad
