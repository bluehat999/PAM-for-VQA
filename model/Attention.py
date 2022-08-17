import torch
import torch.nn as nn
import torch.nn.functional as F
from model.fc import FCNet
import math
from torch.nn.utils.weight_norm import weight_norm
def cosine_similarity(x1, x2, dim=2, eps=1e-8):
    #x1: (B, n, d) x2: (B, m, d)
    w12 = torch.matmul(x1, x2.transpose(-1,-2))
    #w12: (B, n, m)

    w1 = torch.norm(x1, 2, dim).unsqueeze(-1)
    w2 = torch.norm(x2, 2, dim).unsqueeze(-2)
    #w1: (B, n, 1) w2: (B, 1, m)
    w12_norm = torch.matmul(w1, w2).clamp(min=eps)
    return w12 / w12_norm
def maxAttention(att,dim=-1):
    max = torch.max(att,dim)[0].unsqueeze(dim)
    maxatt = (att == max).float()
    return maxatt
class CoAttention(nn.Module):
    def __init__(self, d_model1, d_model2, num_heads):
        # 2048,1024
        super().__init__()
        self.num_heads = num_heads
        hdim = d_model2 // num_heads
        v_projs = []
        for h in range(num_heads):
            v_proj = FCNet([d_model1, hdim],'',0.1)
            v_projs.append(v_proj)
        self.v_projs = nn.ModuleList(v_projs)
        # self.k_projs = FCNet([d_model2, num_heads * hdim],'',0.1)
        self.v_reform = FCNet([d_model2, d_model1],'',0.1)
        self.h_num = 8
        self.feat_dim = d_model2

    def forward(self, x1, x2, v_mask, q_mask): 
        B = x1.size(0)
        N1 = x1.size(1)
        N2 = x2.size(1)
        H = int(self.feat_dim/self.h_num)
        Q = None
        for h in range(self.num_heads):
            q = self.v_projs[h](x1)
            if Q is None:
                Q = q
            else:
                Q = torch.cat((Q,q),-1)
        gate1 = torch.nn.AvgPool1d(N1)(Q.transpose(-1,-2).squeeze()).view(B, 1, self.h_num, H).transpose(1,2)
        gate2 = torch.nn.AvgPool1d(N2)(x2.transpose(-1,-2).squeeze()).view(B, 1, self.h_num, H).transpose(1,2)
        Q = Q.view(B, N1, self.h_num, H).transpose(1,2) #[B, h_num, N1, H]
        D = x2.view(B, N2, self.h_num, H).transpose(1,2) #[B, h_num, N2, H] 
        V2 = x2.view(B, N2, self.h_num, H).transpose(1,2)

        D_t = torch.transpose(D, 2, 3).contiguous()  #[B, h_num, H, N2] 
        aff = torch.matmul(Q, D_t) /  math.sqrt(H) #[B, h_num, N1, N2] 
        # L = cosine_similarity(Q,D,-1)
        L = aff.masked_fill(q_mask, -1e9) if q_mask is not None else aff
        v2q = F.softmax(L, dim=3).transpose(2,3) #[B, h_num, N2, N1]
        # gate2 = torch.matmul(self.flat2(V2).transpose(-1,-2),V2)
        x2_ = torch.matmul(v2q, Q * gate2) #[B, h_num, N2, H] 
        # x2_ = torch.matmul(v2q + maxAttention(v2q), Q * gate2)
        L = aff.masked_fill(v_mask.transpose(2,3), -1e9) if v_mask is not None else aff
        q2v = F.softmax(L, dim = 2) #[B, h_num, N1, N2]
        # gate1 = torch.matmul(self.flat2(Q).transpose(-1,-2),Q)
        x1_ = torch.matmul(q2v, V2 * gate1)  #[B, h_num, N1, H]
        # x1_ = torch.matmul(q2v + maxAttention(q2v), V2 * gate1)
        x1_ = torch.transpose(x1_, 1, 2).contiguous().view(B,N1,-1)
        x2_ = torch.transpose(x2_, 1, 2).contiguous().view(B,N2,-1)

        x1_ = self.v_reform(x1_)

        # g1 = torch.sigmoid(torch.abs(x1_ - x1))
        # g2 = torch.sigmoid(torch.abs(x2_ - x2))

        emb1 = x1 * torch.sigmoid(x1_)
        emb2 = x2 * torch.sigmoid(x2_) 
    
        return emb1, emb2, v2q, q2v

class PathAttention(nn.Module):
    def __init__(self, d_model1, d_model2, num_heads):
        # 2048,1024
        super().__init__()
        self.num_heads = num_heads
        hdim = d_model2 // num_heads
        v_projs = []
        for h in range(num_heads):
            v_proj = FCNet([d_model1, hdim],'',0.1)
            v_projs.append(v_proj)
        self.v_projs = nn.ModuleList(v_projs)
        self.v_reform = FCNet([d_model2, d_model1],'',0.1)
        self.h_num = 8
        self.feat_dim = d_model2
        self.conv = weight_norm(
                            nn.Conv2d(in_channels=self.h_num,
                                      out_channels=self.h_num,
                                      kernel_size=(1, 1),
                                      groups=self.h_num), dim=None)

    def forward(self, x1, x2, a1, a2, a1_, a2_, v_mask, q_mask): 
        B = x1.size(0)
        N1 = x1.size(1)
        N2 = x2.size(1)
        H = int(self.feat_dim/self.h_num)
        Q = None
        for h in range(self.num_heads):
            q = self.v_projs[h](x1)
            if Q is None:
                Q = q
            else:
                Q = torch.cat((Q,q),-1)
        gate1 = torch.nn.AvgPool1d(N1)(Q.transpose(-1,-2).squeeze()).view(B, 1, self.h_num, H).transpose(1,2)
        gate2 = torch.nn.AvgPool1d(N2)(x2.transpose(-1,-2).squeeze()).view(B, 1, self.h_num, H).transpose(1,2)
        Q = Q.view(B, N1, self.h_num, H).transpose(1,2) #[B, h_num, N1, H]
        D = x2.view(B, N2, self.h_num, H).transpose(1,2) #[B, h_num, N2, H] 
        a1sum = a1.sum(2).view(B,self.h_num,1,-1) # node be focused/ importance
        a2sum = a2.sum(2).view(B,self.h_num,1,-1)
        a1_sum = a1_.sum(3).view(B,self.h_num,-1,1) # node be focused/ importance
        a2_sum = a2_.sum(3).view(B,self.h_num,-1,1) #[B, h_num, N1, 1]
        L1_ = torch.tanh(torch.matmul(a2_sum, a1sum))  #[B, h_num, N1, N1]
        L2_ = torch.tanh(torch.matmul(a1_sum, a2sum)) #[B, h_num, N2, N2]
        # L_ = torch.matmul(a1sum,a2sum) #[B, h_num, N1, N2]
        L_ = torch.matmul(a1_,L1_).transpose(2,3) * torch.matmul(a2_,L2_) #[B, h_num, N1, N2]
  
        aff =  self.conv(L_)  * (1.0 / math.sqrt(float(H))) 

        L = aff.masked_fill(v_mask.transpose(2,3), -1e9) if v_mask is not None else aff
        satt1 = F.softmax(L, dim=2) #[B, h_num, N1, N2]
        # gate1 = torch.matmul(self.flat2(Q).transpose(-1,-2),Q)
        m1 = torch.matmul(satt1, D * gate1) #[B, h_num, N1, H] 
        # m1 = torch.matmul(satt1 + maxAttention(satt1,2), D * gate1)
        L = aff.masked_fill(q_mask, -1e9) if q_mask is not None else aff
        satt2 = F.softmax(L, dim = 3).transpose(2,3) #[B, h_num, N2, N1]
        # gate2 = torch.matmul(self.flat2(D).transpose(-1,-2),D)
        m2 = torch.matmul(satt2, Q * gate2)  #[B, h_num, N2, H]
        # m2 = torch.matmul(satt2 + maxAttention(satt2,2), Q * gate2) 

        C_Q = torch.transpose(m1, 1, 2).contiguous().view(B,N1,-1)
        C_D = torch.transpose(m2, 1, 2).contiguous().view(B,N2,-1)

        C_Q = self.v_reform(C_Q)

        # g1 = torch.abs(C_Q - x1)
        # g2 = torch.abs(C_D - x2)


        emb1 = x1 * C_Q
        emb2 = x2 * C_D

        return emb1, emb2, satt1, satt2

class SelfAttention(nn.Module):
    def __init__(self, feat_dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        hdim = feat_dim // num_heads
        projs = []
        for h in range(num_heads):
            proj = FCNet([feat_dim, hdim],'',0.1)
            projs.append(proj)
        self.projs = nn.ModuleList(projs)
        self.h_num = 8
        self.feat_dim = feat_dim
        self.linear = FCNet([feat_dim, feat_dim], 'ReLU', 0.1)
        # self.aoa_layer =  nn.Sequential(nn.Linear((1 + 1) * feat_dim, 2 * feat_dim), nn.GLU())
    def forward(self, x1, x2, adj, bias, mask): 
        B = x1.size(0)
        num = x1.size(1)
        h_len = int(self.feat_dim/self.h_num)
        Q = None
        for h in range(self.num_heads):
            q = self.projs[h](x1)
            if Q is None:
                Q = q
            else:
                Q = torch.cat((Q,q),-1)
        Q = Q.view(B, num, self.h_num, h_len).transpose(1,2) #[B, h_num, N1, H]
        K = x1.view(B, num, self.h_num, h_len).transpose(1,2)
        V = K
        gate = torch.nn.AvgPool1d(x2.size(1))(x2.transpose(-1,-2)).squeeze().view(B, 1, self.h_num, h_len).transpose(1,2)
        K_t = torch.transpose(K * gate, 2, 3) #[batch, h_num, h_len, num]
        aff = torch.matmul(Q * gate, K_t) #[batch, h_num, num, num]
        L = (1.0 / math.sqrt(float(h_len))) * aff 
        if mask is not None:
            L = L.masked_fill(mask, -1e9)
        att = F.softmax(L, dim = 3)
        V_ = torch.matmul(att, V) # [batch, h_num, num, h_len]
        # V_ = torch.matmul(att + maxAttention(att,3), V) # [batch, h_num, num, h_len]
        V_ = torch.transpose(V_, 1, 2).contiguous().view(B,num,-1)
        # x_ = self.aoa_layer(torch.cat([V_,K.squeeze()], -1))
        xout = self.linear(V_) + x1
        
        return xout, att


# ---------------------------
# ---- Feed Forward Nets ----
# ---------------------------
from model.net_utils import MLP
class FFN(nn.Module):
    def __init__(self, in_size, mid_size, out_size, dropout):
        super(FFN, self).__init__()

        self.mlp = MLP(
            in_size=in_size,
            mid_size=mid_size,
            out_size=out_size,
            dropout_r=dropout,
            use_relu=True
        )
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNorm(in_size)

    def forward(self, x):
        return self.norm(x + self.dropout(self.mlp(x)))

class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer=None):
        "Apply residual connection to any sublayer with the same size."
        if sublayer is not None:
            return x + self.dropout(sublayer(self.norm(x)))
        return x + self.dropout(self.norm(x))

class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2