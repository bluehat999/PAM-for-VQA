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
from torch.nn.modules.linear import Linear
from model.language_model import  WordEmbedding, QuestionEmbedding
                                 
from model.fusion import SelfCrossModalFusion
from model.classifier import SimpleClassifier
from model.fc import FCNet
from model.net_utils import AttFlat

class VisualEncoder(nn.Module):
    def __init__(self,dataset,args):
        super().__init__()
        # self.p_emb = Linear(4,dataset.v_dim)
        # self.v_att = AttFlat(dataset.v_dim,dataset.v_dim,1,1,0.2)
        self.v_trans = FCNet([dataset.v_dim, args.num_hid], 'ReLU', 0.1, bias = True)
    def forward(self,v,v_b):
        v_mask = make_mask(v)
        v_emb = self.v_trans(v)
        return v_emb, v_mask
class LanguageEncoder(nn.Module):
    def __init__(self,dataset,args):
        super().__init__()
        # self.p_emb = Linear(2,args.q_dim)
        self.w_emb = WordEmbedding(dataset.dictionary.ntoken, 300, .0, args.op)
        self.q_emb = QuestionEmbedding(300 if 'c' not in args.op else 600,
                              args.q_dim, 1, False, .0)
        self.q_att = AttFlat(args.q_dim,args.q_dim,1,1,0.2)
        self.q_trans = FCNet([args.q_dim, args.num_hid], 'ReLU', 0.1, bias = True)
    def forward(self,q,q_b):
        w_emb = self.w_emb(q)
        q_words = self.q_emb.forward_all(w_emb)  # [batch, q_len, q_dim]
        q_total = self.q_att(q_words) # [batch, q_dim]
        # q_total = self.q_att(q_words,None)

        B = q.size(0)
        q_total0 = q_total.view((B,1,-1))
        qa = torch.cat((q_words,q_total0),1)
        # qa = q_words
        q_mask = make_mask(qa)
        q_emb = self.q_trans(qa)
        return q_emb, q_mask
class PAM(nn.Module):
    def __init__(self, dataset, args, graphFusion,
                  classifier, fusion, relation_type):
        super(PAM, self).__init__()
        self.name = "PAM"
        self.relation_type = relation_type
        self.fusion = fusion
        self.dataset = dataset

        self.encoderv = VisualEncoder(dataset,args)
        self.encoderq = LanguageEncoder(dataset,args)
        self.graphFusion = graphFusion
        self.classifier = classifier
    def forward(self, v, v_b, q, implicit_pos_emb, sem_adj_matrix,
                spa_adj_matrix, labels, q_b, q_objs, q_relations, q_adj):
        """Forward
        v: [batch, num_objs, obj_dim]
        b: [batch, num_objs, b_dim]
        q: [batch_size, seq_length]
        pos: [batch_size, num_objs, nongt_dim, emb_dim]
        sem_adj_matrix: [batch_size, num_objs, num_objs, num_edge_labels]
        spa_adj_matrix: [batch_size, num_objs, num_objs, num_edge_labels]
        q_ibjs: [batch_size, objs_max, obj_len]
        q_relations: [batch_size, rela_max, rela_len]

        return: logits, not probs
        """
        v_emb, v_mask = self.encoderv(v,v_b)
        q_emb, q_mask = self.encoderq(q,q_b)
        joint_emb, v2q, q2v, v2v, q2q = self.graphFusion(v_emb, q_emb, v_mask, q_mask)
        
        logits = self.classifier(joint_emb)

        return logits, v2q, q2v, v2v, q2q


def build_model(dataset, args, device):
    print("On Building!!!")
    graphFusion = SelfCrossModalFusion(
                        dataset.v_dim, args.q_dim, args.num_hid,
                        args.dir_num, args.sem_label_num,args.spa_label_num,
                        num_heads=args.num_heads, device=device)
    classifier = SimpleClassifier(args.num_hid, args.num_hid * 2,
                                dataset.num_ans_candidates, 0.5)
    return PAM(dataset, args, graphFusion,
                classifier, args.fusion, args.relation_type)
def make_mask(feature):
        return (torch.sum(
            torch.abs(feature),
            dim=-1
        ) == 0).unsqueeze(1).unsqueeze(2)
