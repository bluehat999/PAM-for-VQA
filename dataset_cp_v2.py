"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

This code is modified by Linjie Li from Jin-Hwa Kim's repository.
https://github.com/jnhwkim/ban-vqa
MIT License
"""
from __future__ import print_function
import os
import json
import pickle
import numpy as np
import utils
import h5py
import torch
from torch.utils.data import Dataset
import tools.compute_softscore
from dataset import is_howmany

COUNTING_ONLY = False


def _create_entry(img, question, answer, en):
    if answer is not None:
        answer.pop('image_id')
        answer.pop('question_id')
    entry = {
        'question_id': question['question_id'],
        'image_id': question['image_id'],
        'image': img,
        'coco_split': question["coco_split"],
        'question': question['question'],
        'answer': answer,
        'q_bounds': en['bounds'],
        'r_index': en['r_index'],
        'r': en['r']}
    return entry


def _load_dataset(dataroot, name, coco_train_img_id2val, coco_val_img_id2val,
                  label2ans):
    """Load entries

    coco_train_img_id2val/coco_val_img_id2val:
        dict {img_id -> val} val can be used to retrieve image or features
    dataroot: root path of dataset
    name: 'train', 'val'
    """
    question_path = os.path.join(
        dataroot, 'cp_v2_questions/vqacp_v2_%s_questions.json' % name)
    questions = sorted(json.load(open(question_path)),
                       key=lambda x: x['question_id'])
    answer_path = os.path.join(dataroot, 'cache', 'cp_v2_%s_target.pkl' % name)
    answers = pickle.load(open(answer_path, 'rb'))
    answers = sorted(answers, key=lambda x: x['question_id'])
    entity_path = os.path.join(dataroot, 'cache', 'cp_v2_%s_entity_bounds1.pkl' % name)
    entity = pickle.load(open(entity_path, 'rb'))
    entity = sorted(entity, key=lambda x: x['question_id'])
    utils.assert_eq(len(questions), len(answers))
    entries = []
    for question, answer, en in zip(questions, answers, entity):
        utils.assert_eq(question['question_id'], answer['question_id'])
        utils.assert_eq(question['question_id'], en['question_id'])
        utils.assert_eq(question['image_id'], answer['image_id'])
        img_id = question['image_id']
        coco_split = question["coco_split"]
        index = coco_train_img_id2val[img_id]\
            if coco_split == "train2014" else coco_val_img_id2val[img_id]
        if not COUNTING_ONLY \
           or is_howmany(question['question'], answer, label2ans):
            entries.append(_create_entry(index, question, answer, en))
    return entries


class Image_Feature_Loader():
    def __init__(self, coco_split, relation_type, dataroot='data',
                 adaptive=True):
        super(Image_Feature_Loader, self).__init__()
        assert coco_split in ['train', 'val']
        self.adaptive = adaptive
        self.relation_type = relation_type if type(relation_type)!=type("") else [relation_type]
        prefix = '36'

        self.img_id2idx = pickle.load(
            open(os.path.join(dataroot, 'imgids/%s%s_imgid2idx.pkl' %
                              (coco_split, '' if self.adaptive else prefix)),
                 'rb'))
        h5_dataroot = dataroot+"/Bottom-up-features-adaptive" \
            if self.adaptive else dataroot+"/Bottom-up-features-fixed"
        h5_path = os.path.join(h5_dataroot,
                               '%s%s.hdf5' % (coco_split,
                                              '' if self.adaptive else prefix))

        print('loading features from h5 file %s' % h5_path)
        with h5py.File(h5_path, 'r') as hf:
            self.features = np.array(hf.get('image_features'))
            self.spatials = np.array(hf.get('spatial_features'))
            self.bb = np.array(hf.get('image_bb'))
            if "semantic_adj_matrix" in hf.keys() \
               and "semantic" in self.relation_type:
                self.semantic_adj_matrix = np.array(
                                            hf.get('semantic_adj_matrix'))
                print("Loaded semantic adj matrix from file...",
                      self.semantic_adj_matrix.shape)
            else:
                self.semantic_adj_matrix = None
                print("Setting semantic adj matrix to None...")
            if "image_adj_matrix" in hf.keys() \
               and "spatial" in self.relation_type:
                self.spatial_adj_matrix = np.array(hf.get('image_adj_matrix'))
                print("Loaded spatial adj matrix from file...",
                      self.spatial_adj_matrix.shape)
            else:
                self.spatial_adj_matrix = None
                print("Setting spatial adj matrix to None...")
            self.pos_boxes = None
            if self.adaptive:
                self.pos_boxes = np.array(hf.get('pos_boxes'))
        self.tensorize()

    def tensorize(self):
        self.features = torch.from_numpy(self.features)
        self.spatials = torch.from_numpy(self.spatials)
        self.bb = torch.from_numpy(self.bb)
        if self.semantic_adj_matrix is not None:
            self.semantic_adj_matrix = torch.from_numpy(
                                        self.semantic_adj_matrix).double()
        if self.spatial_adj_matrix is not None:
            self.spatial_adj_matrix = torch.from_numpy(
                                        self.spatial_adj_matrix).double()
        if self.pos_boxes is not None:
            self.pos_boxes = torch.from_numpy(self.pos_boxes)


class VQA_cp_Dataset(Dataset):
    def __init__(self, name, dictionary, coco_train_features,
                 coco_val_features, dataroot='data', adaptive=False,
                 pos_emb_dim=64):
        super(VQA_cp_Dataset, self).__init__()
        assert name in ['train', 'test']

        ans2label_path = os.path.join(dataroot, 'cache',
                                      'trainval_ans2label.pkl')
        label2ans_path = os.path.join(dataroot, 'cache',
                                      'trainval_label2ans.pkl')
        self.ans2label = pickle.load(open(ans2label_path, 'rb'))
        self.label2ans = pickle.load(open(label2ans_path, 'rb'))
        self.num_ans_candidates = len(self.ans2label)
        self.dictionary = dictionary
        self.adaptive = adaptive
        self.relation_type = coco_train_features.relation_type
        self.coco_train_features = coco_train_features
        self.coco_val_features = coco_val_features
        self.entries = _load_dataset(dataroot, name,
                                     self.coco_train_features.img_id2idx,
                                     self.coco_val_features.img_id2idx,
                                     self.label2ans)
        self.tokenize()
        self.tensorize()
        self.emb_dim = pos_emb_dim
        self.v_dim = self.coco_train_features.features.size(1 if self.adaptive
                                                            else 2)
        self.s_dim = self.coco_train_features.spatials.size(1 if self.adaptive
                                                            else 2)

    def tokenize(self, max_length=14,obj_max=8, rela_max=8, obj_len=6, rela_len=6):
        """Tokenizes the questions.

        This will add q_token in each entry of the dataset.
        -1 represent nil, and should be treated as padding_idx in embedding
        """
        for entry in self.entries:
            tokens = self.dictionary.tokenize(entry['question'], False)
            tokens = tokens[:max_length]
            if len(tokens) < max_length:
                # Note here we pad to the back of the sentence
                padding = [self.dictionary.padding_idx] * \
                          (max_length - len(tokens))
                tokens = tokens + padding
            utils.assert_eq(len(tokens), max_length)
            entry['q_token'] = tokens
            # r_tokens = entry['r']
            # r_num = len(r_tokens)
            # for i in range(r_num):
            #     if len(r_tokens[i]) < rela_len:
            #         padding = [self.dictionary.padding_idx] * \
            #               (rela_len - len(r_tokens[i]))
            #         r_tokens[i] = r_tokens[i] + padding
            #     utils.assert_eq(len(r_tokens[i]), rela_len)
            # r_tokens = np.array(r_tokens).reshape((-1,rela_len))
            # r_tokens = np.pad(r_tokens,((0,rela_max-r_num),(0,0)),'constant',constant_values=(self.dictionary.padding_idx,))
            # entry['r'] = torch.from_numpy(r_tokens).long()

            # bounds = np.array(entry['q_bounds'], dtype=np.int64)
            # bounds = bounds.reshape((-1,2))
            # num = len(bounds)
            # bounds = np.pad(bounds,((0,obj_max-num),(0,0)),'constant',constant_values=(-1,))
            # objs = []
            # for i in range(num):
            #     if bounds[i][0] != -1:
            #         objs.append(tokens[bounds[i][0]:bounds[i][1]])
            #     else:
            #         objs.append([])
            #     objs[i] = objs[i][:obj_len]
            #     if len(objs[i]) < obj_len:
            #         padding = [self.dictionary.padding_idx] * \
            #             (obj_len - len(objs[i]))
            #         objs[i] = objs[i] + padding
            #     utils.assert_eq(len(objs[i]), obj_len)
            # objs = np.array(objs).reshape((-1,obj_len))
            # objs = np.pad(objs,((0,obj_max-num),(0,0)),'constant',constant_values=(self.dictionary.padding_idx,))
            # entry['q_objs'] = torch.from_numpy(np.array(objs)).long()
            # entry['q_bounds'] = torch.from_numpy(bounds).long()

    def tensorize(self, obj_max=8):
        for entry in self.entries:
            question = torch.from_numpy(np.array(entry['q_token']))
            entry['q_token'] = question

            # r_index = entry['r_index']
            # r_num = len(r_index)
            # index = [[],[]] if r_num==0 else list(zip(*r_index))
            # entry['q_adj'] = torch.sparse_coo_tensor(index, np.ones((r_num)), (obj_max, obj_max)).to_dense().long()

            answer = entry['answer']
            if answer is not None:
                labels = np.array(answer['labels'])
                scores = np.array(answer['scores'], dtype=np.float32)
                if len(labels):
                    labels = torch.from_numpy(labels)
                    scores = torch.from_numpy(scores)
                    entry['answer']['labels'] = labels
                    entry['answer']['scores'] = scores
                else:
                    entry['answer']['labels'] = None
                    entry['answer']['scores'] = None

    def __getitem__(self, index):
        entry = self.entries[index]
        raw_question = entry["question"]
        image_id = entry["image_id"]
        coco_split = entry["coco_split"]

        question = entry['q_token']
        question_id = entry['question_id']

        q_bounds = 0
        q_objs = 0
        q_relations = 0
        q_adj = 0

        if "train" in coco_split:
            coco_features = self.coco_train_features
        elif "val" in coco_split:
            coco_features = self.coco_val_features
        else:
            print("Unknown coco split: %s" % coco_split)

        if coco_features.spatial_adj_matrix is not None:
            spatial_adj_matrix = coco_features.spatial_adj_matrix[
                                    entry["image"]]
        else:
            spatial_adj_matrix = torch.zeros(1).double()
        if coco_features.semantic_adj_matrix is not None:
            semantic_adj_matrix = coco_features.semantic_adj_matrix[
                                    entry["image"]]
        else:
            semantic_adj_matrix = torch.zeros(1).double()

        if not self.adaptive:
            # fixed number of bounding boxes
            features = coco_features.features[entry['image']]
            spatials = coco_features.spatials[entry['image']]
            bb = coco_features.bb[entry["image"]]
        else:
            features = coco_features.features[
                coco_features.pos_boxes[
                    entry['image']][0]:coco_features.pos_boxes[
                                                    entry['image']][1], :]
            spatials = coco_features.spatials[
                coco_features.pos_boxes[
                    entry['image']][0]:coco_features.pos_boxes[
                                                    entry['image']][1], :]
            bb = coco_features.bb[
                coco_features.pos_boxes[
                    entry['image']][0]:coco_features.pos_boxes[
                                                    entry['image']][1], :]

        answer = entry['answer']
        if answer is not None:
            labels = answer['labels']
            scores = answer['scores']
            target = torch.zeros(self.num_ans_candidates)
            if labels is not None:
                target.scatter_(0, labels, scores)
            return features, spatials, question, target, question_id,\
                image_id, bb, spatial_adj_matrix, semantic_adj_matrix,\
                    raw_question, q_bounds, q_objs, q_relations, q_adj

        else:
            return features, spatials, question, question_id, question_id,\
                image_id, bb, spatial_adj_matrix, semantic_adj_matrix,\
                 raw_question,q_bounds, q_objs, q_relations, q_adj

    def __len__(self):
        return len(self.entries)
