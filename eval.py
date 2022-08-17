"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

This code is modified by Linjie Li from Jin-Hwa Kim's repository.
https://github.com/jnhwkim/ban-vqa
MIT License
"""
import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable
from tqdm import tqdm
import json

from dataset import Dictionary, VQAFeatureDataset
from dataset_cp_v2 import VQA_cp_Dataset, Image_Feature_Loader
from model.model import build_model
from train import compute_score_with_logits
from model.position_emb import prepare_graph_variables
from config.parser import Struct
import utils


@torch.no_grad()
def evaluate(models, dataloader, model_hpss, args, device, modelWeight):
    for i in range(len(models)):
        models[i].eval()
    label2ans = dataloader.dataset.label2ans
    num_answers = len(label2ans)
    relation_type = dataloader.dataset.relation_type
    N = len(dataloader.dataset)
    model_hps = model_hpss[0]
    results = []
    score = 0
    pbar = tqdm(total=len(dataloader))

    if args.save_logits:
        idx = 0
        pred_logits = np.zeros((N, num_answers))
        gt_logits = np.zeros((N, num_answers))

    for i, (v, norm_bb, q, target, qid, imgid, bb,
            spa_adj_matrix, sem_adj_matrix, raw_question, q_bounds, q_objs, q_relations, q_adj) in enumerate(dataloader):
        if i>0:
            break
        batch_size = v.size(0)
        num_objects = v.size(1)
        v = Variable(v).to(device)
        bb = Variable(bb).to(device)
        q = Variable(q).to(device)
        q_bounds = Variable(q_bounds).to(device).float()
        pos_emb, sem_adj_matrix, spa_adj_matrix = prepare_graph_variables(
            relation_type, bb, sem_adj_matrix, spa_adj_matrix, num_objects,
            model_hps.nongt_dim, model_hps.imp_pos_emb_dim,
            model_hps.spa_label_num, model_hps.sem_label_num, device)
        pred = None
        for j in range(len(models)):
            model = models[j]
            pred_one, v2q, q2v, v2v, q2q = model(v, bb, q, pos_emb, sem_adj_matrix,
                          spa_adj_matrix, None, q_bounds, q_objs, q_relations, q_adj)
            if type(pred) == type(None):
                pred = modelWeight[j] * pred_one
            else:
                pred += modelWeight[j] * pred_one
        savaAttention(raw_question,qid,None,v2q,q2v,v2v,q2q)
        # break
        # Check if target is a placeholder or actual targets
        if target.size(-1) == num_answers:
            target = Variable(target).to(device)
            batch_score = compute_score_with_logits(
                pred, target, device).sum()
            score += batch_score
            if args.save_logits:
                gt_logits[idx:batch_size+idx, :] = target.cpu().numpy()

        if args.save_logits:
            pred_logits[idx:batch_size+idx, :] = pred.cpu().numpy()
            idx += batch_size
        if args.save_answers:
            qid = qid.cpu()
            pred = pred.cpu()
            current_results = make_json(pred, qid, dataloader)
            results.extend(current_results)

        results_folder = f"{args.output_folder}/partResults"
        if args.save_answers:
            utils.create_dir(results_folder)
            save_to = f"{results_folder}/{args.dataset}_" +\
                f"{args.split}.json"
            json.dump(make_topk_json(raw_question, pred, qid, imgid, dataloader), open(save_to, "w"))

        pbar.update(1)

    score = score / N
    results_folder = f"{args.output_folder}/results"
    if args.save_logits:
        utils.create_dir(results_folder)
        save_to = f"{results_folder}/logits_{args.dataset}" +\
            f"_{args.split}.npy"
        np.save(save_to, pred_logits)

        utils.create_dir("./gt_logits")
        save_to = f"./gt_logits/{args.dataset}_{args.split}_gt.npy"
        if not os.path.exists(save_to):
            np.save(save_to, gt_logits)
    if args.save_answers:
        utils.create_dir(results_folder)
        save_to = f"{results_folder}/{args.dataset}_" +\
            f"{args.split}.json"
        json.dump(results, open(save_to, "w"))
    return score

def get_topk_answer(p,dataloader,k):
    scores, idxs = p.topk(k=k,largest=True,sorted=True)
    answers = {}

    for i in range(len(idxs)):
        ans = dataloader.dataset.label2ans[idxs[i].item()]
        answers[ans] = scores[i].item()
    return answers

def make_topk_json(questions, logits, qIds, imgids, dataloader):
    utils.assert_eq(logits.size(0), len(qIds))
    results = []
    for i in range(logits.size(0)):
        result = {}
        result['question_id'] = qIds[i].item()
        result['image_id'] = imgids[i].item()
        result['question'] = questions[i]
        result['answer'] = get_answer(logits[i], dataloader)
        result['topk'] = get_topk_answer(logits[i], dataloader,10)
        results.append(result)
    return results

def get_answer(p, dataloader):
    _m, idx = p.max(0)
    return dataloader.dataset.label2ans[idx.item()]


def make_json(logits, qIds, dataloader):
    utils.assert_eq(logits.size(0), len(qIds))
    results = []
    for i in range(logits.size(0)):
        result = {}
        result['question_id'] = qIds[i].item()
        result['answer'] = get_answer(logits[i], dataloader)
        result['topk'] = get_topk_answer(logits[i], dataloader,10)
        results.append(result)
    return results
def savaAttention(question,qIds,regions, v2q, q2v, v2v, q2q):
    v2q = v2q.sum(-1)
    q2v = q2v.sum(-1)
    v2v = v2v.sum(-2)
    q2q = q2q.sum(-2)
    attmap = []
    lines = []
    # B = question.size(0)
    for i in range(1):
        item = {}
        q = str(question[i]).split(" ")
        q = np.pad(q,(0,15-len(q)),constant_values=("","")).tolist()
        item["question"] = q
        item["qID"] = qIds[i].item()
        lines.append(str(qIds[i].item())+","+str(question[i])+"\n")
            
        item["v2q"] = v2q[i].tolist()
        item["q2v"] = q2v[i].tolist()
        item["v2v"] = v2v[i].tolist()
        item["q2q"] = q2q[i].tolist()
        for key in ["v2q","q2v","v2v","q2q"]:
            arr = item.get(key)
            lines.append(key+"\n")
            for head in arr:
                row = ""
                for imp in head:
                    row = row +","+ str(round(imp,4))
                row = row[1:] + "\n"
                lines.append(row)
        # draw(q,item["q2q"])
        attmap.append(item)
    # with open("./attention1.json","w") as f:
    # #     # f.write(str(attmap))
        # json.dump(attmap,f)
    with open("./att_pa.csv","w") as f:
        f.writelines(lines)
def savaCount(question,qIds,regions, v2q, q2v, v2v, q2q):
    v2q = v2q.sum(-1)
    q2v = q2v.sum(-1)
    v2v = v2v.sum(-2)
    q2q = q2q.sum(-2)
    attmap = []
    # B = question.size(0)
    for i in range(1):
        item = {}
        q = str(question[i]).split(" ")
        q = np.pad(q,(0,15-len(q)),constant_values=("","")).tolist()
        item["question"] = q
        item["qID"] = qIds[i].item()
        item["v2q"] = [torch.std(torch.mean(v2q[i],-2)),torch.std(torch.std(v2q[i],-2))]
        item["q2v"] = [torch.std(torch.mean(q2v[i],-2)),torch.std(torch.std(q2v[i],-2))]
        item["v2v"] = [torch.std(torch.mean(v2v[i],-2)),torch.std(torch.std(v2v[i],-2))]
        item["q2q"] = [torch.std(torch.mean(q2q[i],-2)),torch.std(torch.std(q2q[i],-2))]
        attmap.append(item)
    print(attmap)
    # with open("./attcount.json","w") as f:
    # #     # f.write(str(attmap))
        # json.dump(attmap,f)
def parse_args():
    parser = argparse.ArgumentParser()

    '''
    For eval logistics
    '''
    parser.add_argument('--save_logits', action='store_true',
                        help='save logits')
    parser.add_argument('--save_answers', action='store_true',
                        help='save poredicted answers')

    '''
    For loading expert pre-trained weights
    '''
    parser.add_argument('--checkpoint', type=int, default=-1)
    parser.add_argument('--output_folder', type=str, default="",
                        help="checkpoint folder")
    parser.add_argument('--q_dim', type=int, default=1024)

    '''
    For dataset
    '''
    parser.add_argument('--data_folder', type=str, default='./data')
    parser.add_argument('--dataset', type=str, default='vqa',
                        choices=["vqa", "vqa_cp"])
    parser.add_argument('--split', type=str, default="val",
                        choices=["train", "val", "test", "test2015"],
                        help="test for vqa_cp, test2015 for vqa")

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    if not torch.cuda.is_available():
        raise ValueError("CUDA is not available," +
                         "this code currently only support GPU.")

    n_device = torch.cuda.device_count()
    print("Found %d GPU cards for eval" % (n_device))
    device = torch.device("cuda")
    dictionary = Dictionary.load_from_file(
                 os.path.join(args.data_folder, 'glove/dictionary.pkl'))
    relations = ["spatial" ]
    modelList = [
        # "pretrained_models/regat_implicit/ban_1_implicit_vqa_196",
        # "pretrained_models/regat_semantic/ban_1_semantic_vqa_7971",
        # "pretrained_models/regat_spatial/butd_spatial_vqa_5942",
        # "/root/code/VQA_ReGAT/saved_models/board/selfv9_vqa_9753_12_8_O_0.2/",
        # "/root/code/VQA_ReGAT/saved_models/board/selfv9_vqa_347_14_81_s2_gateQ_1560/",
        # "/root/code/VQA_ReGAT/saved_models/board/selfv9_vqa_347_13_81_noqa_AVGca_s2+_soft_avgself_2048/",
        "/root/code/VQA_ReGAT/saved_models/15+/selfv9_vqa_347_15_8_2048_noalign/"
        # "/root/code/VQA_ReGAT/saved_models/15+/selfv9_vqa_347_15_8_2048_both_/"
        # "/root/code/VQA_ReGAT/saved_models/15+/selfv9_vqa_347_15_8_2048_ALL/"
        # "/root/code/VQA_ReGAT/saved_models/15+/selfv9_vqa_347_15_8_2048_noalign_noimp/"
        # "/root/code/VQA_ReGAT/saved_models/regat_spatial/selfv8_spatial_vqa_1731_v8:h2"
    ]
    modelWeight = [
        1
        # 0.3,
        # 0.4,
        # 0.3
        ]
    checkpoints = [
        # 19,
        14
        # -1,
        # -1
        ]
    args.output_folder = modelList[0]
    hps_file = f'{args.output_folder}/hps.json'
    model_hps = Struct(json.load(open(hps_file)))
    batch_size = model_hps.batch_size*n_device

    hps_files = []
    model_hpss = []
    for m in modelList:
        f = f'{m}/hps.json'
        hps_files.append(f)
        model_hpss.append(Struct(json.load(open(f))))

    print("Evaluating on %s dataset with model trained on %s dataset" %
          (args.dataset, model_hps.dataset))
    if args.dataset == "vqa_cp":
        coco_train_features = Image_Feature_Loader(
                            'train', model_hps.relation_type,
                            adaptive=model_hps.adaptive,
                            dataroot=model_hps.data_folder)
        coco_val_features = Image_Feature_Loader(
                            'val', model_hps.relation_type,
                            adaptive=model_hps.adaptive,
                            dataroot=model_hps.data_folder)
        eval_dset = VQA_cp_Dataset(
                    args.split, dictionary, coco_train_features,
                    coco_val_features, adaptive=model_hps.adaptive,
                    pos_emb_dim=model_hps.imp_pos_emb_dim,
                    dataroot=model_hps.data_folder)
    else:
        eval_dset = VQAFeatureDataset(
                args.split, dictionary, relations,
                adaptive=model_hps.adaptive,
                pos_emb_dim=model_hps.imp_pos_emb_dim,
                dataroot=model_hps.data_folder)
    models = []
    for i in range(len(modelList)):
        model_hps = model_hpss[i]
        modelpath = modelList[i]
        model = build_model(eval_dset, model_hps).to(device)
        model = nn.DataParallel(model).to(device)
        cp = checkpoints[i]
        if cp > 0:
            checkpoint_path = os.path.join(
                                modelpath,
                                f"model_{cp}.pth")
        else:
            checkpoint_path = os.path.join(modelpath,
                                        f"model.pth")
        print("Loading weights from %s" % (checkpoint_path))
        if not os.path.exists(checkpoint_path):
            raise ValueError("No such checkpoint exists!")
        checkpoint = torch.load(checkpoint_path)
        state_dict = checkpoint.get('model_state', checkpoint)
        matched_state_dict = {}
        unexpected_keys = set()
        missing_keys = set()
        for name, param in model.named_parameters():
            missing_keys.add(name)
        for key, data in state_dict.items():
            if key in missing_keys:
                matched_state_dict[key] = data
                missing_keys.remove(key)
            else:
                unexpected_keys.add(key)
        print("\tUnexpected_keys:", list(unexpected_keys))
        print("\tMissing_keys:", list(missing_keys))
        model.load_state_dict(matched_state_dict, strict=False)
        models.append(model)

    eval_loader = DataLoader(
        eval_dset, batch_size, shuffle=False,
        num_workers=4, collate_fn=utils.trim_collate)

    eval_score = evaluate(
        models, eval_loader, model_hpss, args, device, modelWeight)

    print('\teval score: %.2f' % (100 * eval_score))
