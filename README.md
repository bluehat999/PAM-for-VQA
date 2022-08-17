# Path-wise Attention Memory Network for Visual Question Answering

This repository is the implementation of [Path-wise Attention Memory Network for Visual Question Answering]().

<!-- ![Overview of ReGAT](misc/regat_overview.jpg) -->

## Prerequisites

You may need a machine with 4 GPUs with 16GB memory each, and PyTorch v1.0.1 for Python 3.

1. Install [PyTorch](http://pytorch.org/) with CUDA10.0 and Python 3.7.
2. Install [h5py](http://docs.h5py.org/en/latest/build.html).
3. Install [block.bootstrap.pytorch](https://github.com/Cadene/block.bootstrap.pytorch).

If you are using miniconda, you can install all the prerequisites with `tools/environment.yml`.

## Data

Our implementation uses the pretrained features from [bottom-up-attention](https://github.com/peteanderson80/bottom-up-attention), the adaptive 10-100 features per image. In addition to this, the GloVe vectors and Visual Genome question answer pairs. For your convenience, the below script helps you to download preprocessed data.

```bash
source tools/download.sh
```

In the end, the data folder should be organized as shown below:

```bash
├── data
│   ├── Answers
│   │   ├── v2_mscoco_train2014_annotations.json
│   │   └── v2_mscoco_val2014_annotations.json
│   ├── Bottom-up-features-adaptive
│   │   ├── train.hdf5
│   │   ├── val.hdf5
│   │   └── test2015.hdf5
│   ├── Bottom-up-features-fixed
│   │   ├── train36.hdf5
│   │   ├── val36.hdf5
│   │   └── test2015_36.hdf5
│   ├── cache
│   │   ├── cp_v2_test_target.pkl
│   │   ├── cp_v2_train_target.pkl
│   │   ├── train_target.pkl
│   │   ├── val_target.pkl
│   │   ├── trainval_ans2label.pkl
│   │   └── trainval_label2ans.pkl
│   ├── cp_v2_annotations
│   │   ├── vqacp_v2_test_annotations.json
│   │   └── vqacp_v2_train_annotations.json
│   ├── cp_v2_questions
│   │   ├── vqacp_v2_test_questions.json
│   │   └── vqacp_v2_train_questions.json
│   ├── glove
│   │   ├── dictionary.pkl
│   │   ├── glove6b_init_300d.npy
│   │   └──- glove6b.300d.txt
│   ├── imgids
│   │   ├── test2015_36_imgid2idx.pkl
│   │   ├── test2015_ids.pkl
│   │   ├── test2015_imgid2idx.pkl
│   │   ├── train36_imgid2idx.pkl
│   │   ├── train_ids.pkl
│   │   ├── train_imgid2idx.pkl
│   │   ├── val36_imgid2idx.pkl
│   │   ├── val_ids.pkl
│   │   └── val_imgid2idx.pkl
│   ├── Questions
│   │   ├── v2_OpenEnded_mscoco_test-dev2015_questions.json
│   │   ├── v2_OpenEnded_mscoco_test2015_questions.json
│   │   ├── v2_OpenEnded_mscoco_train2014_questions.json
│   │   └── v2_OpenEnded_mscoco_val2014_questions.json
│   ├── visualGenome
│   │   ├── image_data.json
│   │   └── question_answers.json
```

## Training

```bash
python main.py --config config/default.json
```

## Evaluating

```bash
# take ban_1_implicit_vqa_196 as an example
# to evaluate cp_v2 performance, need to use --dataset cp_v2 --split test
python eval.py --output_folder pretrained_models/regat_implicit/ban_1_implicit_vqa_196 --save_answers --split test2015
python eval.py --save_answers --split test2015
```

## Citation

If you use this code as part of any published research, we'd really appreciate it if you could cite the following paper:

<!-- ```text
@article{pam2022,
  title={Path-wise Attention Memory Network for Visual Question Answering},
  author={Yingxin Xiang, Chengyuan Zhang, Zhichao Han, Hao Yu, Jiaye Li, Lei Zhu},
  journal={Mathematics},
  year={2022}
}
``` -->

## License
