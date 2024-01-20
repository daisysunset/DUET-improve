# DUET
- **Original Paper**
  - [*DUET: Cross-modal Semantic Grounding for Contrastive Zero-shot Learning*](https://arxiv.org/abs/2207.01328)
- **Original github repository**
  - [DUET-github](https://github.com/zjukg/DUET)

## 📕 Code Path

#### Code Structures
There are four parts in the code.
- **model**: It contains the main files for DUET network.
- **data**: It contains the data splits for different datasets.
- **cache**: It contains some cache files.
- **script**: The training scripts for DUET.

```shell
DUET
├── AwA2
├── cache
│   ├── AWA2
│   │   ├── attributeindex2prompt.json
│   │   └── id2imagepixel.pkl
│   ├── CUB
│   │   ├── attributeindex2prompt.json
│   │   ├── id2imagepixel.pkl
│   │   └── mapping.json
│   └── SUN
│   │   ├── attributeindex2prompt.json
│   │   ├── id2imagepixel.pkl
│   │   └── mapping.json
├── CUB_200_2011
├── data
│   ├── AWA2
│   │   ├── APN.mat
│   │   ├── TransE_65000.mat
│   │   ├── att_splits.mat
│   │   ├── attri_groups_9.json
│   │   ├── kge_CH_AH_CA_60000.mat
│   │   └── res101.mat
│   ├── CUB
│   │   ├── APN.mat
│   │   ├── att_splits.mat
│   │   ├── attri_groups_8.json
│   │   ├── attri_groups_8_layer.json
│   │   └── res101.mat
│   └── SUN
│       ├── APN.mat
│       ├── att_splits.mat
│       ├── attri_groups_4.json
│       └── res101.mat
├── log
│   ├── AWA2
│   ├── CUB
│   └── SUN
├── model
│   ├── log.py
│   ├── main.py
│   ├── main_utils.py
│   ├── model_proto.py
│   ├── modeling_lxmert.py
│   ├── opt.py
│   ├── swin_modeling_bert.py
│   ├── util.py
│   └── visual_utils.py
├── PLMs
├── out
│   ├── AWA2
│   ├── CUB
│   └── SUN
└── script
    ├── AWA2
    │   └── AWA2_GZSL.sh
    ├── CUB
    │   └── CUB_GZSL.sh
    └── SUN
        └── SUN_GZSL.sh
├── SUN
```

## 🔬 Dependencies

- ```Python 3```
- ```PyTorch >= 1.8.0```
- ```Transformers>= 4.11.3```
- ```NumPy```
- All experiments are performed with one RTX 3090Ti GPU.
- run following script to create conda virtual envonriment
```shell
conda env create -f cm.yaml
```



## 🎯  Prerequisites
- **Dataset**: 
  - original dataset:  [CUB](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html), [AWA2](https://cvml.ist.ac.at/AwA2/), [SUN](https://cs.brown.edu/~gmpatter/sunattributes.html), and change the ```opt.image_root``` to the dataset root path on your machine
  - For other required feature files: `APN.mat` in `data/`(https://github.com/wenjiaXu/APN-ZSL) and `id2imagepixel.pkl` in `cache/`(https://pan.baidu.com/s/13oyLDNm6uoYpVgcMitrY-A, [`Baidu cloud`, **`19.89G`**, Code: `s07d`]) 
- **Data split**:
  - please download the data folder and place it in ```./data/```.
  - ```Attributeindex2prompt.json``` should generate and place it in ```./cache/dataset/```.
- **Retrained vision Transformer as the vision encoder**
  - [deit-base-distilled-patch16-224](https://huggingface.co/facebook/deit-base-distilled-patch16-224)
  - [swin_base_patch4_window7_224.pth](https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window7_224.pth)
- **Pretrained Language Model**
  - [bert](https://huggingface.co/bert-base-uncased)
  - [roberta](https://huggingface.co/roberta-base)
  - download these models and you need to change path in opt


## 🚀 Train & Eval

The training script for **AWA2_GZSL**:
```shell
bash script/AWA2/AWA2_GZSL.sh
```

### [Parameter](#content)
```
[--dataset {AWA2, SUN, CUB}] [--calibrated_stacking CALIBRATED_STACKING] [--nepoch NEPOCH] [--batch_size BATCH_SIZE] [--manualSeed MANUAL_SEED]
[--classifier_lr LEARNING-RATE] [--xe XE] [--attri ATTRI] [--gzsl] [--patient PATIENT] [--model_name MODEL_NAME] [--mask_pro MASK-PRO] 
[--mask_loss_xishu MASK_LOSS_XISHU] [--xlayer_num XLAYER_NUM] [--construct_loss_weight CONSTRUCT_LOSS_WEIGHT] [--sc_loss SC_LOSS] [--mask_way MASK_WAY]
[--attribute_miss ATTRIBUTE_MISS]
[--prefer_high]
[--langM_path]
[--langMtokenizer_path]
[--visualM_deit_path]
[--visualM_swin_path]
```