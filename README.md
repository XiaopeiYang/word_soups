# Word and Descriptor Soups 🍜 [[ArXiv]](https://arxiv.org/pdf/2311.13612.pdf)
-----------------------------------------------------


Code in this repo uses code from [multimodal prompt learning](https://github.com/muzairkhattak/multimodal-prompt-learning), which in turn uses code from [Co-CoOp and CoOp](https://github.com/KaiyangZhou/CoOp).



## ⏳ Installation
-------------------

* Install dassl library and other requirements.
```bash
# Instructions borrowed from https://github.com/KaiyangZhou/Dassl.pytorch#installation

git clone https://github.com/KaiyangZhou/Dassl.pytorch.git
cd Dassl.pytorch/
pip install -r requirements.txt
python setup.py develop
cd ..

pip install open_clip_torch
pip install pytorch_metric_learning
```

* Create a directory somewhere called `data/`. Download imagenet file from [this shared Google Drive](https://drive.google.com/drive/folders/1kvh5VG4ruGOcSiHKJX9dWJhPAGVgPSZs?usp=drive_link) and unzip it into `data/`. The resulting file tree should look like:
```
data/
|-- imagenet
|-- images4LMU (the folder of fungi-images)
```

* The file '\source\datasets\fungi_small.py' is primarily used for sampling the "FungiSmall" dataset.

Modify the following two lines in `argparse_parameters.py` to reflect where you have your `data/` dir and where you want the pretrained CLIP weights to be cached (which could be many gigabytes)

```python
parser.add_argument('--cache_dir', default = "", type =str) # set to directory where you want large pretrained model weights to be cached
parser.add_argument('--data_dir', default = "", type =str)  # set to parent directory of data/
```
Modify the following one line in `argparse_parameters.py` to reflect whether you use patches for fungi dataset.

```python
parser.add_argument('--use_patches', default = True, type =bool) # 
```
Modify the following one line in `source/gpt_helpers.py` to reflect which gpt description file is used.
The files are under the folder `/gpt_descriptors`
```python
descriptor_fname_dict = {
    'ImageNet':'descriptors_imagenet',
    Fungi_dataset:'fungi_descriptions/fungi_descriptions11_model7b'#change
}
```

## 🍜 Descriptor soups
---------------------------

### (1) Generate Description Features
First, calculate the descriptor features on ImageNet.
Use `preprocess/generate_description_features.py`.
This python file reads from `preprocess/descriptions.list`, 
which is a sorted list of 4227 unique GPT descriptors. They begin with a space and end in a period.
Currently, we use a pretrained model for these features.

**Run:** `python preprocess/generate_description_features.py --dataset ImageNet`

This will save the tuple of description strings, 
description features in `cache/description_features__ViT-B-16_openai.tensor`

### (2) Calculate greedy descriptor soups
This needs to be done for each random seed of ImageNet training split! 

**Run:** 

```bash
python preprocess/get_greedy_descriptor_soup.py --dataset ImageNet --seed 1
python preprocess/get_greedy_descriptor_soup.py --dataset ImageNet --seed 2
python preprocess/get_greedy_descriptor_soup.py --dataset ImageNet --seed 3
```

This will save the greedily selected descriptors in `cache/good_descriptions_seed1__ViT-B-16_openai.list` as a list.

**Example logs:** `example_logs/example_get_greedy_descriptor_soup_output.txt`

Proceed to **Zero-shot comparisons** section for evaluation.

## 🍜 Word soups
--------------------

In the paper, descriptor/word soup is trained on imagenet and shown to have good generalization capabilities. 

### (1) Get Word Features
`preprocess/words.list` contains 10,000 most common English words minus swear words. They have a space prepended. We can use the same `preprocess/generate_description_features.py` to generate the text features from individual words.

**Run:** ```python preprocess/generate_description_features.py --dataset ImageNet --descriptions preprocess/words.list --savename word_features ```

This will save the tuple or words and word features in `cache/word_features__ViT-B-16_openai.tensor`

### (2) Calculate greedy word soups
This needs to be done for each random seed of ImageNet training split!

**Run:**

```bash
python preprocess/get_greedy_word_soup.py --dataset ImageNet --seed 1 --n_descriptors 8
python preprocess/get_greedy_word_soup.py --dataset ImageNet --seed 2 --n_descriptors 8
python preprocess/get_greedy_word_soup.py --dataset ImageNet --seed 3 --n_descriptors 8
```

This will save the greedily selected descriptors in `cache/word_soup_descriptors_seed1__ViT-B-16_openai.list` as a list.

**Example logs:** `example_logs/example_get_greedy_word_soup_output.txt`

Proceed to **Zero-shot comparisons** section for evaluation.

## 🧪 Baselines
-----------------

Results are outputted in CSV format at the end of the experiment. You can copy and paste directly into a spreadsheet.

### Zero-shot comparisons

For all ZS methods presented in Table 3 of the paper (Open-AI handcrafted ensemble, GPT, Raw-GPT, descriptor soup, token offest, word soup), run: 

```bash
sh scripts/run_pt_eval.sh 0 ViT-B-16 openai 512 
```

**Example logs:** `example_logs/example_run_pt_eval_ViT-B-16_openai_output_withpatches.txt` 



For WaffleCLIP with 16 members, run:

The prompt of WaffleCLIP "A photo of a {c}, which (is/has/etc) {random sequence}.".
In WaffleCLIP, Rothet al[https://arxiv.org/abs/2306.07282]. claim that most of the gain in accuracy reported by Menon and Vondrick[https://arxiv.org/abs/2210.07183] can be attributed to prompt ensembling

```bash
sh scripts/waffle_descriptors_eval.sh 16
```

**Example logs:** `example_logs/example_waffle_descriptors_eval_output_withpatches.txt`

### Few-shot OOD comparisons

These scripts train on 3 random splits of 16-shot ImageNet-1K. **"XD(cross dataset) "** stands for test accuracy on an OOD(out of distribution) dataset (FungiSmall). I didn't get the result, because CUDA is out of memory on my cip pool.   

| Method | Command to run | XD  | 
| ------ | -------------- | ------ | 
| CLIP-adapter | `scripts/run_adapter.sh 6e-3 ViT-B-16 512` |
| bitfit | `scripts/bitfit.sh 1.25e-4 ViT-B-16 512` | 
| Cross Entropy | `scripts/run_ce.sh 2e-5 ViT-B-16 512` | 
| Cross Entropy + word soup + diversity loss | `scripts/run_ce_regularized.sh 0.25 10` | 
| ClipOOD | `scripts/run_clipood.sh 2e-5 ViT-B-16 512` | 
| ClipOOD + word soup + diversity loss | `scripts/run_clipood_regularized.sh 0.25 10` | 
| CoOp | `scripts/run_coop.sh 8e-5 ViT-B-16 512` |
| CoOp + word soup + diversity loss | `scripts/run_coop_regularized.sh 0.25 10` | 
| KgCoOp |  `scripts/run_kgcoop.sh 4e-5 ViT-B-16 512` | 
| LoRA |  `scripts/run_lora.sh 1e-5 ViT-B-16 512` | 
| MaPLe |  `scripts/run_maple.sh 0.025 ViT-B-16 512` |
| MaPLe + word soup + diversity loss |  `scripts/run_maple_regularized.sh` | 
| ProDA |  `scripts/run_proda.sh 3.2e-4 ViT-B-16 512` | 
| ProGrad |  `scripts/run_prograd.sh 1.28e-3 ViT-B-16 512` |
| ResBlock-adapter | `scripts/run_resblock_adapter.sh 2.5e-3 ViT-B-16 512` | 
| SSF | `scripts/run_ssf.sh 1e-4 ViT-B-16 512` |
| VPT | `scripts/run_vpt_deep.sh 0.8 ViT-B-16 512` | 

## 🧪 More experiments
-----------------------------

### Base to novel setting

First, generate features for the FungiSmall training dataset:

For descriptor features:

```bash
  python preprocess/generate_description_features.py --dataset FungiSmall --subsample_classes base

```
This will save the tuple of description strings, 
description features in `cache/FungiSmalldescription_features__ViT-B-16_openai.tensor`

For word features:

```bash
  python preprocess/generate_description_features.py --dataset FungiSmall --descriptions preprocess/words.list --savename word_features --subsample_classes base
```
This will save the tuple or words and word features in `cache/FungiSmallword_features__ViT-B-16_openai.tensor`

To get greedy descriptor soup:

```bash
  sh scripts/ablations/run_get_greedy_descriptor_soup.sh FungiSmall

```
if
```python
parser.add_argument('--use_patches', default = True, type =bool) # 
```
This will save the greedily selected descriptors in `cache/FungiSmall_usepatches_Truegood_descriptions_seed1__ViT-B-16_openai.list` as a list.

if
```python
parser.add_argument('--use_patches', default = False, type =bool) # 
```
This will save the greedily selected descriptors in `cache/FungiSmall_usepatches_Falsegood_descriptions_seed1__ViT-B-16_openai.list` as a list.

To get greedy word soup:

```bash
  sh scripts/ablations/run_get_greedy_word_soup.sh FungiSmall

```

if
```python
parser.add_argument('--use_patches', default = True, type =bool) # 
```
This will save the greedily selected descriptors in `cache/FungiSmall_usepatches_Trueword_soup_descriptors_seed1__ViT-B-16_openai.list` as a list.

if
```python
parser.add_argument('--use_patches', default = False, type =bool) # 
```
This will save the greedily selected descriptors in `cache/FungiSmall_usepatches_Falseword_soup_descriptors_seed1__ViT-B-16_openai.list` as a list.

Then run training using provided bash scripts, example:

a pre-trained model using cross-entropy on a subsample of the FungiSmall dataset, and then evaluation.

```sh scripts/run_ce_with_eval.btn.sh 5e-05 > run_ce_with_eval.btn.sh_5e-05.o ```

**Example logs:** `example_logs/examples_run_ce_with_eval.btn.sh_5e-05_usepatchesfalsefungiseed_images.txt`,`example_logs/examples_run_ce_with_eval.btn.sh_5e-05_usepatchesfalsefungiseed_patches.txt`

To modify the following lines in main_novelclasses.py script to use different  "description/word soup" seed lists generated by patches or full images from the FungiSmall dataset.
``` python
else:
    default_word_descriptor_file = 'cache/{}_usepatches_{}word_soup_descriptors_seed{}__{}_{}.list'.format(
        args.dataset,
        False, #change this line for use_patches
        args.seed, 
        args.modelname, 
        args.pretrained
    )
    default_desc_descriptor_file = 'cache/{}_usepatches_{}good_descriptions_seed{}__{}_{}.list'.format(
        args.dataset,
        False, #change this line for use_patches
        args.seed, 
        args.modelname, 
        args.pretrained
    )
```
See any bash script called `scripts/*.btn.sh`.

 

### More baselines

Many more baselines in the `scripts/ablations` folder. Run these at your pleasure.
