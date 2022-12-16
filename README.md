# Long-tailed Classification via Balanced Gradient Margin Loss
This is the source code for our paper: Long-tailed Classification via Balanced Gradient Margin Loss based on Pytorch.

## Installation
**Requirements**
* Python 3.8.10
* torchvision 0.12.0
* Pytorch 1.11.0
* yacs 0.1.8
* ...

More details can be seen in requirements.txt

**Install BaGMar**
```bash
git clone https://github.com/lt-project-repository/lt-project.git
cd lt-project
pip install -r requirements.txt
```
Note that the torch version should be compatible with your cuda version, and 

**Dataset Preparation**
* [CIFAR-10 & CIFAR-100](https://www.cs.toronto.edu/~kriz/cifar.html)
* [ImageNet](http://image-net.org/index)
* [iNaturalist 2018](https://github.com/visipedia/inat_comp/tree/master/2018)
* [Places](http://places2.csail.mit.edu/download.html)

Change the dataset path in `main.py` accordingly. In the configuration file (ending with '.yaml'), norm_logits is equivalent to prediction normalization(PN), and margin_cls means Balanced Gradient Margin Loss(BaGMar loss).

## Get Started

### CIFAR100
Cross Entropy(ce)
```bash
python main.py --cfg config/CIFAR100_LT/ce_imba100.yaml
```
Cross Entropy + PN
```bash
python main.py --cfg config/CIFAR100_LT/ce_pn_imba100.yaml
```
BaGMar loss(cm)
```bash
python main.py --cfg config/CIFAR100_LT/cm_imba100.yaml
```
BaGMar loss + PN
```bash
python main.py --cfg config/CIFAR100_LT/cm_pn_imba100.yaml
```


## Results and Models
### CIFAR100

| Imbalance Factor   | Top-1 Accuracy(τ=1)       | Top-1 Accuracy(fine-tune τ) | Log           | Model |
| ----------- | ---------- | -------------- | ------------- | ----- |
| 200 | 47.4%   | 47.5%(τ=1.5)        | [link](https://drive.google.com/file/d/1qi7HEkCk1SEpgjWX2qfDJkR6x8vmGblJ/view?usp=sharing)        | [link](https://drive.google.com/file/d/1jEpJR8H8EF2idOiXkc4nh5XU6lKZjiXX/view?usp=sharing)  |
| 100 | 50.9%   |51.0%(τ=1.5)        | [link](https://drive.google.com/file/d/1LMZxARsjDVs5Leq0uCKOAJCFgjlao-8H/view?usp=sharing)       | [link](https://drive.google.com/file/d/1Wmt1PP5WMroqb9ASiljpOX8BNRS-K9Au/view?usp=sharing) |
| 50  | 55.1%  |  55.2%(τ=0.9)        | [link](https://drive.google.com/file/d/1dGV5MEue6tp85RU2zUSt08WLqe06x9R-/view?usp=sharing)        | [link](https://drive.google.com/file/d/1dGV5MEue6tp85RU2zUSt08WLqe06x9R-/view?usp=sharing)  |
| 10  | 62.5% | 62.6%(τ=1.1)        | [link](https://drive.google.com/file/d/11FNa46iEfOI7d62W7xjWQwbcUqzFaRVv/view?usp=sharing)        | [link](https://drive.google.com/file/d/1RIfiPjvx4V_QeiBDrJTW0ZAO8KjN3Z3m/view?usp=sharing)  |

### Large-scale Datasets
|  Dataset  | BaGMar loss | BaGMar loss + PN(τ) | Log | Model |
| ----------- | ---------- | -------------- | ------------- | ----- |
| ImageNet-LT | 53.4%   | 54.9%(τ=1.7)        | [train.log](https://drive.google.com/file/d/1LK66jDyKofhg1nYw4efjJbLjTc1UJ-sj/view?usp=sharing) [fine-tune-τ.log](https://drive.google.com/file/d/1uW_qsgPsU8XQpRE1p7pNXMbjQJ_eSGRC/view?usp=sharing)       | [link](https://drive.google.com/file/d/11aZuiXN0ULDn_wImEctHVcwEOSZaK10e/view?usp=sharing)  |
| iNa'2018 | 71.2%   |73.3%(τ=1.5) | [train.log](https://drive.google.com/file/d/1oqY0xa-Bxc8avT0k_TnsZEMZ5ogBBlXm/view?usp=sharing) [fine-tune-τ.log](https://drive.google.com/file/d/16-7fq73yjLOwOqKS_-c4Xci13OcLAFMK/view?usp=sharing)       | [link](https://drive.google.com/file/d/137xd182BR4qh0M5ib24UssUNUS-Tat7t/view?usp=sharing)  |
| Places-LT	  | 41.0%  |  41.4%(τ=1.4)| [train.log](https://drive.google.com/file/d/19apnKe8La2a0QECvpT7veCg92ydCoR3P/view?usp=sharing) [fine-tune-τ.log](https://drive.google.com/file/d/17tGlqvFLgBa_qs4UCeZxwgbU9VQApWEZ/view?usp=sharing)       | [link](https://drive.google.com/file/d/1tcesX6pECynXDbDPaL-G0Z0qPxosL_D0/view?usp=sharing)  |

## to do list
- [x] Support Cifar100-LT dataset
- [ ] Support imageNet-LT
- [ ] Support iNaturalist2018
- [ ] Support Places365-LT
- [x] More results and models

## Acknowledgment
We refer to some codes from [BalancedMetaSoftmax](https://github.com/jiawei-ren/BalancedMetaSoftmax-Classification). Many thanks to the authors.
