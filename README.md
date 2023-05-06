# Gradient-Aware Logit Adjustment Loss for Long-tailed Classifier
This is the source code for our paper: Gradient-Aware Logit Adjustment Loss for Long-tailed Classifier based on Pytorch.

## Installation
**Requirements**
* Python 3.8.10
* torchvision 0.12.0
* Pytorch 1.11.0
* yacs 0.1.8
* ...

More details can be seen in requirements.txt

**Install GALA**
```bash
git clone https://github.com/lt-project-repository/lt-project.git
cd lt-project
pip install -r requirements.txt
```
Note that the torch version should be compatible with your cuda version.  In the configuration file (ending with '.yaml'), `norm_logits` is equivalent to Prediction Re-balancing Strategy, and `margin_cls` means GALA Loss.

**Dataset Preparation**
* [CIFAR-10 & CIFAR-100](https://www.cs.toronto.edu/~kriz/cifar.html)
* [ImageNet](http://image-net.org/index)
* [iNaturalist 2018](https://github.com/visipedia/inat_comp/tree/master/2018)
* [Places](http://places2.csail.mit.edu/download.html)

Change the dataset path in `main.py` accordingly.

## Get Started

### CIFAR100
Cross Entropy(ce)
```bash
python main.py --cfg config/CIFAR100_LT/ce_imba100.yaml
```
Cross Entropy + Prediction Re-balancing Strategy
```bash
python main.py --cfg config/CIFAR100_LT/ce_pn_imba100.yaml
```
GALA loss
```bash
python main.py --cfg config/CIFAR100_LT/gala_imba100.yaml
```
GALA loss + Prediction Re-balancing Strategy
```bash
python main.py --cfg config/CIFAR100_LT/gala_pn_imba100.yaml
```


## Results and Models
### CIFAR100
|  Imbalance Factor  |  GALA loss  |  GALA loss + Prediction Re-balancing Strategy  |  Log  |  Model  |
| :------: | :------: | :------: | :------: | :------: |
|  200  |  48.0%  |  48.5%  |  [link](https://drive.google.com/file/d/1kg3eH71Zs5MhbmtlqKtv9YNgruBBjjOj/view?usp=sharing)  | [link](https://drive.google.com/file/d/1lDfddC2idKjjblrivlZKRY2RNuUUvjAR/view?usp=sharing)  |
|  100  |  52.1%  |  52.3%  |  [link](https://drive.google.com/file/d/1NyfxLGIddnfwORWJi5D-PLP-6vqShPAb/view?usp=sharing)  |  [link](https://drive.google.com/file/d/1QRhQPS9U-m-CvlUfsEOaarkR5RS_vP3R/view?usp=sharing)  |
|  50   |  56.0%  |  56.5%  |  [link](https://drive.google.com/file/d/1z-wBhclun8nbiJU-pVn043OC5OSZ51yS/view?usp=sharing)  |  [link](https://drive.google.com/file/d/16R6QC7kQvjjX8ej86XJ-qlO_pVHcJG94/view?usp=sharing)  |
|  10   |  63.8%  |  64.2%  |  [link](https://drive.google.com/file/d/1TRNyaW73NSUugUCbMQHnWX0ny6sX3hix/view?usp=sharing)  |  [link](https://drive.google.com/file/d/1ilLqFJzGX4k-QeD1V2TAW6saS5cGyqqW/view?usp=sharing)  |

### Large-scale Datasets
|  Dataset  | GALA loss | GALA loss + Prediction Re-balancing Strategy(τ) | Log | Model |
| :------: | :------: | :------: | :------: | :------: |
| ImageNet-LT | 53.4%   | 54.9%(τ=1.7)        | [train.log](https://drive.google.com/file/d/1LK66jDyKofhg1nYw4efjJbLjTc1UJ-sj/view?usp=sharing)   [fine-tune-τ.log](https://drive.google.com/file/d/1uW_qsgPsU8XQpRE1p7pNXMbjQJ_eSGRC/view?usp=sharing)       | [link](https://drive.google.com/file/d/11aZuiXN0ULDn_wImEctHVcwEOSZaK10e/view?usp=sharing)  |
| iNa'2018 | 71.2%   |73.3%(τ=1.5) | [train.log](https://drive.google.com/file/d/1oqY0xa-Bxc8avT0k_TnsZEMZ5ogBBlXm/view?usp=sharing)   [fine-tune-τ.log](https://drive.google.com/file/d/16-7fq73yjLOwOqKS_-c4Xci13OcLAFMK/view?usp=sharing)       | [link](https://drive.google.com/file/d/137xd182BR4qh0M5ib24UssUNUS-Tat7t/view?usp=sharing)  |
| Places-LT	  | 41.0%  |  41.4%(τ=1.4)| [train.log](https://drive.google.com/file/d/19apnKe8La2a0QECvpT7veCg92ydCoR3P/view?usp=sharing)   [fine-tune-τ.log](https://drive.google.com/file/d/17tGlqvFLgBa_qs4UCeZxwgbU9VQApWEZ/view?usp=sharing)       | [link](https://drive.google.com/file/d/1tcesX6pECynXDbDPaL-G0Z0qPxosL_D0/view?usp=sharing)  |

## to do list
- [x] Support CIFAR100-LT dataset
- [ ] Support ImageNet-LT
- [ ] Support iNaturalist2018
- [ ] Support Places365-LT
- [x] More results and models

## Acknowledgment
We refer to some codes from [BalancedMetaSoftmax](https://github.com/jiawei-ren/BalancedMetaSoftmax-Classification). Many thanks to the authors.
