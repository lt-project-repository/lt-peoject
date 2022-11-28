# Long-tailed Classification via Balanced Gradient Margin Loss

This is the source code for our paper: Long-tailed Classification via Balanced Gradient Margin Loss based on Pytorch.

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
BaGMar loss
```bash
python main.py --cfg config/CIFAR100_LT/BaGMar_imba100.yaml
```
BaGMar loss + PN
```bash
python main.py --cfg config/CIFAR100_LT/BaGMar_pn_imba100.yaml
```

## Results and Models for CIFAR100

| Imbalance Factor   | Top-1 Accuracy(τ=1)       | Top-1 Accuracy(fine-tune τ) | Log           | Model |
| ----------- | ---------- | -------------- | ------------- | ----- |
| 200 | 47.4%   | 47.5%(τ=1.5)        | [link](https://drive.google.com/file/d/1qi7HEkCk1SEpgjWX2qfDJkR6x8vmGblJ/view?usp=sharing)        | [link](https://drive.google.com/file/d/1jEpJR8H8EF2idOiXkc4nh5XU6lKZjiXX/view?usp=sharing)  |
| 100 | 50.9%   |51.0%(τ=1.5)        | [link](https://drive.google.com/file/d/1LMZxARsjDVs5Leq0uCKOAJCFgjlao-8H/view?usp=sharing)       | [link](https://drive.google.com/file/d/1Wmt1PP5WMroqb9ASiljpOX8BNRS-K9Au/view?usp=sharing) |
| 50  | 55.1%  |  55.2%(τ=0.9)        | [link](https://drive.google.com/file/d/1dGV5MEue6tp85RU2zUSt08WLqe06x9R-/view?usp=sharing)        | [link](https://drive.google.com/file/d/1dGV5MEue6tp85RU2zUSt08WLqe06x9R-/view?usp=sharing)  |
| 10  | 62.5% | 62.6%(τ=1.1)        | [link](https://drive.google.com/file/d/11FNa46iEfOI7d62W7xjWQwbcUqzFaRVv/view?usp=sharing)        | [link](https://drive.google.com/file/d/1RIfiPjvx4V_QeiBDrJTW0ZAO8KjN3Z3m/view?usp=sharing)  |

## To do list
- [x] Support Cifar100-LT dataset
- [ ] Support imageNet-LT
- [ ] Support iNaturalist2018
- [ ] Support Places365-LT
- [ ] More results and models
- 
## Acknowledgment
We refer to some codes from [BalancedMetaSoftmax](https://github.com/jiawei-ren/BalancedMetaSoftmax-Classification). Many thanks to the authors.
