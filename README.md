# Domain Watermark: Effective and Harmless Dataset Copyright Protection is Closed at Hand

This repository contains the official implementation required to replicate the primary results presented in our paper [Domain Watermark: Effective and Harmless Dataset Copyright Protection is Closed at Hand](https://www.researchgate.net/publication/374440504_Domain_Watermark_Effective_and_Harmless_Dataset_Copyright_Protection_is_Closed_at_Hand) in NeurIPS 2023. This research project is developed based on Python 3 and Pytorch, created by [Junfeng Guo](https://junfenggo.github.io/) and [Yiming Li](http://liyiming.tech/).

Our implementation is based on [Industrial Scale Data Poisoning via Gradient Matching](https://github.com/JonasGeiping/poisoning-gradient-matching).

If our work or this repo is useful for your research, please cite our paper as follows:

```
@inproceedings{guo2023domain,
  title={Domain Watermark: Effective and Harmless Dataset Copyright Protection is Closed at Hand},
  author={Guo, Junfeng and Li, Yiming and Wang, Lixu and Xia, Shu-Tao and Huang, Heng and Liu, Cong and Li, Bo},
  booktitle={NeurIPS},
  year={2023}
}
```


## Dependencies

- PyTorch => 1.6.*
- torchvision > 0.5.*
- higher [best to directly clone https://github.com/facebookresearch/higher and use ```pip install .```]





## USAGE

The wrapper for the Domain Watermark can be found in dw.py. The default values are set for attacking ResNet-18 on CIFAR-10.

There are a buch of optional arguments in the ```forest/options.py```. Here are some of them:

- ```--patch_size```, ```--eps```, and ```--budget``` : determine the power of backdoor attack.
- ```--dataset``` : which dataset to poison.
- ```--net``` : which model to attack on.
- ```--retrain_scenario``` : enable the retraining during poison crafting.
- ```--poison_selection_strategy``` : enables the data selection (choose ```max_gradient```)
- ```--ensemble``` : number of models used to craft poisons.
- ```--sources``` : Number of sources to be triggered in inference time.




## Evaluation

We here give a demonstration for implementation of Domain Watermark. Before runing our code, please first download the hardly-generalized domain samples for CIFAR-10 through:  https://www.dropbox.com/sh/hry5v7fxzzxcfr0/AADolCGag9DvY0RQaCzPsBVfa?dl=0   

After downloading the hardly-generalized domains samples and set the path, you can change the path in "Domain_Watermark/forest/witchcoven/witch_base.py" for each dataset. We here only provide CIFAR-10 task for evaluation. 

After setting the path, you can launch the programm by:

bash run.sh 

We also provide a set of trained model in the saved_models file in https://www.dropbox.com/s/lulp90pp4iey75t/saved_models.zip?dl=0  You can use the benign model (model_benign.pth) and watermarked model (model_DW.pth) for evaluating our approach. 


The watermarked samples can be obtained and examined by runing:

python test_eval.py
