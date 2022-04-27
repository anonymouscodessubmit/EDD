# PCD and EDD
Source code for PCD-DARTS and EDD-DARTS

## version: pytorch 1.7.0

## NAS-BENCH-201:  
The performance database file "NAS-Bench-201-v1_1-096897.pth" of NAS-BENCH-201 is the prerequisite.
Please download from https://github.com/D-X-Y/NAS-Bench-201 and replace the file path in the startup script.

`
cd AutoDL-Projects_edd/;
bash ./scripts-search/algos/[DARTS-V1-100.sh|DARTS-V1-10.sh] [cifar10|cifar100] 1 -1
`

`
cd AutoDL-Projects_pcd/;
bash ./scripts-search/algos/[DARTS-V1-100.sh|DARTS-V1-10.sh] [cifar10|cifar100] 1 -1
`

## DARTS search space:  
`
cd PCDEDD-DARTS/
`

`
python train_search_cyc.py
`

`
python train_search_pl.py
`

## S1-S4 search space:  
`
cd SmoothDARTS/sota/pcdedd
`

`
python train_search_cyc.py --search_space=[s1|s2|s3|s4] --decay_strength=[0.1|0.11|0.12|0.13] --dataset [cifar10|cifar100|svhn]
`

`
python train_search_pl.py --search_space=[s1|s2|s3|s4] --decay_strength=[0.12|0.24|0.48|0.96] --dataset [cifar10|cifar100|svhn]
`
