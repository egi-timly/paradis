<h1  align="center" ><b> Para</b>meter <b>Dis</b>tribution Shift NAS</h1>

<div align=justify>

This is the official code repository for the <b>ParaDis</b> method and its associated paper [<b>Insights from a Space- and Data- Agnostic Approach to Zero-Cost Proxies</b>](https://openreview.net/pdf?id=sVWJczov4Q) <b>(TMLR 2026)</b>.

ParaDis is a genetic-based zero-cost NAS framework which drives the search with Zero-Cost Proxies (ZCPs) by separating them into the explorer and discriminator roles. Specifically, we use a parameter-based discriminator and pair it with various SOTA ZCPs.

This repository contains:
- code for the genetic algorithm to produce final populations of the search in the NAS-Bench-Suite-Zero benchmark and the DARTS space, under two settings: `paradis` and `baseline`. `paradis` scripts search using the dedicated ParaDis discriminator, while `baseline` scripts search by pairing 2 specified metrics. In the `baseline` scripts, using the same metric for `metric1` and `metric2` emulates the use of a single ZCP for the whole search as in the paper's baseline experiments.
- dependency code forked from the [NASLib ZC](https://github.com/automl/NASLib/tree/zerocost) repository. All rights to these code templates belong to the authors of the benchmark paper [NAS-Bench-Suite-Zero: Accelerating Research on Zero-Cost Proxies](https://proceedings.neurips.cc/paper_files/paper/2022/file/b3835dd49b7d5bb062aecccc14d8a675-Supplemental-Datasets_and_Benchmarks.pdf).
- implementations of all ZCPs presented in the paper in the NBS0 framework.
- implementations of mutation and crossover operations for all search spaces within the NBS0 framework.
- re-implementation of DARTS in the NBS0 framework + mutation and crossover operations.

<h3 align="center"> Setup & Data </h3>

These setup instructions are largely ported over from the NBS0 setup.

Creating a new conda environment is recomended. [Install PyTorch GPU/CPU](https://pytorch.org/get-started/locally/) for your setup.

```bash
conda create -n zerocost  python=3.7
conda activate zerocost
```

Install all requirements.

```bash
pip install -r requirements.txt
```

Download all the ZC proxies evaluations, which contains the scores for each proxy and validation accuracy for each architecture. The ```gdown (pip install gdown)``` package is required to download from google drive. The following command will download the data.

```bash
source scripts/download_nbs_zero.sh <search_space>
source scripts/download_nbs_zero.sh nb201
source scripts/download_nbs_zero.sh all
```

Download all the NAS benchmarks and their associated datasets.
```bash
source scripts/download_data.sh all 
```
Alternatively, you can download the benchmark for a specific search space and dataset/task as follows:
```bash
source scripts/download_data.sh <search_space> <dataset> 
source scripts/download_data.sh nb201 cifar10
source scripts/download_data.sh nb201 all 
```
<h3 align="center"> Experiments </h3>

How to run ParaDis on NBS0:
```bash
python3 paradis_main.py --search_space <search_space> --dataset <dataset> --metric1 <explorer> --crossover
```

How to run the baselines on NBS0:
```bash
python3 baseline.py --search_space <search_space> --dataset <dataset> --metric1 <explorer> --metric2 <discriminator> --crossover
```

How to run ParaDis on DARTS:
```bash
python3 paradis_darts.py --dataset <dataset> --metric1 <explorer> --crossover
```

How to run the baselines on DARTS:
```bash
python3 baseline_darts.py --dataset <dataset> --metric1 <explorer> --metric2 <discriminator> --crossover
```

By default, all hyperparameters are the same as the paper's main experiments. These hyperparameters can be modified in the command line (see associated scripts for arguments).

The script will produce logs and output files in the root folder. The location can be modified using the `--log_path` and `--output_path` arguments:
- `XXX_<search_space>_<dataset>_<metrics>_<timestamp>.log` contains the logs of the genetic search.
- `XXX_<search_space>_<dataset>_<metrics>_<timestamp>.csv` contains a table summary of all architectures generated (by mutation or crossover) during the search.
- `XXX_<search_space>_<dataset>_<metrics>_<timestamp>_pophist.csv` contains history of the population at the end of each evolutionary step. The end population of the search can be fetched from the last N lines of this output file.

</div>
