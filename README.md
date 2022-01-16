# SSDNet

Author: [Yang Lin](https://yanglin1997.github.io/)

E-mail: linyang1997@yahoo.com.au

===========================================================================

A **PyTorch** implementation of **SSDNet (ICDM 2021)**.

<div style="text-align:center"><img src ="SSDNet.jpg" ,width=100/></div>

## Abstract
<p align="justify">
In this paper, we present SSDNet, a novel deep learning approach for time series forecasting. SSDNet combines the Transformer architecture with state space models to provide probabilistic and interpretable forecasts, including trend and seasonality components and previous time steps important for the prediction. The Transformer architecture is used to learn the temporal patterns and estimate the parameters of the state space model directly and efficiently, without the need for Kalman filters. We comprehensively evaluate the performance of SSDNet on five data sets, showing that SSDNet is an effective method in terms of accuracy and speed, outperforming state-of-the-art deep learning and statistical methods, and able to provide meaningful trend and seasonality components.</p>

This repository provides an implementation for SSDNet as described in the paper:

> SSDNet: State Space Decomposition Neural Network for Time Series Forecasting.
> Yang Lin, Irena Koprinska, Mashud Rana
> ICDM, 2021.
> [[Paper]](https://arxiv.org/pdf/2112.10251.pdf)

**Citing**

If you find SSDNet and the new datasets useful in your research, please consider adding the following citation:

```bibtex
@inproceedings{Yang21SSDNet,
              author    = {Yang Lin and Irena Koprinska and Mashud Rana},
              title     = {SSDNet: State Space Decomposition Neural Network for Time Series Forecasting},
              year = {2021},
              booktitle={Proceedings of the IEEE International Conference on Data Mining (ICDM)},
}
```

## List of Implementations:

Sanyo: http://dkasolarcentre.com.au/source/alice-springs/dka-m4-b-phase

Hanergy: http://dkasolarcentre.com.au/source/alice-springs/dka-m16-b-phase

Solar: https://www.nrel.gov/grid/solar-power-data.html

Electricity: https://archive.ics.uci.edu/ml/datasets/ElectricityLoadDiagrams20112014

Exchange: https://github.com/laiguokun/multivariate-time-series-data/tree/master/exchange_rate



## To run:

1. Preprocess the data:
  
   ```bash
   python preprocess_Sanyo.py
   python preprocess_Hanergy.py
   python preprocess_solar.py
   python preprocess_elect.py
   python preprocess_exchange.py
   ```

2. Restore the saved model and make prediction:
   
   ```bash
   python train.py --dataset='Sanyo' --model-name='base_model_Sanyo' --restore-file='best'
   python train.py --dataset='Hanergy' --model-name='base_model_Hanergy' --restore-file='best'
   python train.py --dataset='Solar' --model-name='base_model_Solar' --restore-file='best'
   python train.py --dataset='elect' --model-name='base_model_elect' --restore-file='best'
   python train.py --dataset='exchange' --model-name='base_model_exchange' --restore-file='best'
   ```

3. Train the model:
  
   ```bash
   python train.py --dataset='Sanyo' --model-name='base_model_Sanyo' 
   python train.py --dataset='Hanergy' --model-name='base_model_Hanergy'
   python train.py --dataset='Solar' --model-name='base_model_Solar' 
   python train.py --dataset='elect' --model-name='base_model_elect' 
   python train.py --dataset='exchange' --model-name='base_model_exchange' 
   ```
