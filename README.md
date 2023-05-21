## Overview

This is the Tensorflow2 implementation of paper "Enhancing Privacy and Scalability in Deep Learning-based CSI feedback Autoencoders: A  Novel Approach to Separate Training". 
## Requirements

Our experiment packages version are below:
- Python >= 3.7
- numpy == 1.19.1
- tensorflow >= 2.2.0
- pandas == 1.1.3
- h5py == 2.10.0
- tensorboard == 2.2.1
- scipy == 1.4.1
- Keras == 2.3.1

## Project Preparation

#### A. Data Preparation

The channel state information (CSI) Eigenvectors was generated based on the CDL channel at  UMa scenario, as per the Scenarios and Requirements for AI-enhanced CSI from 3GPP Release 16 discussion. You can acquire our experiment datasets on [Google Drive](https://drive.google.com/drive/folders/1_lAMLk_5k1Z8zJQlTr5NRnSD6ACaNRtj?usp=sharing) and [Baidu Netdisk](https://pan.baidu.com/s/1Ggr6gnsXNwzD4ULbwqCmjA).

Some settings are below:
| **Parameter**                         | **Value**                                                                                          |
| ------------------------------------- | -------------------------------------------------------------------------------------------------- |
| Duplex, Waveform                      | FDD, OFDM                                                                                          |
| Scenario                              | Dense Urban (Macro only), UMi, InH                                                                 |
| Frequency Range                       | FR1 only                                                                                           |
| Inter-BS distance                     | 200m                                                                                               |
| Channel model                         | According to TR 38.901                                                                             |
| Antenna setup and port layouts at gNB | 32 ports: (8,8,2,1,1,2,8), (dH,dV) = (0.5, 0.8)λ 16 ports: (8,4,2,1,1,2,4), (dH,dV) = (0.5, 0.8)λ
| Antenna setup and port layouts at UE  | 4RX: (1,2,2,1,1,1,2), (dH,dV) = (0.5, 0.5)λ for (rank 1-4)                                         |
| BS Tx power                           | 41 dBm                                                                                             |
| BS antenna height                     | 25m                                                                                                |
| UE antenna height & gain              | Follow TR36.873                                                                                    |
| UE receiver noise figure              | 9dB                                                                                                |
| Modulation                            | Up to 256QAM                                                                                       |
| Coding on PDSCH                       | LDPC
| Max code-block                        |size=8448bit  
|Slot/non-slot                          |14 OFDM symbol slot
|SCS                                    |15kHz for 2GHz
|Simulation bandwidth                   |10MHz
|Frame structure                        |Slot Format 0 (all downlink) for all slots
|CSI feedback periodicity               |5 ms
|Scheduling delay                       |4 ms
| Traffic model                         | Full buffer                                                                                        |
| UE distribution                       | 80% indoor (3km/h), 20% outdoor (30km/h)                                                           |
| UE receiver                           | MMSE-IRC                                                                                           |
| Feedback assumption                   | Ideal                                                                                              |
| Channel estimation                    | Realistic

#### B. System illustrations
![Separate training decoder for 3×1 system.png](https://github.com/xls318027/CSI-Separate-training/blob/a1eb15aa653c8280666afeaad82e33d29ade33cc/Separate%20training%20decoder%20for%203%C3%971%20system.png)
<center><b>Separate training decoder for 3×1 system<b></center>
![Separate training encoder for 1×3 system.png](https://github.com/xls318027/CSI-Separate-training/blob/ca494729b3de4e39d64bbd0553b1e86897a30172/Separate%20training%20encoder%20for%201%C3%973%20system.png)
Separate training encoder for 1×3 system
#### C. Code files lists and explanation



## Train TransNet from Scratch


## Results and Reproduction




## Acknowledgment



Thanks  the Github project members for the open source [Transformer tutorial](https://github.com/datawhalechina/Learn-NLP-with-Transformers), our base model for TransNet is based on their work.
