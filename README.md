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

<center><b>Separate training decoder for 3×1 system</b></center>
  
![Separate training encoder for 1×3 system.png](https://github.com/xls318027/CSI-Separate-training/blob/ca494729b3de4e39d64bbd0553b1e86897a30172/Separate%20training%20encoder%20for%201%C3%973%20system.png)
<center><b>Separate training encoder for 1×3 system<b></center>

## Results and Reproduction

| $Feedback\,\,bits$                                            | \multicolumn{3}{c}{49}    | \multicolumn{3}{c}{87}  | \multicolumn{3}{c}{130}   | \multicolumn{3}{c}{242} |
|---------------------------------------------------------------|---------------------------|-------------------------|---------------------------|-------------------------|
| \multirow{2}{*}{$Model\ pairs$}                               | \multicolumn{2}{c}{FLOPs} | \multirow{2}{*}{$SGCS$} | \multicolumn{2}{c}{FLOPs} | \multirow{2}{*}{$SGCS$} | \multicolumn{2}{c}{FLOPs} | \multirow{2}{*}{$SGCS$} | \multicolumn{2}{c}{FLOPs} | \multirow{2}{*}{$SGCS$} |
|                                                               | Encoder                   | Decoder                 |                           | Encoder                 | Decoder                   |                         | Encoder                   | Decoder                 |                 | Encoder                 | Decoder                 |                 |
| $E_1+D_1$                                                     | \multirow{3}{*}{21.37M}   | \multirow{3}{*}{21.37M} | 0.6387                    | \multirow{3}{*}{21.40M} | \multirow{3}{*}{21.40M}   | 0.6762                  | \multirow{3}{*}{21.43M}   | \multirow{3}{*}{21.43M} | 0.6995          | \multirow{3}{*}{21.52M} | \multirow{3}{*}{21.52M} | 0.7719          |
| $E_{\boldsymbol{r}}+\boldsymbol{A}_{\mathbf{1}}+D_1$          |                           |                         | \textbf{0.6341}           |                         |                           | \textbf{0.6687}         |                           |                         | \textbf{0.6896} |                         |                         | \textbf{0.7370} |
| $                                                             |                           |                         | $0.2954$                  |                         |                           | $0.3495$                |                           |                         | $0.4328$        |                         |                         | $0.4806$        |
| $E_2+D_2$                                                     | \multirow{3}{*}{21.37M}   | \multirow{3}{*}{17.83M} | 0.6292                    | \multirow{3}{*}{21.40M} | \multirow{3}{*}{17.86M}   | 0.6699                  | \multirow{3}{*}{21.43M}   | \multirow{3}{*}{17.89M} | 0.6973          | \multirow{3}{*}{21.52M} | \multirow{3}{*}{17.98M} | 0.7712          |
| $E_{\boldsymbol{r}}+\boldsymbol{A}_{\mathbf{2}}+D_2$          |                           |                         | \textbf{0.6222}           |                         |                           | \textbf{0.6605}         |                           |                         | \textbf{0.6839} |                         |                         | \textbf{0.7200} |
| $E\left( without\,\,\boldsymbol{A}_{\mathbf{2}} \right) +D_2$ |                           |                         | $0.2784$                  |                         |                           | $0.3360$                |                           |                         | $0.3719$        |                         |                         | $0.4652$        |
| $E_3+D_3$                                                     | \multirow{3}{*}{21.37M}   | \multirow{3}{*}{2.71M}  | 0.6194                    | \multirow{3}{*}{21.40M} | \multirow{3}{*}{2.74M}    | 0.6453                  | \multirow{3}{*}{21.43M}   | \multirow{3}{*}{2.77M}  | 0.6707          | \multirow{3}{*}{21.52M} | \multirow{3}{*}{2.86M}  | 0.7072          |
| $E_{\boldsymbol{r}}+\boldsymbol{A}_{\mathbf{3}}+D_3$          |                           |                         | \textbf{0.6025}           |                         |                           | \textbf{0.6284}         |                           |                         | \textbf{0.6512} |                         |                         | \textbf{0.6759} |
| $E\left( without\,\,\boldsymbol{A}_{\mathbf{3}} \right) +D_3$ |                           |                         | $0.2276$                  |                         |                           | $0.2709$                |                           |                         | $0.2917$        |                         |                         | $0.3601$        |
| %12                                                           | 0.9357                    | 0.9389                  | 0.8751                    | 0.8791                  | 0.8190                    | 0.9000                  | 0.7519                    | 0.8273                  |



## Acknowledgment



This work was supported in part by the National Natural Science Foundation
of China under Grant (No.92067202), Grant (No.62071058) and CICT Mobile Communication Technology Co., Ltd. 
