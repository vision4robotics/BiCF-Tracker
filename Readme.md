# BiCF: Learning Bidirectional Incongruity-Aware Correlation Filter for Efficient UAV Object Tracking

<div align="center">
    <img src="https://raw.githubusercontent.com/vision4robotics/BiCF-Tracker/master/results/main_fig.png" alt="main_fig">
</div>

Matlab implementation of our Bidirectional Incongruity-Aware Correlation Filters (BiCF) tracker. 

| **Test passed**                                              |
| ------------------------------------------------------------ |
| [![matlab-2017](https://img.shields.io/badge/matlab-2017-yellow.svg)](https://www.mathworks.com/products/matlab.html) [![matlab-2018](https://img.shields.io/badge/matlab-2018-yellow.svg)](https://www.mathworks.com/products/matlab.html) [![matlab-2019](https://img.shields.io/badge/matlab-2019-yellow.svg)](https://www.mathworks.com/products/matlab.html) |

## Abstract

> For more info, please refer to our [paper](https://vision4robotics.github.io/publication/2020_icra_bicf-tracker/BiCF.pdf) and [video](https://youtu.be/fS12kosv37s).

　　Correlation filters (CFs) have shown excellent performance in unmanned aerial vehicle (UAV) tracking scenarios due to their high computational efficiency. During the UAV tracking process, viewpoint variations are usually accompanied by changes in the object and background appearance, which poses a unique challenge to CF-based trackers. Since the appearance is gradually changing over time, an ideal tracker can not only forward predict the object position but also backtrack to locate its position in the previous frame. There exist response-based errors in the reversibility of the tracking process containing the information on the changes in appearance. However, some existing methods do not consider the forward and backward errors based on while using only the current training sample to learn the filter. For other ones, the applicants of considerable historical training samples impose a computational burden on the UAV. In this work, a novel bidirectional incongruity-aware correlation filter (BiCF) is proposed. By integrating the response-based bidirectional incongruity error into the CF, BiCF can efficiently learn the changes in appearance and suppress the inconsistent error. Extensive experiments on 243 challenging sequences from three UAV datasets (UAV123, UAVDT, and DTB70) are conducted to demonstrate that BiCF favorably outperforms other 25 state-of-the-art trackers and achieves a real-time speed of 45.4 FPS on a single CPU, which can be applied in UAV efficiently.

## Quantitative results

<details open>
  <summary><b>UAV123@10fps</b></summary>
<div align="center">
    <img src="https://raw.githubusercontent.com/vision4robotics/BiCF-Tracker/master/results/UAV123_error_OPE.png" alt="UAV123_error">
    <img src="https://raw.githubusercontent.com/vision4robotics/BiCF-Tracker/master/results/UAV123_overlap_OPE.png" alt="UAV123_overlap">
</div>
</details>

<details>
  <summary><b>DTB70</b></summary>
<div align="center">
    <img src="https://raw.githubusercontent.com/vision4robotics/BiCF-Tracker/master/results/DTB70_error_OPE.png" alt="DTB70_error">
    <img src="https://raw.githubusercontent.com/vision4robotics/BiCF-Tracker/master/results/DTB70_overlap_OPE.png" alt="DTB70_overlap">
</div>

<details>
  <summary><b>UAVDT</b></summary>
<div align="center">
    <img src="https://raw.githubusercontent.com/vision4robotics/BiCF-Tracker/master/results/UAVDT_error_OPE.png" alt="UAVDT_error">
    <img src="https://raw.githubusercontent.com/vision4robotics/BiCF-Tracker/master/results/UAVDT_overlap_OPE.png" alt="UAVDT_overlap">
</div>

## Getting started

Run `demo_BiCF.m` script to test the tracker.

## Acknowledgements

The feature extraction modules and some of the parameters are borrowed from the ECO tracker (https://github.com/martin-danelljan/ECO).

