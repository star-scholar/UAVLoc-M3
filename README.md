<h1 align="center"> UAV-VisLoc: A Large-scale Dataset for UAV Visual Localization </h1>

## 0. Table of Contents

* [Introduction](#1-introduction)

* [About Dataset](#2-about-dataset)

  * [Overall Dataset](#21-overall-dataset) 

  * [Dataset Example](#22-dataset-example)

* [Contributions](#3-contributions)

* [Citation](#4-citation)

## 1. Introduction

We propose an unsupervised UAV localization method structured as a coarse-to-ffne framework. Our method is based on feature point extraction and matching, and we employ a structural-mask weighting strategy to enhance geometric consistency and reliability of match-ing. The framework initially partitions large-scale satellite imagery into numerous patches, identifying the optimal match through a search process. Then, precise lo-calization is achieved through a coordinate reffnement method based on homogra-phy matrix estimation. Furthermore, we constructed UAVLoc-M3, a dataset tai-lored to the requirements of our task, characterized by multi-scale, multi-scene, and multi-temporal variability. Comprising UAV imagery and large-scale satellite maps, this dataset covers three distinct regions encompassing diverse geographic scenarios under varying illumination conditions. 

| ![fig1.png](https://github.com/IntelliSensing/UAV-VisLoc/blob/main/img/fig1.png) | 
|:--:| 
| *Fig. 1 Schematic of UAV visual localization task.* |

## 2. About Dataset

### 2.1 Overall Dataset

You can download our overall dataset (932MB) on [ScienceDB](https://www.scidb.cn/s/aUjYFb).
DOI:10.57760/sciencedb.29772
