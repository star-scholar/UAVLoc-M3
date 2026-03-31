<h1 align="center"> Unsupervised UAV Visual Localization: Dataset and Method via Local Feature Point </h1>

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

The UAVLoc-M3 dataset comprises a total of 4,946 high-resolution UAV-captured images, as shown in Figure 5, along with 13 large-scale satellite maps acquired from
9Bing Maps that comprehensively depict the UAV flight areas.

1) UAV Sampling: The experimental data were collected in real-world scenarios using UAV platforms equipped with Sony ILCE-7RM2 and Sony ILCE-6000 aerial survey cameras. Images were captured at intervals during flight missions across three regions in China: Chongming Island in Shanghai, the Anyang River Basin in Henan
Province, and Cuijiaqiao Town in Anyang County, Henan Province. The UAVs operated at relative flight altitudes ranging from 300 to 800 meters, capturing images
from various flight attitudes and perspectives. The flight paths traversed diverse terrains and landforms, including villages, towns, rivers, and farmlands. Data acquisition spanned both spring and summer seasons under clear and foggy weather conditions.

2) Satellite Image Sampling: We compared the quality of satellite maps from four different sources and ultimately selected Bing Maps satellite imagery for its superior performance in image clarity, acquisition timing, and seasonal consistency. The geographic extent (latitude and longitude range) of the satellite maps was delineated based on the UAV flight coverage. To ensure a consistent scale ratio between the satellite image resolution and the resolution of UAV images captured at different altitudes, we acquired Bing satellite map data at Zoom Levels 17 and 18 using the LocaSpace Viewer software platform. At Zoom Level 17, the ground resolution of the satellite images is approximately 0.965 meters per pixel; at Zoom Level 18, it is approximately 0.508 meters per pixel. At both resolution levels, features such as building outlines, main roads, and large vehicles can be clearly identified.

3) Dataset Composition: The sampled UAV and satellite maps were structured into the UAVLoc-M3 dataset. The detailed composition is presented in Table 1. The dataset records not only the attribute information of the UAV images—including image resolution, geographic coordinates (latitude and longitude), acquisition date, flight altitude, and orientation angles (pitch, yaw, and roll)—but also provides the geographic extent information (i.e., corresponding GPS longitude and latitude ranges) of the satellite maps.


| ![fig2.png](https://github.com/IntelliSensing/UAV-VisLoc/blob/main/img/fig2.png) | 
|:--:| 
| *Fig. 2 An example of drone images and satellite map. The red dots in the satellite map represent the center points of drone images. The satellite map encompasses various terrains such as cities, towns, farms, and rivers. We also show the drone images of these terrains.* |



| ![table2.png](https://github.com/IntelliSensing/UAV-VisLoc/blob/main/img/table2.png)| 
|:--:| 
| *TABLE I: IMAGE ATTRIBUTE COMPOSITION FOR THE UAV-VisLoc DATASET.* |


The dataset contents are as follows:

| Drone Images | Satellite Maps | Cites | Categories |
| ------------ | -------------- | ----- | ---------- |
| 6,742        | 11             | 11    | 7          |


More detailed file structure:

```
├── UAV-VisLoc/
│   ├── satellite_coordinates_range.csv   /* format as: filename latitude longitude
│   ├── 01/
│       ├── drone/                    /* Drone Images
│           ├── 01_0001.JPG
│           ├── 01_0002.JPG
│           ├── 01_0003.JPG
|           ...
│       ├── satellite01.tif              	   /* Satellite Maps
│       ├── 01.csv			   		   /* format as: filename latitude longitude height ···
│   ├── 02/
│       ├── drone/                     /* Drone Images
│           ├── 02_0001.JPG
│           ├── 02_0002.JPG
│           ├── 02_0003.JPG
|           ...
│       ├── satellite02.tif               		/* Satellite Maps
│       ├── 02.csv				        /* format as: filename latitude longitude height ···
│   ├── 03/
│       ├── drone/                      /* Drone Images
│           ├── 03_0001.JPG
│           ├── 03_0002.JPG
│           ├── 03_0003.JPG
|           ...
│       ├── satellite03.tif              	    /* Satellite Maps
│       ├── 03.csv						/* format as: filename latitude longitude height ···
```
