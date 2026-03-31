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

| ![fig1.png](https://github.com/star-scholar/UAVLoc-M3/blob/main/img/figure1.png) | 
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


| ![fig2.png](https://github.com/star-scholar/UAVLoc-M3/blob/main/img/figure2.png) | 
|:--:| 
| *Fig. 2 Sample Images of the UAVLoc-M3 Dataset. For each of the three distinct regions, the satellite image is shown on the left, and example UAV images of different scenes within that region are shown on the right. The corresponding location of each UAV image in the satellite image is indicated by a red circle.* |



| ![table2.png](https://github.com/star-scholar/UAVLoc-M3/blob/main/img/table1.png)| 
|:--:| 
| *TABLE I: Data Composition of the UAVLoc-M3 Dataset.* |

More detailed file structure:

```
├── UAVLoc-M3_dataset/
│   ├── AnYangRiver/
|       ├── dem/
|           ├── N36E113.hgt
|           ├── N36E114.hgt
|           ...
│       ├── drone/                    /* Drone Images
│           ├── DSC07012.jpg
│           ├── DSC07013.jpg
│           ├── DSC07014.jpg
|           ...
│       ├── satellite              	   /* Satellite Maps
│           ├── AnYangRiver.jpg
│       ├── AnYangRiver.xlsx			   		   /* format as: id lon lat GPS_Alt roll pitch yaw
│       ├── location_with_dem.xlsx			   		   /* format as: id target_lon target_lat relative_altitude
│       ├── pxGT_dem.xlsx			   		   /* format as: id pixel_x pixel_y
│   ├── CuiJiaQiao/
|       ├── dem/
|           ├── N36E114.hgt
|           ├── N36E115.hgt
|           ...
│       ├── drone/                    /* Drone Images
│           ├── DSC00472.jpg
│           ├── DSC00473.jpg
│           ├── DSC00474.jpg
|           ...
│       ├── satellite              	   /* Satellite Maps
│           ├── CuiJiaQiao.jpg
│       ├── CuiJiaQiao.xlsx			   		   /* format as: id lon lat GPS_Alt roll pitch yaw
│       ├── location_with_dem.xlsx			   		   /* format as: id target_lon target_lat relative_altitude
│       ├── pxGT_dem.xlsx			   		   /* format as: id pixel_x pixel_y
│   ├── Chongmingdao/
|       ├── dem/
|           ├── N31E121.hgt
|           ├── N31E122.hgt
|           ...
│       ├── drone/                    /* Drone Images
│           ├── 1497~1516,3459~3508,4986~5191
|               ├── DSC01497.jpg
|               ├── DSC01498.jpg
|               ├── DSC01499.jpg
|               ...
│           ├── 1517~1652,3310~3458
│           ├── 1653~1867,3094~3309
|           ...
│       ├── satellite              	   /* Satellite Maps
│           ├── 1497~1517,3459~3515,4986~5191.jpg
│           ├── 1517~1653,3310~3459.jpg
│           ├── 1653~1868,3094~3310.jpg
│       ├── Chongmingdao.xlsx			   		   /* format as: id lon lat GPS_Alt roll pitch yaw
│       ├── location_with_dem.xlsx			   		   /* format as: id target_lon target_lat relative_altitude
│       ├── pxGT_dem.xlsx			   		   /* format as: id pixel_x pixel_y

