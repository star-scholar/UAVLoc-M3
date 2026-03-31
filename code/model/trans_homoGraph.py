import os
import cv2
import torch
import numpy as np
import pandas as pd
from models.matching import Matching
from models.utils import frame2tensor
from math import ceil
from itertools import product

DRONE_SIZE = 320
TILE_SIZE = 600
RANSAC_threshold = 3.0
DRONE_PATH = "/home/lyt/UAVLoc-M3/result/test/drone/"
RESULT_EXCEL_PATH = "/home/lyt/UAVLoc-M3/result/test/result/drone_matching_results.xlsx"
TILE_PATH = "/home/lyt/UAVLoc-M3/result/test/result/"


def compute_homography_opencv(src_points, dst_points, method=cv2.RANSAC, ransac_threshold=5.0):
    """
    使用两组坐标点计算单应性矩阵
    """
    # 将坐标数据类型转换为float型
    src_pts = np.array(src_points, dtype=np.float32)
    dst_pts = np.array(dst_points, dtype=np.float32)
    
    # 计算单应性矩阵
    if method in [cv2.RANSAC, cv2.LMEDS]:
        H, mask = cv2.findHomography(
            srcPoints=src_pts,
            dstPoints=dst_pts,
            method=method,
            ransacReprojThreshold=ransac_threshold,
            maxIters=2000,
            confidence=0.995
        )
        inliers = mask.ravel().tolist()
        #print(f": {np.sum(mask)} / {len(src_points)}")
        return H, inliers
    else:
        H, status = cv2.findHomography(src_pts, dst_pts, method=0)
        return H, list(range(len(src_points)))

def transform_point_by_homography(point, H):
    """
    使用单应性矩阵将一个点从源坐标系映射到目标坐标系
    参数:
        point: 源点坐标，可以是 (x, y) 或 (x, y, 1)
        H: 3x3 单应性矩阵
    返回:
        transformed_point: 映射后的点 (x', y')
    """
    # 确保点是齐次坐标 (x, y, 1)
    if len(point) == 2:
        homogeneous_point = np.array([point[0], point[1], 1.0])
    else:
        homogeneous_point = np.array(point)
    
    # 应用单应性变换
    transformed_homogeneous = H @ homogeneous_point
    
    # 齐次坐标归一化
    if transformed_homogeneous[2] != 0:
        transformed_point = transformed_homogeneous[:2] / transformed_homogeneous[2]
    else:
        # 处理无穷远点的情况
        transformed_point = transformed_homogeneous[:2]
    
    return transformed_point

def transform_central_point(size_drone, point_position, match_excel_path):
    """
    用单应性矩阵方法纠正点的坐标
    """
    # 读入相匹配的若干组点对
    points = pd.read_excel(match_excel_path)
    uav_points = points[['image1_x', 'image1_y']].to_numpy()
    sat_points = points[['image2_x', 'image2_y']].to_numpy()
    # 计算单应性矩阵H
    H, inliers = compute_homography_opencv(uav_points, sat_points, cv2.RANSAC, RANSAC_threshold)
    # 计算变换后的点
    central_point = np.array((size_drone/2, size_drone/2))
    fact_point = transform_point_by_homography(central_point, H)
    
    return point_position+fact_point

def point_img(point, input_path, output_path, output_name):
    """
    在一张图片上绘制一个点并输出标注后的图片
    """
    img = cv2.imread(input_path)
    img = cv2.circle(img, tuple(map(int, point)), 3, (255,0,0), -1)
    cv2.imwrite(os.path.join(output_path, output_name), img)

def save_tile(satellite_img, center, size, save_path):
    """保存修正后的卫星切片范围图片"""
    tile = satellite_img[int(round(center[1]-size/2)):int(round(center[1]+size/2)), int(round(center[0]-size/2)):int(round(center[0]+size/2))].copy()
    cv2.imwrite(save_path, tile)
    print("已保存修正后的卫星切片范围图片")

# ------------------ main ------------------
if __name__ == "__main__":
    
    # 把匹配结果表格读入pandas数组
    results = pd.read_excel(RESULT_EXCEL_PATH)
    results['trans_x'] = 0
    results['trans_y'] = 0
    print(results)
    # 计算每行的变换结果
    for i in range(0, results.shape[0]):
        # 获取图片id、坐标
        drone_id = results.iloc[i, 0]
        pixel_x = results.iloc[i, 1]
        pixel_y = results.iloc[i, 2]
        # 变换坐标点
        match_excel_path = os.path.join(TILE_PATH, f"drone_{drone_id}/top1_x{pixel_x}_y{pixel_y}/matching_points.xlsx")
        point_position = np.array([pixel_x, pixel_y])
        trans_position = transform_central_point(DRONE_SIZE, point_position, match_excel_path)
        # 存入坐标点
        results.iloc[i, 3] = int(round(trans_position[0]))
        results.iloc[i, 4] = int(round(trans_position[1]))
        # 保存修正后的卫星切片范围图片
        #save_tile_path = os.path.join(TILE_PATH, f"drone_{drone_id}/top1_x{pixel_x}_y{pixel_y}/trans_tile.png")
        #save_tile(cv2.imread(SATELLITE_PATH), trans_position, TILE_SIZE, save_tile_path)
        # 保存中心点绘制点结果
        drone_path = os.path.join(DRONE_PATH, f"{drone_id}.jpg")
        tile_path = os.path.join(TILE_PATH, f"drone_{drone_id}/top1_x{pixel_x}_y{pixel_y}/orig_tile.png")
        point_path = os.path.join(TILE_PATH, f"drone_{drone_id}/top1_x{pixel_x}_y{pixel_y}/")
        point_img(np.array((DRONE_SIZE/2, DRONE_SIZE/2)), drone_path, point_path, "point_drone.png")
        point_img(np.array((trans_position[0]-point_position[0], trans_position[1]-point_position[1])), tile_path, point_path, "point_tile.png")
        
        
    # 把pandas数组写回表格
    print(results)
    results.to_excel(RESULT_EXCEL_PATH, sheet_name='Sheet1', index=False)    
    
    # 读入相匹配的若干组点对
    #points = pd.read_excel(input_file)
    #uav_points = points[['X_0', 'Y_0']].to_numpy()
    #sat_points = points[['X_1', 'Y_1']].to_numpy()

    # 计算单应性矩阵H
    #H, inliers = compute_homography_opencv(uav_points, sat_points, cv2.RANSAC, RANSAC_threshold)
    #print("homograph:\n", H)
    # 计算变换后的点
    #central_point = np.array((160.0, 160.0))
    #fact_point = transform_point_by_homography(central_point, H)
    #print(f"central_point: {central_point}")
    #print(f"fact_point: {fact_point}")
    
    
    
    
    
