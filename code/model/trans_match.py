"""
SuperPoint + SuperGlue 图像匹配完整程序
用于提取两幅图像的特征点并进行匹配
保存匹配点对到excel表格
"""

import os
import torch
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import time
import warnings
warnings.filterwarnings("ignore")

ANGLES = [0, 90, 180, 270]  # 旋转候选
MAX_KEYPOINTS = 1024
KEYPOINT_THRESHOLD = 0.005
WEIGHTS = 'outdoor'
MATCH_THRESHOLD = 0.002
TOP_N = 25
RESULT_EXCEL_PATH = "/home/lyt/UAVLoc-M3/result/test/result/drone_matching_results.xlsx"
DRONE_PATH = "/home/lyt/UAVLoc-M3/result/test/drone/"
TILE_PATH = "/home/lyt/UAVLoc-M3/result/test/result/"


# 假设 superpoint.py 和 superglue.py 在同一目录下
try:
    from models.superpoint import SuperPoint
    from models.superglue import SuperGlue
    print("成功导入 SuperPoint 和 SuperGlue 模块")
except ImportError:
    print("警告: 无法直接导入 superpoint 和 superglue 模块")
    print("请确保 superpoint.py 和 superglue.py 在正确路径")
    # 备用导入方式
    import sys
    sys.path.append('.')
    from superpoint import SuperPoint
    from superglue import SuperGlue

class SuperPointFeatureExtractor:
    """
    SuperPoint 特征点提取器封装类
    用于提取图像特征点和描述符
    """
    def __init__(self, config=None):
        """
        初始化 SuperPoint 特征提取器
        
        参数:
            config: 配置字典，包含 SuperPoint 参数
        """
        # 默认配置
        default_config = {
            'descriptor_dim': 256,
            'nms_radius': 4,
            'keypoint_threshold': 0.005,
            'max_keypoints': 1024,
            'remove_borders': 4,
            'weights_path': 'models/weights/superpoint_v1.pth',  # 预训练权重路径
            'cuda': torch.cuda.is_available()
        }
        
        # 合并配置
        if config:
            default_config.update(config)
        self.config = default_config
        
        # 初始化模型
        print(f"初始化 SuperPoint 模型 (CUDA: {self.config['cuda']})...")
        self.model = SuperPoint(self.config)
        
        # 加载预训练权重
        if Path(self.config['weights_path']).exists():
            checkpoint = torch.load(self.config['weights_path'], 
                                   map_location='cpu')
            self.model.load_state_dict(checkpoint)
            print(f"加载预训练权重: {self.config['weights_path']}")
        else:
            print("警告: 未找到预训练权重，使用随机初始化的模型")
        
        # 设置为评估模式
        self.model.eval()
        
        # 移动到 GPU（如果可用）
        self.device = 'cuda' if self.config['cuda'] else 'cpu'
        self.model = self.model.to(self.device)
    
    def preprocess_image(self, image_path):
        """
        预处理图像
        
        参数:
            image_path: 图像文件路径
            
        返回:
            image_tensor: 预处理后的图像张量
            original_image: 原始图像（用于可视化）
            scale_factors: 缩放因子
        """
        # 读取图像
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"无法读取图像: {image_path}")
        
        # 保存原始图像用于可视化
        original_image = image.copy()
        
        # 转换为灰度图
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # 调整尺寸为8的倍数（SuperPoint要求）
        H, W = gray.shape
        H_new = (H // 8) * 8
        W_new = (W // 8) * 8
        
        if H_new != H or W_new != W:
            gray = cv2.resize(gray, (W_new, H_new), 
                             interpolation=cv2.INTER_AREA)
            print(f"调整图像尺寸: {H}x{W} -> {H_new}x{W_new}")
        
        # 转换为浮点型并归一化
        gray = gray.astype(np.float32) / 255.0
        
        # 转换为 PyTorch 张量
        gray_tensor = torch.from_numpy(gray).unsqueeze(0).unsqueeze(0)
        gray_tensor = gray_tensor.to(self.device)
        
        scale_factors = (H_new / H, W_new / W)
        
        return gray_tensor, original_image, scale_factors
    
    def extract_features(self, image_path):
        """
        提取图像特征点和描述符
        
        参数:
            image_path: 图像文件路径
            
        返回:
            pred: 包含关键点和描述符的字典
            original_image: 原始图像
        """
        print(f"提取特征点: {image_path}")
        
        # 预处理图像
        image_tensor, original_image, scale_factors = self.preprocess_image(image_path)
        
        # 提取特征
        with torch.no_grad():
            pred = self.model({'image': image_tensor})
        
        # 调整关键点坐标到原始图像尺寸
        if 'keypoints' in pred:
            keypoints = pred['keypoints'][0].cpu().numpy()
            # 根据缩放因子调整坐标
            keypoints[:, 0] = keypoints[:, 0] / scale_factors[1]
            keypoints[:, 1] = keypoints[:, 1] / scale_factors[0]
            pred['keypoints'] = [keypoints]
        
        return pred, original_image

class SuperGlueMatcher:
    """
    SuperGlue 匹配器封装类
    用于匹配两幅图像的特征点
    """
    def __init__(self, config=None):
        """
        初始化 SuperGlue 匹配器
        
        参数:
            config: 配置字典，包含 SuperGlue 参数
        """
        # 默认配置
        default_config = {
            'weights': 'outdoor',  # 'indoor' 或 'outdoor'
            'sinkhorn_iterations': 20,
            'match_threshold': 0.2,
            'cuda': torch.cuda.is_available()
        }
        
        # 合并配置
        if config:
            default_config.update(config)
        self.config = default_config
        
        # 初始化模型
        print(f"初始化 SuperGlue 模型 (权重: {self.config['weights']})...")
        self.model = SuperGlue(self.config)
        
        # 设置为评估模式
        self.model.eval()
        
        # 移动到 GPU（如果可用）
        self.device = 'cuda' if self.config['cuda'] else 'cpu'
        self.model = self.model.to(self.device)
    
    def prepare_match_data(self, pred1, pred2, image1_shape, image2_shape):
        """
        准备匹配数据
        """
        # 确保有关键点和描述符
        required_keys = ['keypoints', 'descriptors', 'scores']
        for key in required_keys:
            if key not in pred1 or key not in pred2:
                raise ValueError(f"缺少必要的键: {key}")
    
        # 转换函数：处理 Tensor 或 numpy 数组
        def convert_to_tensor(data, name=""):
            if isinstance(data, torch.Tensor):
                # 如果已经在正确的设备上，直接返回
                return data if data.device == self.device else data.to(self.device)
            elif isinstance(data, np.ndarray):
                return torch.from_numpy(data).to(self.device)
            else:
                raise TypeError(f"{name}: 期望 Tensor 或 ndarray，得到 {type(data)}")
    
        # 提取并转换数据
        # 注意：pred['keypoints'][0] 可能是 Tensor 或 numpy
        kpts0 = pred1['keypoints'][0]
        kpts1 = pred2['keypoints'][0]
        scores0 = pred1['scores'][0]
        scores1 = pred2['scores'][0]
    
        # 准备数据
        data = {
            'keypoints0': convert_to_tensor(kpts0, "keypoints0").unsqueeze(0),
            'keypoints1': convert_to_tensor(kpts1, "keypoints1").unsqueeze(0),
            'descriptors0': pred1['descriptors'][0].unsqueeze(0).to(self.device),
            'descriptors1': pred2['descriptors'][0].unsqueeze(0).to(self.device),
            'scores0': convert_to_tensor(scores0, "scores0").unsqueeze(0),
            'scores1': convert_to_tensor(scores1, "scores1").unsqueeze(0),
            'image0': torch.zeros(1, 1, image1_shape[0], image1_shape[1]).to(self.device),
            'image1': torch.zeros(1, 1, image2_shape[0], image2_shape[1]).to(self.device),
        }
    
        return data
    
    def match_features(self, data):
        """
        匹配特征点
        
        参数:
            data: 包含两幅图像特征的数据字典
            
        返回:
            matches: 匹配结果
            match_scores: 匹配分数
        """
        print("进行特征点匹配...")
        
        with torch.no_grad():
            pred = self.model(data)
        
        # 提取匹配结果
        matches = pred['matches0'][0].cpu().numpy()
        match_scores = pred['matching_scores0'][0].cpu().numpy()
        
        return matches, match_scores


def rotate_image(image, angle):
    """将图像按特定角度旋转"""
    if angle == 0:
        return image
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    return rotated

def save_matches_to_excel(keypoints1, keypoints2, matches, match_scores, 
                          output_file, match_threshold=0.2, top_n=None):
    """
    将匹配点对保存到Excel文件
    
    参数:
        keypoints1: 第一张图像的关键点坐标 (M, 2)
        keypoints2: 第二张图像的关键点坐标 (N, 2)
        matches: 匹配数组 (M,)
        match_scores: 匹配分数数组 (M,)
        output_file: 输出Excel文件路径
        match_threshold: 匹配分数阈值（当top_n为None时使用）
        top_n: 保存前n个最高分数的匹配点对，如果为None则使用阈值筛选
    """
    print(f"保存匹配点对到Excel: {output_file}")
    
    # 收集匹配点对
    matched_pairs = []
    
    for idx1, idx2 in enumerate(matches):
        if idx2 != -1:  # 有效匹配
            score = match_scores[idx1] if idx1 < len(match_scores) else 1.0
            
            if idx1 < len(keypoints1) and idx2 < len(keypoints2):
                kp1 = keypoints1[idx1]
                kp2 = keypoints2[idx2]
                
                # 添加到匹配对列表
                matched_pairs.append({
                    'image1_x': float(kp1[0]),
                    'image1_y': float(kp1[1]),
                    'image2_x': float(kp2[0]),
                    'image2_y': float(kp2[1]),
                    'match_score': float(score)
                })
    
    # 如果没有匹配点对
    if not matched_pairs:
        print("警告: 没有找到有效的匹配点对")
        return
    
    # 创建DataFrame
    df = pd.DataFrame(matched_pairs)
    
    # 根据条件筛选
    if top_n is not None:
        # 按匹配分数降序排列，取前n个
        df = df.sort_values(by='match_score', ascending=False).head(top_n)
        print(f"选取匹配分数最高的 {len(df)} 个点对")
    elif match_threshold is not None:
        # 使用阈值筛选
        df = df[df['match_score'] >= match_threshold]
        print(f"使用阈值 {match_threshold} 筛选，得到 {len(df)} 个点对")
    else:
        print(f"不进行筛选，得到 {len(df)} 个点对")
    
    # 如果筛选后DataFrame为空
    if df.empty:
        print("警告: 筛选后没有匹配点对")
        return
    
    # 保存到Excel
    try:
        df.to_excel(output_file, index=False)
        print(f"成功保存 {len(df)} 个匹配点对到 {output_file}")
        
        # 显示前几行数据
        print("\n前5个匹配点对:")
        print(df.head())
        
        # 显示分数统计
        print(f"\n匹配分数统计:")
        print(f"  最高分: {df['match_score'].max():.3f}")
        print(f"  最低分: {df['match_score'].min():.3f}")
        print(f"  平均分: {df['match_score'].mean():.3f}")
        
    except Exception as e:
        print(f"保存Excel文件时出错: {e}")
        
        # 如果Excel写入失败，尝试保存为CSV
        csv_file = output_file.with_suffix('.csv')
        try:
            df.to_csv(csv_file, index=False)
            print(f"已保存为CSV文件: {csv_file}")
        except Exception as e2:
            print(f"保存CSV文件时也出错: {e2}")





def process_one_image(drone_path, tile_path, match_excel_path):
    """处理单对图像"""
    # 检查输入图像是否存在
    for img_path in [drone_path, tile_path]:
        if not Path(img_path).exists():
            print(f"错误: 图像不存在: {img_path}")
            return
    drone_img = cv2.imread(drone_path)
    tile_img = cv2.imread(tile_path)
    
    # 存储每个旋转角度的匹配结果
    all_results = []
    best_score = -1
    best_angle = 0
    best_matches = None
    best_match_scores = None
    best_keypoints1_rot = None
    best_keypoints2 = None    
        
    # 1. 初始化 SuperPoint 和 SuperGlue 特征提取器
    sp_config = {
        'max_keypoints': MAX_KEYPOINTS,
        'keypoint_threshold': KEYPOINT_THRESHOLD,
    }   
    superpoint = SuperPointFeatureExtractor(sp_config)
    
    sg_config = {
        'weights': WEIGHTS,
        'match_threshold': MATCH_THRESHOLD,
    }    
    superglue = SuperGlueMatcher(sg_config)
    
    print("\n" + "="*50)
    print(f"开始处理图像对:")
    print(f"  Drone图像: {drone_path}")
    print(f"  Tile图像: {tile_path}")
    print("="*50)
    
    # 2. 对drone图像提取特征点（只需要一次）
    print("\n提取Drone图像特征点...")
    pred_drone, _ = superpoint.extract_features(drone_path)
    keypoints_drone = pred_drone['keypoints'][0]
    kp_drone_count = len(keypoints_drone)
    print(f"Drone图像检测到 {kp_drone_count} 个特征点")
    
    # 3. 对每个旋转角度进行匹配
    for angle in ANGLES:
        print(f"\n{'='*30}")
        print(f"尝试旋转角度: {angle}度")
        print(f"{'='*30}")
        
        # 旋转tile图像
        if angle == 0:
            tile_rotated = tile_img.copy()
            rotation_matrix = np.eye(2)
            translation = np.array([0, 0])
        else:
            # 计算旋转矩阵
            (h, w) = tile_img.shape[:2]
            center = (w // 2, h // 2)
            
            # 获取旋转矩阵
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            
            # 计算旋转后的图像边界
            cos = np.abs(M[0, 0])
            sin = np.abs(M[0, 1])
            new_w = int((h * sin) + (w * cos))
            new_h = int((h * cos) + (w * sin))
            
            # 调整旋转矩阵以考虑平移
            M[0, 2] += (new_w / 2) - center[0]
            M[1, 2] += (new_h / 2) - center[1]
            
            # 旋转图像
            tile_rotated = cv2.warpAffine(tile_img, M, (new_w, new_h))
        
        # 保存旋转后的图像用于特征提取
        temp_tile_path = os.path.join(TILE_PATH, f"temp_tile_rotated_{angle}.jpg")
        cv2.imwrite(temp_tile_path, tile_rotated)
        
        # 提取旋转后drone图像的特征点
        print(f"提取旋转后Tile图像特征点...")
        pred_tile, _ = superpoint.extract_features(temp_tile_path)
        keypoints_tile_rot = pred_tile['keypoints'][0]
        kp_tile_count = len(keypoints_tile_rot)
        print(f"旋转{angle}度后检测到 {kp_tile_count} 个特征点")
        
        # 准备匹配数据
        image_drone_shape = drone_img.shape[:2]
        image_tile_shape = tile_rotated.shape[:2]
        
        try:
            match_data = superglue.prepare_match_data(
                pred_drone, pred_tile, 
                image_drone_shape, image_tile_shape
            )
        except Exception as e:
            print(f"准备匹配数据时出错: {e}")
            os.remove(temp_tile_path)
            continue
        
        # 进行特征匹配
        matches, match_scores = superglue.match_features(match_data)
        
        # 统计匹配结果
        valid_matches = np.sum(matches != -1)
        print(f"找到 {valid_matches} 个有效匹配")
        
        if valid_matches > 0:
            # 计算平均匹配分数
            valid_mask = matches != -1
            valid_scores = match_scores[valid_mask]
            avg_score = np.mean(valid_scores) if len(valid_scores) > 0 else 0
            
            print(f"平均匹配分数: {avg_score:.4f}")
            
            # 记录结果
            all_results.append({
                'angle': angle,
                'valid_matches': valid_matches,
                'avg_score': avg_score,
                'keypoints_drone': keypoints_drone,
                'keypoints_tile': keypoints_tile_rot,
                'matches': matches,
                'match_scores': match_scores
            })
            
            # 更新最佳结果
            if avg_score > best_score:
                best_score = avg_score
                best_angle = angle
                best_matches = matches
                best_match_scores = match_scores
                best_keypoints1 = keypoints_drone
                best_keypoints2_rot  = keypoints_tile_rot
        else:
            print(f"该旋转角度下未找到有效匹配")
        
        # 删除临时文件
        os.remove(temp_tile_path)
    
    # 4. 检查是否找到匹配
    if best_score == -1:
        print(f"\n在所有旋转角度下均未找到有效匹配")
        return
    
    print(f"\n{'='*50}")
    print(f"最佳旋转角度: {best_angle}度")
    print(f"平均匹配分数: {best_score:.4f}")
    print(f"有效匹配数量: {np.sum(best_matches != -1)}")
    print(f"{'='*50}")
    
    # 5. 将匹配点旋转回原始tile图像坐标
    print(f"\n将匹配点旋转回原始坐标...")
    
    if best_angle == 0:
        # 不需要旋转
        keypoints_tile_img = best_keypoints2_rot.copy()
    else:
        # 获取原始图像尺寸和旋转参数
        (h, w) = tile_img.shape[:2]
        center = (w // 2, h // 2)
        
        # 计算旋转矩阵
        M = cv2.getRotationMatrix2D(center, best_angle, 1.0)
        
        # 计算旋转后的图像边界
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])
        new_w = int((h * sin) + (w * cos))
        new_h = int((h * cos) + (w * sin))
        
        # 调整旋转矩阵以考虑平移
        M[0, 2] += (new_w / 2) - center[0]
        M[1, 2] += (new_h / 2) - center[1]
        
        # 计算逆变换矩阵
        M_inv = cv2.invertAffineTransform(M)
        
        # 对匹配的tile关键点进行逆变换
        keypoints_tile_img = []
        for kp in best_keypoints2_rot:
            # 转换为齐次坐标
            kp_homogeneous = np.array([kp[0], kp[1], 1])
            # 应用逆变换
            kp_orig_homogeneous = M_inv @ kp_homogeneous
            keypoints_tile_img.append(kp_orig_homogeneous[:2])
        
        keypoints_tile_img = np.array(keypoints_tile_img)
  
#    # 输出特征点统计信息
#    kp1_count = len(pred1['keypoints'][0]) if 'keypoints' in pred1 else 0
#    kp2_count = len(pred2['keypoints'][0]) if 'keypoints' in pred2 else 0
#    print(f"图像1: 检测到 {kp1_count} 个特征点")
#    print(f"图像2: 检测到 {kp2_count} 个特征点")
    

    
#    # 4. 准备匹配数据
#    print("\n" + "="*50)
#    print("步骤 2: 准备匹配数据")
#    print("="*50)
    
#    image1_shape = image1.shape[:2]
#    image2_shape = image2.shape[:2]
    
#    match_data = superglue.prepare_match_data(pred1, pred2, image1_shape, image2_shape)
    
#    # 5. 进行特征匹配
#    print("\n" + "="*50)
#    print("步骤 3: 匹配特征点")
#    print("="*50)
    
#    matches, match_scores = superglue.match_features(match_data)
    
#    # 统计匹配结果
#    valid_matches = np.sum(matches != -1)
#    print(f"找到 {valid_matches} 个有效匹配")
    
#    # 6. 提取关键点坐标
#    keypoints1 = pred1['keypoints'][0]
#    keypoints2 = pred2['keypoints'][0]
    
    # 7. 保存匹配点对到Excel
    print("\n" + "="*50)
    print("步骤 4: 保存匹配点对到Excel")
    print("="*50)
    
    output_path = Path(match_excel_path)
    
    # 确保输出目录存在
    output_path.parent.mkdir(exist_ok=True, parents=True)
    
    # 调用保存函数
    save_matches_to_excel(
        keypoints1=best_keypoints1,
        keypoints2=keypoints_tile_img,
        matches=best_matches,
        match_scores=best_match_scores,
        output_file=output_path,
        match_threshold=MATCH_THRESHOLD,
        top_n = TOP_N
    )
    
       

if __name__ == "__main__":
    
    # 把匹配结果表格读入pandas数组
    results = pd.read_excel(RESULT_EXCEL_PATH)
    
    for i in range(0, results.shape[0]):
        # 获取图片id、坐标
        drone_id = results.iloc[i, 0]
        pixel_x = results.iloc[i, 1]
        pixel_y = results.iloc[i, 2]
        # 获取无人机图、切片图、结果表格路径
        drone_path = os.path.join(DRONE_PATH, f"{drone_id}.jpg")
        tile_path = os.path.join(TILE_PATH, f"drone_{drone_id}/top1_x{pixel_x}_y{pixel_y}/orig_tile.png")
        match_excel_path = os.path.join(TILE_PATH, f"drone_{drone_id}/top1_x{pixel_x}_y{pixel_y}/matching_points.xlsx")
        # 保存匹配点对坐标到表格
        process_one_image(drone_path, tile_path, match_excel_path)
        
    
    
    
    
    
    
    
    
    
    
    
    