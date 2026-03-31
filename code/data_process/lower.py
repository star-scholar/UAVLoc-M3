"""将图像扩展名JPG改为jpg"""

import os
import glob

def normalize_image_extensions(directory, target_extension=".jpg"):
    """
    将目录中所有图像文件的扩展名统一为小写
    """
    # 支持的图像格式模式
    patterns = ["*.JPG",]
    
    for pattern in patterns:
        search_pattern = os.path.join(directory, pattern)
        
        for file_path in glob.glob(search_pattern):
            # 获取文件名和目录
            dir_name, filename = os.path.split(file_path)
            name, ext = os.path.splitext(filename)
            
            # 新的小写扩展名路径
            new_path = os.path.join(dir_name, f"{name}{target_extension}")
            
            # 重命名文件
            if file_path != new_path:
                try:
                    os.rename(file_path, new_path)
                    print(f"重命名: {filename} -> {name}{target_extension}")
                except OSError as e:
                    print(f"重命名失败 {filename}: {e}")

# 使用示例
normalize_image_extensions("D:\\project\\dataset\\UAV_VisLoc\\drone")