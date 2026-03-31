"""将tif图像转换为jpg图像"""

from PIL import Image

# 设置一个合适的值
Image.MAX_IMAGE_PIXELS = 10000000000  # 100亿像素

def tiff_to_jpg(tiff_path, jpg_path):
    try:
        # 打开.tif图像
        with Image.open(tiff_path) as img:
            # 保存为.jpg格式
            img.convert('RGB').save(jpg_path, "JPEG")
        print(f"转换成功: {tiff_path} -> {jpg_path}")
    except Exception as e:
        print(f"转换失败: {e}")

# 输入.tif文件路径和输出.jpg文件路径
tiff_file = r"D:/project/UAV_VisLoc_dataset/01/satellite01.tif"
jpg_file = r"D:/project/UAV_VisLoc_dataset/01/satellite01.jpg"

# 调用函数进行转换
tiff_to_jpg(tiff_file, jpg_file)

