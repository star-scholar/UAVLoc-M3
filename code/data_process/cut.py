"""将图像裁剪压缩为320×320的正方形，中心点不变"""

import os
import cv2

# 指定根目录路径
root_directory = "D:\\project\\dataset\\shanghai\\drone"
# 遍历文件夹
for dirpath, dirnames, filenames in os.walk(root_directory):
    # dirpath 是遍历到的当前文件夹路径
    # dirnames 是当前路径下所有子目录的列表
    # filenames 是当前路径下所有非目录文件的列表

    # 打印当前文件夹路径
    print("当前文件夹路径:", dirpath)

    # 遍历当前文件夹内的所有文件
    for filename in filenames:
        # 获取文件的完整路径
        file_path = os.path.join(dirpath, filename)
        # 打印文件名和完整路径
        print("文件名:", filename)
        print("文件完整路径:", file_path)

        # 读取图片
        img = cv2.imread(file_path)
        height, width, channels = img.shape

        # 定义裁剪区域的坐标和尺寸
        a = min(height,width) # 宽高中的较小值
        start_x = width/2-a/2  # 裁剪区域左上角的x坐标
        start_y = height/2-a/2  # 裁剪区域左上角的y坐标
        end_x = width/2+a/2
        end_y = height/2+a/2

        # 裁剪图片
        cropped_img = img[int(start_y):int(end_y), int(start_x):int(end_x)]
        # 压缩图片
        cropped_img = cv2.resize(cropped_img,(224,224))

        # 显示裁剪后的图片
        # cv2.imshow('Cropped Image', cropped_img)

        # 裁剪图片保存
        cv2.imwrite(file_path, cropped_img)

        # 等待按键后关闭窗口
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
