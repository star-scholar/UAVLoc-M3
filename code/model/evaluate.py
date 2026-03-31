#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import os
import sys
import csv

true_file_path = "/home/lyt/UAVLoc-M3_dataset/AnYangRiver/pxGT_dem.xlsx"
pred_file_path = "/home/lyt/UAVLoc-M3/result/test/result/drone_matching_results.xlsx"
output_file_path = "/home/lyt/UAVLoc-M3/result/test/result/eval_distances.csv"
meters_per_pixel = 0.975 # Bing地图17级图像使用0.975，18级图像使用0.487
distance_thresholds_m = [100,200,300]

# ---------- 帮助函数 ----------
def file_header_bytes(path, n=16):
    with open(path, 'rb') as f:
        return f.read(n)

def detect_text_encoding(path, samplesize=4096):
    # 尝试 utf-8, gbk, latin1
    encs = ['utf-8', 'gbk', 'latin1']
    with open(path, 'rb') as f:
        sample = f.read(samplesize)
    for e in encs:
        try:
            sample.decode(e)
            return e
        except Exception:
            continue
    return None

def detect_csv_delimiter(path, encoding):
    try:
        with open(path, 'r', encoding=encoding, errors='replace') as f:
            sample = f.read(4096)
        dialect = csv.Sniffer().sniff(sample, delimiters=[',','\t',';','|'])
        return dialect.delimiter
    except Exception:
        # 回退策略
        if ',' in sample:
            return ','
        if '\t' in sample:
            return '\t'
        if ';' in sample:
            return ';'
        return ','

def find_column(df, candidates):
    cols = list(df.columns)
    lower_map = {c.lower().strip(): c for c in cols}
    for cand in candidates:
        key = cand.lower().strip()
        if key in lower_map:
            return lower_map[key]
    # 规范化匹配（去掉非字母数字）
    def norm(s):
        return ''.join(ch.lower() for ch in str(s) if ch.isalnum())
    norm_map = {norm(c): c for c in cols}
    for cand in candidates:
        nc = norm(cand)
        if nc in norm_map:
            return norm_map[nc]
    # 部分包含匹配
    for cand in candidates:
        parts = [p for p in cand.lower().split() if p]
        for c in cols:
            lc = c.lower()
            if all(p in lc for p in parts):
                return c
    return None

# ---------- 读取函数 ----------
def load_file(path):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    head = file_header_bytes(path, n=16)
    # xlsx 是 zip -> 以 PK 开头
    if head.startswith(b'PK'):
        try:
            return pd.read_excel(path, engine='openpyxl')
        except Exception as e:
            raise RuntimeError(f"尝试以 openpyxl 读取 .xlsx 失败: {e}")
    # 老式 Excel .xls OLE header: D0 CF 11 E0 A1 B1 1A E1
    if head.startswith(b'\xD0\xCF\x11\xE0'):
        try:
            return pd.read_excel(path, engine='xlrd')
        except Exception as e:
            raise RuntimeError(f"尝试以 xlrd 读取 .xls 失败: {e}（如提示缺少 xlrd，请执行 pip install xlrd）")
    # 否则尝试作为文本/CSV
    enc = detect_text_encoding(path)
    if enc is None:
        # 最后手段：尝试多种编码读取 CSV
        encs = ['utf-8', 'gbk', 'latin1']
    else:
        encs = [enc, 'utf-8', 'gbk', 'latin1']
    last_error = None
    for e in encs:
        try:
            delim = detect_csv_delimiter(path, e)
            df = pd.read_csv(path, encoding=e, sep=delim, engine='python')
            return df
        except Exception as ex:
            last_error = ex
            continue
    # 尝试 pandas 默认的 read_excel（让 pandas 自动选择 engine）
    try:
        return pd.read_excel(path)
    except Exception as e:
        raise RuntimeError(f"无法识别或读取文件（不是标准 .xlsx/.xls/CSV），最后尝试失败: {last_error}; pandas.read_excel 错误: {e}")

# ---------- 主流程 ----------
def main(true_path, pred_path):
    for p in (true_path, pred_path):
        if not os.path.exists(p):
            print(f"错误：文件不存在 - {p}")
            return 1

    try:
        true_df = load_file(true_path)
    except Exception as e:
        print(f"读取文件失败: {true_path}, 错误信息: {e}")
        print("\n提示：如果文件确实是 Excel，请用 Excel 打开并另存为新的 .xlsx；")
        print("如果是 CSV，请把扩展名改为 .csv 或确保编码为 UTF-8/GBK。")
        return 1

    try:
        pred_df = load_file(pred_path)
    except Exception as e:
        print(f"读取文件失败: {pred_path}, 错误信息: {e}")
        return 1

    print("成功读取文件。")
    print("真实值列：", list(true_df.columns))
    print("预测值列：", list(pred_df.columns))

    # 尝试识别列名
    true_x_col = find_column(true_df, ['pixel_x', 'pixel x', 'x', 'true_x', 'gt_x', 'pixelx'])
    true_y_col = find_column(true_df, ['pixel_y', 'pixel y', 'y', 'true_y', 'gt_y', 'pixely'])
    pred_x_col = find_column(pred_df, ['trans_x', 'Best Position x', 'Best_Position_x', 'bestpositionx', 'best_x', 'best x', 'Best Position X'])
    pred_y_col = find_column(pred_df, ['trans_y', 'Best Position y', 'Best_Position_y', 'bestpositiony', 'best_y', 'best y', 'Best Position Y'])

    if true_x_col is None or true_y_col is None:
        print("找不到真实值列。请检查真实值表的列名（可接受例：pixel_x, pixel_y）。")
        return 1
    if pred_x_col is None or pred_y_col is None:
        print("找不到预测值列。请检查预测值表的列名（可接受例：Best Position_x, Best Position_y）。")
        return 1

    # 转换为数值并合并（按行顺序）
    t_x = pd.to_numeric(true_df[true_x_col], errors='coerce')
    t_y = pd.to_numeric(true_df[true_y_col], errors='coerce')
    p_x = pd.to_numeric(pred_df[pred_x_col], errors='coerce')
    p_y = pd.to_numeric(pred_df[pred_y_col], errors='coerce')

    combined = pd.DataFrame({
        'true_x': t_x,
        'true_y': t_y,
        'pred_x': p_x,
        'pred_y': p_y
    })

    # 如果行数不一致，截取最小长度并告知
    # 也先尝试按索引对齐（如果索引和长度相等，则无需截断）
    if len(t_x) != len(p_x):
        min_len = min(len(t_x), len(p_x))
        print(f"注意：真实值行数 = {len(t_x)}，预测值行数 = {len(p_x)}。将按行号对齐并截取前 {min_len} 行进行评估。")
        combined = combined.iloc[:min_len]

    # 丢弃任何含 NaN 的行
    before = len(combined)
    combined = combined.dropna()
    after = len(combined)
    if after == 0:
        print("所有用于计算的坐标被解析为非数值（NaN）。请检查列的数据格式。")
        return 1
    if after < before:
        print(f"注意：丢弃了 {before - after} 行（含非数值项）。剩余 {after} 行用于计算。")

    # 计算距离（像素）
    distances = np.sqrt((combined['true_x'] - combined['pred_x'])**2 + (combined['true_y'] - combined['pred_y'])**2)

    # 指标
    mse_pixels = np.mean(distances**2)
    mse_meters = mse_pixels * (meters_per_pixel**2)
    # rmse
    rmse_pixels = np.sqrt(mse_pixels)
    rmse_meters = rmse_pixels * meters_per_pixel

    mae_pixels = np.mean(distances)
    mae_meters = mae_pixels * meters_per_pixel

    
    thresholds_pixels = [d / meters_per_pixel for d in distance_thresholds_m]

    total_samples = len(distances)
    precision_results = {}
    for t_pixel, t_meter in zip(thresholds_pixels, distance_thresholds_m):
        within_threshold = np.sum(distances <= t_pixel)
        precision_results[t_meter] = (within_threshold / total_samples) * 100

    # 输出
    print(f"\n样本数量: {total_samples}")
    print(f"平均绝对误差（MAE）：{mae_meters:.2f} 米")
    print(f"均方误差（MSE）：{mse_meters:.2f} 米²")
    print(f"均方根误差（RMSE）：{rmse_meters:.2f} 米")
    print("\n定位精度：")
    for t_meter, p in precision_results.items():
        print(f"误差小于 {t_meter} 米 的定位精度：{p:.2f}%")

    # 可选：保存每条记录的像素误差到 csv
    out_df = combined.copy()
    out_df['distance_pixels'] = distances.values
    out_df.to_csv(output_file_path, index=False)

    return 0

if __name__ == "__main__":
    sys.exit(main(true_file_path, pred_file_path))
