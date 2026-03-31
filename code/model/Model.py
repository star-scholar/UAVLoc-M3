import os
import cv2
import torch
import numpy as np
import pandas as pd
from models.matching import Matching
from models.utils import frame2tensor
from math import ceil
from itertools import product

# ------------------ 配置（可调） ------------------
TILE_SIZE = 600  # 根据飞行高度和卫星图精度确定，安阳河、崔家桥设置为 600，崇明岛设置为 450
STRIDE = 300  # 根据 TILE_SIZE 确定，建议设置为 TILE_SIZE 的一半
ANGLES = [0, 90, 180, 270]  # 旋转候选
SCALES = [1.0]  # 缩放候选
TOPK = 1  # 保存 top-K 的结果输出
MASK_BONUS = 1.5  # 若匹配点两边均在结构掩膜上，额外加权分数，建议设置为 1.5
CANNY_UAV = (100, 200)  # UAV Canny 阈值，建议设置为 (100, 200)
CANNY_SAT = (100, 200)  # SAT tile Canny 阈值，建议设置为 (100, 200)
DEVICE = 0 if torch.cuda.is_available() else 'cpu'

# SuperGlue超参数
ITERATION = 10 # 迭代次数
KEY = 0.001 # 关键点检测阈值，建议设置为 0.001

# 输入路径
drone_images_dir = "/home/lyt/UAVLoc-M3/result/test/drone/"
sat_path = "/home/lyt/UAVLoc-M3_dataset/AnYangRiver/satellite/AnYangRiver.jpg"
save_dir_base = "/home/lyt/UAVLoc-M3/result/test/result/"
# 若要保存切分后的卫星图块，则将以下代码取消注释
#output_dir = "/home/lyt/UAVLoc-M3/result/test/tiles/"

# ------------------ 工具函数 ------------------
def crop_tiles(img, tile_size=TILE_SIZE, stride=STRIDE):
    h, w = img.shape[:2]
    tiles, coords = [], []
    for y in range(0, h - tile_size + 1, stride):
        for x in range(0, w - tile_size + 1, stride):
            tiles.append(img[y:y + tile_size, x:x + tile_size].copy())
            coords.append((x, y))
    return tiles, coords


def rotate_image(image, angle):
    if angle == 0:
        return image
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    return rotated


def get_structure_mask(gray_img, canny_thresh):
    """返回 uint8 二值掩膜（0/255），并进行膨胀以增强连通性"""
    t1, t2 = canny_thresh
    edges = cv2.Canny(gray_img, t1, t2)
    # 膨胀让边缘更粗，便于关键点命中
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    edges = cv2.dilate(edges, kernel, iterations=1)
    return edges  # 0/255


def draw_match_vis(uav_gray, tile_gray, kpts0, kpts1, matches, inlier_mask=None, out_path=None):
    """
    画匹配连线：拼接 UAV | Tile，然后画匹配对（若提供 inlier_mask 则只画内点）
    kpts0: (N0,2), kpts1: (N1,2), matches: length N0 array with matched index or -1
    """
    # Convert to BGR for colored drawing
    uav_bgr = cv2.cvtColor(uav_gray, cv2.COLOR_GRAY2BGR)
    tile_bgr = cv2.cvtColor(tile_gray, cv2.COLOR_GRAY2BGR)
    h0, w0 = uav_gray.shape
    h1, w1 = tile_gray.shape
    # resize tile vertically to match UAV height for nicer visualization if heights differ
    if h0 != h1:
        scale = h0 / h1
        tile_bgr = cv2.resize(tile_bgr, (int(w1 * scale), h0))
        scale_tile_x = (lambda x: int(x * scale))
        scale_tile_y = (lambda y: int(y * scale))
    else:
        scale = 1.0
        scale_tile_x = (lambda x: int(x))
        scale_tile_y = (lambda y: int(y))

    # new sizes
    _, w1r, _ = tile_bgr.shape
    canvas = np.concatenate([uav_bgr, tile_bgr], axis=1)
    offset = w0

    # draw matches
    valid_idx = np.where(matches > -1)[0]
    for i in valid_idx:
        j = int(matches[i])
        (x0, y0) = kpts0[i]
        (x1, y1) = kpts1[j]
        x0i, y0i = int(round(x0)), int(round(y0))
        x1i, y1i = scale_tile_x(x1), scale_tile_y(y1)
        # choose color based on inlier status if given
        if inlier_mask is not None:
            color = (0, 255, 0) if inlier_mask[i] else (0, 0, 255)
        else:
            color = (255, 0, 0)
        cv2.circle(canvas, (x0i, y0i), 3, color, -1)
        cv2.circle(canvas, (offset + x1i, y1i), 3, color, -1)
        cv2.line(canvas, (x0i, y0i), (offset + x1i, y1i), color, 1)
    if out_path is not None:
        cv2.imwrite(out_path, canvas)
    return canvas


# ------------------ 处理单个无人机图像 ------------------
def process_drone_image(uav_path, sat_path, save_dir_base):
    """处理单个无人机图像并返回最佳匹配位置"""
    # 从路径中提取无人机图像编号
    drone_id = os.path.basename(uav_path).split('.')[0]
    save_dir = os.path.join(save_dir_base, f"drone_{drone_id}")
    os.makedirs(save_dir, exist_ok=True)

    # 加载图像
    uav_img = cv2.imread(uav_path, cv2.IMREAD_GRAYSCALE)
    if uav_img is None:
        raise FileNotFoundError(f"Cannot find UAV image: {uav_path}")
    sat_img_color = cv2.imread(sat_path, cv2.IMREAD_COLOR)
    if sat_img_color is None:
        raise FileNotFoundError(f"Cannot find SAT image: {sat_path}")

    # 生成 UAV 掩膜并保存（中间图）
    uav_mask = get_structure_mask(uav_img, CANNY_UAV)
    cv2.imwrite(os.path.join(save_dir, "uav_mask.png"), uav_mask)
    print("Saved UAV mask ->", os.path.join(save_dir, "uav_mask.png"))

    # 初始化 SuperGlue（config 可调）
    config = {
        'superpoint': {
            'nms_radius': 3,
            'keypoint_threshold': KEY,
            'max_keypoints': 1024
        },
        'superglue': {
            'weights': 'outdoor',
            'sinkhorn_iterations': ITERATION,
            'match_threshold': 0.2,
        }
    }
    matching = Matching(config).eval().to(DEVICE)
    print("Loaded Matching (SuperPoint+SuperGlue)")

    # UAV tensor
    uav_tensor = frame2tensor(uav_img, DEVICE)

    # 切分卫星 tile
    tiles, coords = crop_tiles(sat_img_color, tile_size=TILE_SIZE, stride=STRIDE)
    print(f"Generated {len(tiles)} tiles")

    
    # 输出卫星tile
    #for i, (tile, coord) in enumerate(zip(tiles, coords)):
    #    x, y = coord
    #    filename = f"{output_dir}/tile_{i:04d}_x{x}_y{y}.jpg"
    #    cv2.imwrite(filename, tile)
    #    print(f"Saved tile {i}: {filename} (coord: x={x}, y={y})")
       

    # 存放每个 tile 的最佳 variant 结果 (score, x, y, best_variant_img, best_variant_mask, best_pred_info)
    results = []

    for idx, ((x, y), tile) in enumerate(zip(coords, tiles)):
        # note: coords aligned with tiles order
        # iterate candidate rotations & scales, keep the best according to mask-weighted score
        best_score = 0.0
        best_info = None  # (variant_img, variant_mask, pred_dict, uav_kpts, sat_kpts, matches_arr)
        for angle, scale in product(ANGLES, SCALES):
            # rotate
            rot = rotate_image(tile, angle)
            # scale
            if scale != 1.0:
                new_w = int(rot.shape[1] * scale)
                new_h = int(rot.shape[0] * scale)
                if new_w < 16 or new_h < 16:
                    continue
                var = cv2.resize(rot, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            else:
                var = rot

            # convert to gray and build mask
            var_gray = cv2.cvtColor(var, cv2.COLOR_BGR2GRAY)
            var_mask = get_structure_mask(var_gray, CANNY_SAT)

            # run SP+SG (wrap with try to avoid unexpected failures)
            try:
                with torch.no_grad():
                    pred = matching({'image0': uav_tensor, 'image1': frame2tensor(var_gray, DEVICE)})
            except Exception as e:
                print(f"Matching failed tile idx {idx} angle {angle} scale {scale}: {e}")
                continue

            # extract outputs
            kpts0 = pred['keypoints0'][0].cpu().numpy()  # UAV keypoints (N0,2)
            kpts1 = pred['keypoints1'][0].cpu().numpy()  # Tile keypoints (N1,2)
            matches = pred['matches0'][0].cpu().numpy()  # (N0,) indices into kpts1 or -1

            if matches is None or matches.size == 0:
                continue

            # valid matched indices
            valid_idx = np.where(matches > -1)[0]
            if valid_idx.size == 0:
                continue

            # compute mask-weighted score
            weighted_score = 0.0
            pts0 = kpts0[valid_idx]
            pts1 = kpts1[matches[valid_idx].astype(int)]
            for (p0, p1) in zip(pts0, pts1):
                x0, y0 = int(round(p0[0])), int(round(p0[1]))
                x1, y1 = int(round(p1[0])), int(round(p1[1]))
                m0 = 0
                m1 = 0
                if 0 <= y0 < uav_mask.shape[0] and 0 <= x0 < uav_mask.shape[1]:
                    m0 = 1 if uav_mask[y0, x0] > 0 else 0
                if 0 <= y1 < var_mask.shape[0] and 0 <= x1 < var_mask.shape[1]:
                    m1 = 1 if var_mask[y1, x1] > 0 else 0
                # base weight 1.0, bonus if both are structural
                w = 1.0 + (MASK_BONUS if (m0 and m1) else 0.0)
                weighted_score += w

            # optionally compute geometric inliers via RANSAC (for info)
            inliers_count = 0
            if pts0.shape[0] >= 4:
                try:
                    H_mat, mask = cv2.findHomography(pts0.reshape(-1, 1, 2), pts1.reshape(-1, 1, 2), cv2.RANSAC, 5.0)
                    if mask is not None:
                        inliers_count = int(np.sum(mask))
                except Exception:
                    inliers_count = 0

            # final decision uses weighted_score (you may fuse with inliers_count if needed)
            if weighted_score > best_score:
                best_score = weighted_score
                best_info = {
                    'variant_img': var.copy(),
                    'variant_mask': var_mask.copy(),
                    'pred': pred,
                    'pts0': pts0.copy(),
                    'pts1': pts1.copy(),
                    'matches': matches.copy(),
                    'angle': angle,
                    'scale': scale,
                    'inliers_count': inliers_count
                }

        # append result for this tile (use original tile coordinate x,y)
        results.append((best_score, x, y, best_info))
        if idx % 50 == 0:
            print(f"Processed {idx}/{len(tiles)} tiles")

    # sort results by score
    results_sorted = sorted(results, key=lambda x: x[0], reverse=True)
    topk = results_sorted[:TOPK]

    # 记录最佳匹配位置
    best_match = topk[0] if topk else (0, 0, 0, None)
    best_x, best_y = best_match[1], best_match[2]

    # save top-K artifacts and overview
    # 若要输出每张图匹配结果的可视化图，则将下面的行取消注释
    #sat_overview = sat_img_color.copy()
    txt_path = os.path.join(save_dir, "top{}_results.txt".format(TOPK))
    with open(txt_path, "w") as ftxt:
        for rank, (score, x, y, info) in enumerate(topk, start=1):
            ftxt.write(f"Rank {rank}: tile_origin=({x},{y}), score={score:.3f}\n")
            print(f"Rank {rank}: tile_origin=({x},{y}), score={score:.3f}")

            tile_out_dir = os.path.join(save_dir, f"top{rank}_x{x}_y{y}")
            os.makedirs(tile_out_dir, exist_ok=True)

            # baseline: save original tile from big SAT
            orig_tile = sat_img_color[y:y + TILE_SIZE, x:x + TILE_SIZE]
            cv2.imwrite(os.path.join(tile_out_dir, "orig_tile.png"), orig_tile)

            if info is None:
                ftxt.write("  No matching info (no matches found)\n")
                continue

            # save variant image & mask
            var_img = info['variant_img']
            var_mask = info['variant_mask']
            angle = info['angle']
            scale = info['scale']
            cv2.imwrite(os.path.join(tile_out_dir, f"variant_angle{angle}_scale{scale}.png"), var_img)
            cv2.imwrite(os.path.join(tile_out_dir, f"variant_mask_angle{angle}_scale{scale}.png"), var_mask)

            # save UAV mask & UAV image copy
            cv2.imwrite(os.path.join(tile_out_dir, "uav.png"), uav_img)
            cv2.imwrite(os.path.join(tile_out_dir, "uav_mask.png"), uav_mask)

            # matching visualization (draw lines)
            pred = info['pred']
            kpts0 = pred['keypoints0'][0].cpu().numpy()
            kpts1 = pred['keypoints1'][0].cpu().numpy()
            matches_arr = pred['matches0'][0].cpu().numpy()
            # compute RANSAC inlier mask if possible for visualization
            valid_idx = np.where(matches_arr > -1)[0]
            inlier_mask_vis = None
            if valid_idx.size >= 4:
                pts0_vis = kpts0[valid_idx]
                pts1_vis = kpts1[matches_arr[valid_idx].astype(int)]
                try:
                    Hmat, inliers = cv2.findHomography(pts0_vis.reshape(-1, 1, 2), pts1_vis.reshape(-1, 1, 2),
                                                       cv2.RANSAC, 5.0)
                    if inliers is not None:
                        # map back to length N0 mask (False by default)
                        inlier_mask_vis = np.zeros_like(matches_arr, dtype=bool)
                        # mark those matched indices that are inliers
                        for idx_local, mm in enumerate(valid_idx):
                            inlier_mask_vis[mm] = bool(inliers[idx_local])
                except Exception:
                    inlier_mask_vis = None

            # create match visualization image and save
            tile_gray_for_vis = cv2.cvtColor(var_img, cv2.COLOR_BGR2GRAY)
            vis_path = os.path.join(tile_out_dir, "match_vis.png")
            draw_match_vis(uav_img, tile_gray_for_vis, kpts0, kpts1, matches_arr, inlier_mask=inlier_mask_vis,
                           out_path=vis_path)

            # Also save a combined comparison image: UAV | UAV mask | Tile variant | Tile mask
            uav_rgb = cv2.cvtColor(uav_img, cv2.COLOR_GRAY2BGR)
            uav_mask_rgb = cv2.cvtColor(uav_mask, cv2.COLOR_GRAY2BGR)
            tile_var_rgb = var_img.copy()
            tile_mask_rgb = cv2.cvtColor(var_mask, cv2.COLOR_GRAY2BGR)

            # resize for consistent height
            H0 = uav_rgb.shape[0]
            tile_var_rgb_rs = cv2.resize(tile_var_rgb, (tile_var_rgb.shape[1], H0))
            tile_mask_rgb_rs = cv2.resize(tile_mask_rgb, (tile_mask_rgb.shape[1], H0))

            comp = np.hstack([uav_rgb, uav_mask_rgb, tile_var_rgb_rs, tile_mask_rgb_rs])
            comp_path = os.path.join(tile_out_dir, "comparison_uav_mask_tile_mask.png")
            cv2.imwrite(comp_path, comp)

            # annotate overview with rectangle and label
            #cv2.rectangle(sat_overview, (x, y), (x + TILE_SIZE, y + TILE_SIZE), (0, 0, 255), 3)
            #cv2.putText(sat_overview, f"Top{rank}", (x, max(10, y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

    # save overview and index file
    # 若要输出每张图匹配结果的可视化图，则将下面的行取消注释
    #overview_path = os.path.join(save_dir, "overview_topk.png")
    #cv2.imwrite(overview_path, sat_overview)
    #print("Saved overview ->", overview_path)
    print("Saved results text ->", txt_path)

    return drone_id, best_x, best_y


# ------------------ 主流程 ------------------
if __name__ == "__main__":
    print("Device:", DEVICE)

    # 获取所有无人机图像
    drone_images = [os.path.join(drone_images_dir, f) for f in os.listdir(drone_images_dir)
                    if f.endswith('.jpg') or f.endswith('.png')]

    print(f"Found {len(drone_images)} drone images to process")

    # 存储结果
    results = []

    # 处理每个无人机图像
    for uav_path in drone_images:
        try:
            print(f"Processing {uav_path}...")
            drone_id, x, y = process_drone_image(uav_path, sat_path, save_dir_base)
            results.append({"drone_id": drone_id, "pixel_x": x, "pixel_y": y})
            print(f"Completed {uav_path}. Best match at ({x}, {y})")
        except Exception as e:
            print(f"Error processing {uav_path}: {e}")
            results.append({"drone_id": os.path.basename(uav_path).split('.')[0],
                            "pixel_x": -1, "pixel_y": -1, "error": str(e)})

    # 保存结果到Excel
    df = pd.DataFrame(results)
    excel_path = os.path.join(save_dir_base, "drone_matching_results.xlsx")
    df.to_excel(excel_path, index=False)
    print(f"Results saved to {excel_path}")

    print("All done.")