"""根据无人机测得的位姿信息估计视场中心点经纬度坐标"""

import os
import struct
import pandas as pd
import math
from pyproj import Transformer

INPUT_FILE = "/home/lyt/Nie/UAV_VisLoc/01.xlsx"
DEM_DIR = "/home/lyt/Nie/UAV_VisLoc/dem/"
OUTPUT_FILE = "/home/lyt/Nie/UAV_VisLoc/location_with_dem.xlsx"

# === 1. 坐标转换器 (WGS84 <-> Web Mercator) ===
wgs84_to_merc = Transformer.from_crs("epsg:4326", "epsg:3857", always_xy=True)
merc_to_wgs84 = Transformer.from_crs("epsg:3857", "epsg:4326", always_xy=True)

# === 2. 读取 HGT 文件 ===
def get_elevation_from_hgt(lat, lon, dem_dir="data1/dem"):
    """
    给定经纬度，从对应的 .hgt 文件获取地面高度
    """
    lat_floor = int(math.floor(lat))
    lon_floor = int(math.floor(lon))

    # 拼接文件名，例如 N36E113.hgt
    lat_prefix = f"N{lat_floor:02d}" if lat_floor >= 0 else f"S{-lat_floor:02d}"
    lon_prefix = f"E{lon_floor:03d}" if lon_floor >= 0 else f"W{-lon_floor:03d}"
    hgt_file = os.path.join(dem_dir, f"{lat_prefix}{lon_prefix}.hgt")

    if not os.path.exists(hgt_file):
        raise FileNotFoundError(f"未找到 HGT 文件: {hgt_file}")

    # HGT 文件分辨率：1°x1° tile, 3601x3601 栅格
    size = 3601
    lat_diff = lat - lat_floor
    lon_diff = lon - lon_floor

    row = int((1 - lat_diff) * (size - 1))  # 从北向南
    col = int(lon_diff * (size - 1))        # 从西向东

    with open(hgt_file, "rb") as f:
        f.seek((row * size + col) * 2)  # 每个高度值2字节
        data = f.read(2)
        if not data:
            return None
        elevation = struct.unpack(">h", data)[0]  # big-endian 16-bit
        return float(elevation)

# === 3. 计算相机拍摄点在地面的经纬度 ===
def calculate_target_location(lon, lat, altitude, roll, pitch, yaw, dem_dir):
    ground_alt = get_elevation_from_hgt(lat, lon, dem_dir)
    relative_alt = altitude - ground_alt

    roll_rad = math.radians(roll)
    pitch_rad = math.radians(pitch)
    yaw_rad = math.radians(yaw)

    dx_body = relative_alt * math.tan(pitch_rad)
    dy_body = relative_alt * math.tan(roll_rad)

    dx_world = dx_body * math.sin(yaw_rad) - dy_body * math.cos(yaw_rad)
    dy_world = dx_body * math.cos(yaw_rad) + dy_body * math.sin(yaw_rad)

    x, y = wgs84_to_merc.transform(lon, lat)
    x_new = x + dx_world
    y_new = y + dy_world

    lon_new, lat_new = merc_to_wgs84.transform(x_new, y_new)
    return lon_new, lat_new, relative_alt

# === 4. 主程序 ===
def main():
    input_file = INPUT_FILE
    dem_dir = DEM_DIR
    output_file = OUTPUT_FILE

    df = pd.read_excel(input_file)

    results = []
    for _, row in df.iterrows():
        lon_new, lat_new, rel_alt = calculate_target_location(
            row["lon"], row["lat"], row["GPS_Alt"],
            row["roll"], row["pitch"], row["yaw"],
            dem_dir=dem_dir
        )
        results.append({
            "id": row["id"],
            "target_lon": lon_new,
            "target_lat": lat_new,
            "relative_altitude": rel_alt
        })

    out_df = pd.DataFrame(results)
    print(out_df.head())
    out_df.to_excel(output_file, index=False)
    print(f"结果已保存到 {output_file}")

if __name__ == "__main__":
    main()
