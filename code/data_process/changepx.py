"""根据视场中心点的经纬度坐标和卫星图像信息计算视场中心点像素坐标"""

import pandas as pd

# 大图参数
'''
<anyanghe>
top_left = (36.1511827146, 113.8018793280)
top_right = (36.1511827146, 113.9392149304)
bottom_left = (36.0424177655, 113.8018793280)
bottom_right = (36.0424177655, 113.9392149304)
img_width = 12800
img_height = 12544

<cuijiaqiao>
top_left = (36.1688993878, 114.4116281439)
top_right = (36.1688993878, 114.5846419905)
bottom_left = (36.0979257426, 114.4116281439)
bottom_right = (36.0979257426, 114.5846419905)
img_width = 16128
img_height = 8192

<1497>
top_left = (31.6212584497, 121.4718407462)
top_right = (31.6212584497, 121.5092777775)
bottom_left = (31.5887263080, 121.4718407462)
bottom_right = (31.5887263080, 121.5092777775)
img_width = 6955
img_height = 7114

<1517>
top_left = (31.6523912632, 121.4322418326)
top_right = (31.6523912632, 121.4890306816)
bottom_left = (31.6211067671, 121.4322418326)
bottom_right = (31.6211067671, 121.4890306816)
img_width = 10592
img_height = 6836

<1653>
top_left = (31.6890817400, 121.3614018577)
top_right = (31.6890817400, 121.4327709441)
bottom_left = (31.6457556228, 121.3614018577)
bottom_right = (31.6457556228, 121.4327709441)
img_width = 13283
img_height = 9485

<1868>
top_left = (31.7309561888, 121.2815311174)
top_right = (31.7309561888, 121.3616613460)
bottom_left = (31.6837186588, 121.2815311174)
bottom_right = (31.6837186588, 121.3616613460)
img_width = 14927
img_height = 10342

<2090>
top_left = (31.7785562696, 121.2478234383)
top_right = (31.7785562696, 121.2887647080)
bottom_left = (31.7303165384, 121.2478234383)
bottom_right = (31.7303165384, 121.2887647080)
img_width = 7628
img_height = 10552

<2244>
top_left = (31.8556185117, 121.2381968073)
top_right = (31.8556185117, 121.2732852768)
bottom_left = (31.7782984617, 121.2381968073)
bottom_right = (31.7782984617, 121.2732852768)
img_width = 6515
img_height = 16916

<3509>
top_left = (31.6152237944, 121.5065229255)
top_right = (31.6152237944, 121.5649595569)
bottom_left = (31.5852940608, 121.5065229255)
bottom_right = (31.5852940608, 121.5649595569)
img_width = 10856
img_height = 6552

<3663>
top_left = (31.5911857486, 121.5642367494)
top_right = (31.5911857486, 121.6247229469)
bottom_left = (31.5649894568, 121.5642367494)
bottom_right = (31.5649894568, 121.6247229469)
img_width = 11261
img_height = 5719

<3815>
top_left = (31.5700247170, 121.6233644944)
top_right = (31.5700247170, 121.6880373773)
bottom_left = (31.5486436669, 121.6233644944)
bottom_right = (31.5486436669, 121.6880373773)
img_width = 12062
img_height = 4663

<3972>
top_left = (31.5528180642, 121.6874618782)
top_right = (31.5528180642, 121.7431796518)
bottom_left = (31.5298498599, 121.6874618782)
bottom_right = (31.5298498599, 121.7431796518)
img_width = 10379
img_height = 5013

<4111>
top_left = (31.5354604057, 121.7425593464)
top_right = (31.5354604057, 121.8006069822)
bottom_left = (31.5034867909, 121.7425593464)
bottom_right = (31.5034867909, 121.8006069822)
img_width = 10824
img_height = 6984

'''
# 四角坐标 (纬度, 经度)
top_left = (36.1688993878, 114.4116281439)
top_right = (36.1688993878, 114.5846419905)
bottom_left = (36.0979257426, 114.4116281439)
bottom_right = (36.0979257426, 114.5846419905)
img_width = 16128
img_height = 8192

lat_max = top_left[0]
lat_min = bottom_left[0]
lon_min = top_left[1]
lon_max = top_right[1]

def latlon_to_pixel(lat, lon):
    """将经纬度映射到图像像素坐标"""
    x = (lon - lon_min) / (lon_max - lon_min) * img_width
    y = (lat_max - lat) / (lat_max - lat_min) * img_height
    return int(x), int(y)

# === 读取修正后经纬度文件 ===
df = pd.read_excel("/home/lyt/Nie/cuijiaqiao/location_with_dem.xlsx")  # 包含 id, target_lon, target_lat

results = []
for _, row in df.iterrows():
    x, y = latlon_to_pixel(row["target_lat"], row["target_lon"])
    results.append({
        "id": row["id"],
        "pixel_x": x,
        "pixel_y": y
    })

# 保存到新的 Excel
out_df = pd.DataFrame(results)
print(out_df)
out_df.to_excel("/home/lyt/Nie/cuijiaqiao/pxGT_dem.xlsx", index=False)
