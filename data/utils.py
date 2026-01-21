import cv2
from osgeo import gdal
import numpy as np

def export_geotiff(filename, data_array, dataset_obj):
    """将预测结果导出为GeoTIFF（保留地理参考）"""
    max_side = max(dataset_obj.orig_w, dataset_obj.orig_h)
    canvas = cv2.resize(data_array, (max_side, max_side), interpolation=cv2.INTER_CUBIC)
    
    y_offset = (max_side - dataset_obj.orig_h) // 2
    x_offset = (max_side - dataset_obj.orig_w) // 2
    actual_img = canvas[y_offset:y_offset+dataset_obj.orig_h, x_offset:x_offset+dataset_obj.orig_w]
    actual_img = actual_img * (dataset_obj.global_max - dataset_obj.global_min) + dataset_obj.global_min
    
    driver = gdal.GetDriverByName('GTiff')
    out_ds = driver.Create(filename, dataset_obj.orig_w, dataset_obj.orig_h, 1, gdal.GDT_Float32)
    out_ds.SetGeoTransform(dataset_obj.geo_transform)
    out_ds.SetProjection(dataset_obj.projection)
    out_ds.GetRasterBand(1).WriteArray(actual_img)
    out_ds.FlushCache()
    out_ds = None