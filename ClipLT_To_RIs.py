# -*- coding: utf-8 -*-
"""
Created on Thu Feb 17 17:39:04 2022

@author: wcamaro
"""

from osgeo import gdal, osr
import osgeo.gdalnumeric as gdn

import os 
import numpy as np

os.environ['PROJ_LIB'] = r'C:\Users\wcamaro\Anaconda3\Library\share\proj'

os.environ['GDAL_DATA'] =  r'C:\Users\wcamaro\Anaconda3\Library\share\gdal'

Rad_Inds = ['NDVI', 'SIPI','NDWI']

# images_path = r'C:\Users\wcamaro\Documents\ECHOES\06Datasets\WP5\LandsatImages\Clipfiles\Dyfi\LT04_LT05'
#images_path = r'C:\Users\wcamaro\Documents\ECHOES\06Datasets\WP5\LandsatImages\Clipfiles\Dyfi\LT07'
# images_path = r'C:\Users\wcamaro\Documents\ECHOES\06Datasets\WP5\LandsatImages\Clipfiles\Dyfi\LT08_LT09'
# RIS_path = r'C:\Users\wcamaro\Documents\ECHOES\06Datasets\WP5\LandsatImages\RIs\Dyfi\LT04_LT05'
#RIS_path = r'C:\Users\wcamaro\Documents\ECHOES\06Datasets\WP5\LandsatImages\RIs\Dyfi\LT07'
# RIS_path = r'C:\Users\wcamaro\Documents\ECHOES\06Datasets\WP5\LandsatImages\RIs\Dyfi\LT08_LT09'


images_path = r'C:\Users\wcamaro\Documents\ECHOES\06Datasets\WP5\LandsatImages\Clipfiles\Ballyteige\LT04_LT05'
#images_path = r'C:\Users\wcamaro\Documents\ECHOES\06Datasets\WP5\LandsatImages\Clipfiles\Ballyteige\LT07'
# images_path = r'C:\Users\wcamaro\Documents\ECHOES\06Datasets\WP5\LandsatImages\Clipfiles\Ballyteige\LT08_LT09'
RIS_path = r'C:\Users\wcamaro\Documents\ECHOES\06Datasets\WP5\LandsatImages\RIs\Ballyteige\LT04_LT05'
#RIS_path = r'C:\Users\wcamaro\Documents\ECHOES\06Datasets\WP5\LandsatImages\RIs\Ballyteige\LT07'
# RIS_path = r'C:\Users\wcamaro\Documents\ECHOES\06Datasets\WP5\LandsatImages\RIs\Ballyteige\LT08_LT09'
LTfiles = os.listdir(images_path)


cloud_free_codes =[]

# #L04-L05-L07
for i in range(0,65535):
    b = str(np.binary_repr(i+1).zfill(16)) 
    if (b[4:8]) == '0101': 
        cloud_free_codes.append(i+1)
    b = None
    
#L08-L09
# for i in range(0,65535):
#     b = str(np.binary_repr(i+1).zfill(16)) 
#     if (b[4:8]) == '0101'and (b[0:2]) == '01':  
#         cloud_free_codes.append(i+1)
#     b = None
    

for f in LTfiles:
    rf = gdal.Open(images_path + '\\' + f)
    bands = [rf.GetRasterBand(i) for i in range(1, rf.RasterCount + 1)]
    rf_array =  np.array([gdn.BandReadAsArray(band) for band in bands]).astype('float32')
    mask = np.isin(rf_array[4],cloud_free_codes)
    for ri in Rad_Inds:
        if ri == 'NDVI':
            array = ((rf_array[3]*2.75e-05-0.2) - (rf_array[2]*2.75e-05-0.2)) / ((rf_array[3]*2.75e-05-0.2) + (rf_array[2]*2.75e-05-0.2))
            array [mask == 0] = np.nan
        elif ri == 'SIPI':
            array = ((rf_array[3]*2.75e-05-0.2) - (rf_array[0]*2.75e-05-0.2)) / ((rf_array[3]*2.75e-05-0.2) - (rf_array[2]*2.75e-05-0.2))
            array[np.isinf(array)] = np.nan
            array[np.isneginf(array)] = np.nan
            array [mask == 0] = np.nan
        else:
            array = ((rf_array[1]*2.75e-05-0.2) - (rf_array[3]*2.75e-05-0.2)) / ((rf_array[1]*2.75e-05-0.2) + (rf_array[3]*2.75e-05-0.2))
            array[np.isinf(array)] = np.nan
            array[np.isneginf(array)] = np.nan
            array [mask == 0]  = np.nan
        name = RIS_path + '//' + f[:-4] + '_%s' % ri + f[-4:] 
        driver = gdal.GetDriverByName('GTiff') 
        proj = osr.SpatialReference(wkt=rf.GetProjection())
        ncols, nrows = np.shape(array)
        outputRaster= driver.Create(name,nrows,ncols,1,gdal.GDT_Float32)
        outputRaster.SetGeoTransform(rf.GetGeoTransform())
        outputRaster.SetProjection(proj.ExportToWkt())
        outputRaster.FlushCache()
        outband = outputRaster.GetRasterBand(1)
        outband.SetNoDataValue(-999999)
        outband.WriteArray(array)
        outband = None
        outputRaster = None

