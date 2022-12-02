# -*- coding: utf-8 -*-
"""
Created on Fri May  6 11:56:29 2022

@author: wcamaro
"""

import os
from osgeo import gdal, osr
import osgeo.gdalnumeric as gdn
import numpy as np
import datetime
from scipy.optimize import curve_fit
#from matplotlib import pyplot


output_years_file = r'C:\Users\wcamaro\Documents\ECHOES\06Datasets\WP5\LandsatImages\RIs\images_by_year.txt'
RIS_path_04_05 = r'C:\Users\wcamaro\Documents\ECHOES\06Datasets\WP5\LandsatImages\RIs\LT04_LT05'
RIS_path_07 = r'C:\Users\wcamaro\Documents\ECHOES\06Datasets\WP5\LandsatImages\RIs\LT07'
RIS_path_08_09 = r'C:\Users\wcamaro\Documents\ECHOES\06Datasets\WP5\LandsatImages\RIs\LT08_LT09'

# NDVI_files_04_05 = [RIS_path_04_05 + '\\' + i for i in os.listdir(RIS_path_04_05) if 'NDVI' in i]
# NDVI_files_07 = [RIS_path_07 + '\\' +i for i in os.listdir(RIS_path_07) if 'NDVI' in i]
# NDVI_files_08_09 = [RIS_path_08_09 + '\\' +i for i in os.listdir(RIS_path_08_09) if 'NDVI' in i]

output_path = r'C:\Users\wcamaro\Documents\ECHOES\06Datasets\WP5\LandsatImages\Change_Detection_Alg_Outcomes'

# SIPI_files_04_05 = [RIS_path_04_05 + '\\' + i for i in os.listdir(RIS_path_04_05) if 'SIPI' in i]
# SIPI_files_07 = [RIS_path_07 + '\\' +i for i in os.listdir(RIS_path_07) if 'SIPI' in i]
# SIPI_files_08_09 = [RIS_path_08_09 + '\\' +i for i in os.listdir(RIS_path_08_09) if 'SIPI' in i]

NDWI_files_04_05 = [RIS_path_04_05 + '\\' + i for i in os.listdir(RIS_path_04_05) if 'NDWI' in i]
NDWI_files_07 = [RIS_path_07 + '\\' +i for i in os.listdir(RIS_path_07) if 'NDWI' in i]
NDWI_files_08_09 = [RIS_path_08_09 + '\\' +i for i in os.listdir(RIS_path_08_09) if 'NDWI' in i]


# NDVI_files = sorted((NDVI_files_04_05+NDVI_files_07+NDVI_files_08_09), key=lambda x: int(x.split("\\")[-1].split("_")[3]))
# SIPI_files = sorted((SIPI_files_04_05+SIPI_files_07+SIPI_files_08_09), key=lambda x: int(x.split("\\")[-1].split("_")[3]))
NDWI_files = sorted((NDWI_files_04_05+NDWI_files_07+NDWI_files_08_09), key=lambda x: int(x.split("\\")[-1].split("_")[3]))



def func_ols(x, a0, a1, b1, c1):
    return a0 + a1*np.cos((2*np.pi/365)*x) + b1*np.sin((2*np.pi/365)*x) + c1*(x)



def ind_reg_components_RIs(RI_files):
    dt_array = []
    rf_array =[]
    
    for f in RI_files:
        dt = datetime.datetime.strptime((f.split("\\")[-1].split("_")[3]),'%Y%m%d').toordinal()
        dt_array.append(dt)
        rf = gdal.Open(f)
        bands = [rf.GetRasterBand(i) for i in range(1, rf.RasterCount + 1)]
        rf_array.append(np.array([gdn.BandReadAsArray(band) for band in bands]).astype('float32'))
    rf_array = np.concatenate(rf_array)
    dt_array = np.asarray(dt_array)
        
    return (dt_array,  rf_array)


def rmse(predictions, targets):
    return np.sqrt(((predictions[~np.isnan(targets)] - targets[~np.isnan(targets)]) ** 2).mean())


def finding_changes_from_curves(x,y):
    RI_estimated_ols = np.zeros(np.shape(y))
    a0_ols = np.zeros(np.shape(y))
    a1_ols = np.zeros(np.shape(y))
    b1_ols = np.zeros(np.shape(y))
    c1_ols = np.zeros(np.shape(y))
    diff_estimate_ols = np.zeros(np.shape(y))
    break_points = np.zeros(np.shape(y))
    outliers = np.zeros(np.shape(y))
    abn_slope = np.zeros(np.shape(y))
    RMSE_ols = np.zeros(np.shape(y))
    comp_diff_rmse_ols  = np.zeros(np.shape(y))
    comp_abn_slope = np.zeros(np.shape(y))
    x_fit = x[~np.isnan(y)]
    y_fit = y[~np.isnan(y)]
    init = 0
    end = 15
    while end <= len(y_fit): 
        x0 = x_fit[init:end]
        y0 = y_fit[init:end]
        break_points_est = np.zeros(np.shape(x0))
        popt_ols, _ = curve_fit(func_ols, x0, y0)
        y_estimated = func_ols(x0,  popt_ols[0], popt_ols[1], popt_ols[2], popt_ols[3])
        diff_yest_y0 = abs(y_estimated - y0)
        RMSE_yest_y0 = rmse(y_estimated, y0)
        comp_diff_RMSE = diff_yest_y0/(3*RMSE_yest_y0)
        break_points_est [comp_diff_RMSE>1] = 1 
        check_cum_break = np.convolve(break_points_est,np.ones(3,dtype=int),'valid')
        indx_3Breaks = [i for i, c in enumerate(check_cum_break)  if c==3]
        
        if len(indx_3Breaks) == 0 and end + 3 < len(y_fit):
            end = end + 3
           
        elif len(indx_3Breaks) == 0 and end + 3 >= len(y_fit):
            end = len(y_fit)
            x0 = x_fit[init:end]
            y0 = y_fit[init:end]
            break_points_est = np.zeros(np.shape(x0))
            popt_ols, _ = curve_fit(func_ols, x0, y0)
            y_estimated = func_ols(x0,  popt_ols[0], popt_ols[1], popt_ols[2], popt_ols[3])
            diff_yest_y0 = abs(y_estimated - y0)
            RMSE_yest_y0 = rmse(y_estimated, y0)
            comp_diff_RMSE = diff_yest_y0/(3*RMSE_yest_y0)
            break_points_est [comp_diff_RMSE>1] = 1 
            check_cum_break = np.convolve(break_points_est,np.ones(3,dtype=int),'valid')
            indx_3Breaks = [i for i, c in enumerate(check_cum_break)  if c==3]
            if  len(indx_3Breaks) == 0:
                RI_estimated_ols[(x>=x0[0]) & (x<x0[-1:])] = func_ols(x[np.where((x>=x0[0]) & (x<x0[-1:]))],  popt_ols[0], popt_ols[1], popt_ols[2], popt_ols[3])
                a0_ols[(x>=x0[0]) & (x<x0[-1:])] = popt_ols[0]
                a1_ols[(x>=x0[0]) & (x<x0[-1:])] = popt_ols[1]
                b1_ols[(x>=x0[0]) & (x<x0[-1:])] = popt_ols[2]
                c1_ols[(x>=x0[0]) & (x<x0[-1:])] = popt_ols[3]
                diff_estimate_ols[(x>=x0[0]) & (x<x0[-1:])]  = abs(y[(x>=x0[0]) & (x<x0[-1:])] - RI_estimated_ols[(x>=x0[0]) & (x<x0[-1:])])
                RMSE_ols[(x>=x0[0]) & (x<x0[-1:])]  = rmse(RI_estimated_ols[(x>=x0[0]) & (x<x0[-1:])] , y[(x>=x0[0]) & (x<x0[-1:])]) 
                comp_diff_rmse_ols[(x>=x0[0]) & (x<x0[-1:])]  =  diff_estimate_ols[(x>=x0[0]) & (x<x0[-1:])] /(3*RMSE_ols[(x>=x0[0]) & (x<x0[-1:])] )
                outliers [comp_diff_rmse_ols>1] = 1
                comp_abn_slope [(x>=x0[0]) & (x<x0[-1:])] = popt_ols[3]/((3*RMSE_ols[(x>=x0[0]) & (x<x0[-1:])])/(x0[-1:]-x0[0]))
                abn_slope [comp_abn_slope>1] = 100
                break
            else:
                RI_estimated_ols[(x>=x0[0]) & (x<x0[-1:])] = func_ols(x[np.where((x>=x0[0]) & (x<x0[-1:]))],  popt_ols[0], popt_ols[1], popt_ols[2], popt_ols[3])
                a0_ols[(x>=x0[0]) & (x<x0[-1:])] = popt_ols[0]
                a1_ols[(x>=x0[0]) & (x<x0[-1:])] = popt_ols[1]
                b1_ols[(x>=x0[0]) & (x<x0[-1:])] = popt_ols[2]
                c1_ols[(x>=x0[0]) & (x<x0[-1:])] = popt_ols[3]
                diff_estimate_ols[(x>=x0[0]) & (x<x0[-1:])]  = abs(y[(x>=x0[0]) & (x<x0[-1:])] - RI_estimated_ols[(x>=x0[0]) & (x<x0[-1:])])
                RMSE_ols[(x>=x0[0]) & (x<x0[-1:])]  = rmse(RI_estimated_ols[(x>=x0[0]) & (x<x0[-1:])] , y[(x>=x0[0]) & (x<x0[-1:])]) 
                comp_diff_rmse_ols[(x>=x0[0]) & (x<x0[-1:])]  =  diff_estimate_ols[(x>=x0[0]) & (x<x0[-1:])] /(3*RMSE_ols[(x>=x0[0]) & (x<x0[-1:])] )
                outliers [comp_diff_rmse_ols>1] = 1
                comp_abn_slope [(x>=x0[0]) & (x<x0[-1:])] = popt_ols[3]/((3*RMSE_ols[(x>=x0[0]) & (x<x0[-1:])])/(x0[-1:]-x0[0]))
                abn_slope [comp_abn_slope>1] = 100
                x_break_ind =  [i for i, c in enumerate(x) if c == x0[indx_3Breaks[0]]]
                break_points[x_break_ind] = 4
                break
        else:
            # print ('estoy aca')
            if indx_3Breaks[0] <3:
                RI_estimated_ols[(x>=x0[0]) & (x<x0[-1:])] = func_ols(x[np.where((x>=x0[0]) & (x<x0[-1:]))],  popt_ols[0], popt_ols[1], popt_ols[2], popt_ols[3])
                a0_ols[(x>=x0[0]) & (x<x0[-1:])] = popt_ols[0]
                a1_ols[(x>=x0[0]) & (x<x0[-1:])] = popt_ols[1]
                b1_ols[(x>=x0[0]) & (x<x0[-1:])] = popt_ols[2]
                c1_ols[(x>=x0[0]) & (x<x0[-1:])] = popt_ols[3]
                diff_estimate_ols[(x>=x0[0]) & (x<x0[-1:])]  = abs(y[(x>=x0[0]) & (x<x0[-1:])] - RI_estimated_ols[(x>=x0[0]) & (x<x0[-1:])])
                RMSE_ols[(x>=x0[0]) & (x<x0[-1:])]  = rmse(RI_estimated_ols[(x>=x0[0]) & (x<x0[-1:])] , y[(x>=x0[0]) & (x<x0[-1:])]) 
                comp_diff_rmse_ols[(x>=x0[0]) & (x<x0[-1:])]  =  diff_estimate_ols[(x>=x0[0]) & (x<x0[-1:])] /(3*RMSE_ols[(x>=x0[0]) & (x<x0[-1:])] )
                outliers [comp_diff_rmse_ols>1] = 1
                comp_abn_slope [(x>=x0[0]) & (x<x0[-1:])] = popt_ols[3]/((3*RMSE_ols[(x>=x0[0]) & (x<x0[-1:])])/(x0[-1:]-x0[0]))
                abn_slope [comp_abn_slope>1] = 100
                x_break_ind =  [i for i, c in enumerate(x) if c == x0[indx_3Breaks[0]]]
                break_points[x_break_ind] = 20
                init = init + 3
                end = end + 3
                break_points[x==x_fit[init]] = 4
            else:
                RI_estimated_ols[(x>=x0[0]) & (x<x0[indx_3Breaks[0]])] = func_ols(x[np.where((x>=x0[0]) & (x<x0[indx_3Breaks[0]]))],  popt_ols[0], popt_ols[1], popt_ols[2], popt_ols[3])
                a0_ols[(x>=x0[0]) & (x<x0[indx_3Breaks[0]])] = popt_ols[0]
                a1_ols[(x>=x0[0]) & (x<x0[indx_3Breaks[0]])] = popt_ols[1]
                b1_ols[(x>=x0[0]) & (x<x0[indx_3Breaks[0]])] = popt_ols[2]
                c1_ols[(x>=x0[0]) & (x<x0[indx_3Breaks[0]])] = popt_ols[3]
                diff_estimate_ols[(x>=x0[0]) & (x<x0[indx_3Breaks[0]])]  = abs(y[(x>=x0[0]) & (x<x0[indx_3Breaks[0]])] - RI_estimated_ols[(x>=x0[0]) & (x<x0[indx_3Breaks[0]])])
                RMSE_ols[(x>=x0[0]) & (x<x0[indx_3Breaks[0]])]  = rmse(RI_estimated_ols[(x>=x0[0]) & (x<x0[indx_3Breaks[0]])] , y[(x>=x0[0]) & (x<x0[indx_3Breaks[0]])]) 
                comp_diff_rmse_ols[(x>=x0[0]) & (x<x0[indx_3Breaks[0]])]  =  diff_estimate_ols[(x>=x0[0]) & (x<x0[indx_3Breaks[0]])] /(3*RMSE_ols[(x>=x0[0]) & (x<x0[indx_3Breaks[0]])] )
                outliers [comp_diff_rmse_ols>1] = 1
                comp_abn_slope [(x>=x0[0]) & (x<x0[indx_3Breaks[0]])] = popt_ols[3]/((3*RMSE_ols[(x>=x0[0]) & (x<x0[indx_3Breaks[0]])])/(x0[indx_3Breaks[0]]-x0[0]))
                abn_slope [comp_abn_slope>1] = 100
                x_break_ind =  [i for i, c in enumerate(x) if c == x0[indx_3Breaks][0]]
                break_points[x_break_ind] = 4
                init = init + indx_3Breaks[0]
                end = init + 15
                
    outliers [break_points == 4] = 0
    outliers [break_points == 20] = 0
    abn_slope [break_points == 4] = 0
    abn_slope [break_points == 20] = 0
    abn_slope [outliers == 1] = 0
    whole_break_points = break_points + outliers  + abn_slope

    return RI_estimated_ols, a0_ols, a1_ols, b1_ols, c1_ols, RMSE_ols, whole_break_points



def change_detection_algorithm (RI_files):    
    i = 0
    dt_RI, rf_RI  = ind_reg_components_RIs(RI_files)
    len_RI_F = len(RI_files)
    # flat_RI = rf_RI.flatten(order='F')[4166580:4167650]
    flat_RI = rf_RI.flatten(order='F')
    flat_RI_estimated = np.zeros(np.shape(flat_RI), dtype = np.float16)
    flat_a0 = np.zeros(np.shape(flat_RI), dtype = np.float16)
    flat_a1 = np.zeros(np.shape(flat_RI), dtype = np.float16)
    flat_b1 = np.zeros(np.shape(flat_RI), dtype = np.float16)
    flat_c1 = np.zeros(np.shape(flat_RI), dtype = np.float16)
    flat_RMSE = np.zeros(np.shape(flat_RI), dtype = np.float16)
    break_points = np.zeros(np.shape(flat_RI), dtype = np.float16)
    for inv in range(0, np.shape(rf_RI)[1]*np.shape(rf_RI)[2]):
        flat_sample = flat_RI[i:i+ len_RI_F]
        x = dt_RI
        y = np.asarray(flat_sample)
        
        if len(y[~np.isnan(y)])>=15:
            flat_RI_estimated[i:i+ len_RI_F], flat_a0[i:i+ len_RI_F], flat_a1[i:i+ len_RI_F], flat_b1[i:i+ len_RI_F], flat_c1[i:i+ len_RI_F], flat_RMSE[i:i+ len_RI_F], break_points[i:i+ len_RI_F] = finding_changes_from_curves(x,y)
            
        else:
            flat_RI_estimated[i:i+ len_RI_F] = np.nan
            flat_a0[i:i+ len_RI_F] = np.nan
            flat_a1[i:i+ len_RI_F] = np.nan
            flat_b1[i:i+ len_RI_F] = np.nan
            flat_c1[i:i+ len_RI_F] = np.nan
            flat_RMSE[i:i+ len_RI_F] = np.nan
            break_points[i:i+ len_RI_F] = np.nan
            
        i = i+len_RI_F
    
    
    return dt_RI, flat_RI, flat_RI_estimated, flat_a0, flat_a1, flat_b1, flat_c1, flat_RMSE, break_points,  np.shape(rf_RI)


def write_tiff(driver, filename, nrows, ncols, rf, proj, array):
    outputRaster= driver.Create(filename,nrows,ncols,1,gdal.GDT_Float32)
    outputRaster.SetGeoTransform(rf.GetGeoTransform())
    outputRaster.SetProjection(proj.ExportToWkt())
    outputRaster.FlushCache()
    outband = outputRaster.GetRasterBand(1)
    outband.SetNoDataValue(-999999)
    outband.WriteArray(array)
    outband = None
    outputRaster = None

def array_to_tiff(RI, RI_filenames, dt_RI, flat_RI_estimated_ols, flat_RI_a0, flat_RI_a1,  
                  flat_RI_b1, flat_RI_c1, flat_RI_RMSE, flat_RI_break_points, RI_shape, output_path):
    RI_estimated_array = np.reshape(flat_RI_estimated_ols, RI_shape, order = 'F')
    RI_breakpoints_array = np.reshape(flat_RI_break_points, RI_shape, order = 'F')
    RI_a0_array = np.reshape(flat_RI_a0, RI_shape, order = 'F')
    RI_a1_array = np.reshape(flat_RI_a1, RI_shape, order = 'F')
    RI_b1_array = np.reshape(flat_RI_b1, RI_shape, order = 'F')
    RI_c1_array = np.reshape(flat_RI_c1, RI_shape, order = 'F')
    RI_RMSE_array = np.reshape(flat_RI_RMSE, RI_shape, order = 'F')
    RI_breakpoints_array = np.reshape(flat_RI_break_points, RI_shape, order = 'F')
    rf = gdal.Open(RI_filenames[0])
    proj = osr.SpatialReference(wkt=rf.GetProjection())
    driver = gdal.GetDriverByName('GTiff') 
    ncols, nrows = RI_shape[1:]
    for i in range(0, len(dt_RI)):
        filename_estimated = output_path + "\\" + RI + "\\"+ 'Landsat_OLS_Estimated_%s_%s.tif' %(RI,datetime.date.fromordinal(dt_RI[i]))
        filename_a0= output_path + "\\" + RI + "\\"+ 'Landsat_OLS_Estimated_%s_%s_a0.tif' %(RI,datetime.date.fromordinal(dt_RI[i]))
        filename_a1 = output_path + "\\" + RI + "\\"+ 'Landsat_OLS_Estimated_%s_%s_a1.tif' %(RI,datetime.date.fromordinal(dt_RI[i]))
        filename_b1 = output_path + "\\" + RI + "\\"+ 'Landsat_OLS_Estimated_%s_%s_b1.tif' %(RI,datetime.date.fromordinal(dt_RI[i]))
        filename_c1 = output_path + "\\" + RI + "\\"+ 'Landsat_OLS_Estimated_%s_%s_c1.tif' %(RI,datetime.date.fromordinal(dt_RI[i]))
        filename_RMSE = output_path + "\\" + RI + "\\"+ 'Landsat_OLS_Estimated_%s_%s_RMSE.tif' %(RI,datetime.date.fromordinal(dt_RI[i]))
        filename_break_points = output_path + "\\" + RI + "\\"+ 'Landsat_BreakPoints_%s_%s.tif' %(RI,datetime.date.fromordinal(dt_RI[i]))
        
        write_tiff(driver, filename_estimated, nrows, ncols, rf, proj, RI_estimated_array[i])
        write_tiff(driver, filename_a0, nrows, ncols, rf, proj, RI_a0_array[i])
        write_tiff(driver, filename_a1, nrows, ncols, rf, proj, RI_a1_array[i])
        write_tiff(driver, filename_b1, nrows, ncols, rf, proj, RI_b1_array[i])
        write_tiff(driver, filename_c1, nrows, ncols, rf, proj, RI_c1_array[i])
        write_tiff(driver, filename_RMSE, nrows, ncols, rf, proj, RI_RMSE_array[i])
        write_tiff(driver, filename_break_points, nrows, ncols, rf, proj, RI_breakpoints_array[i])    
        
def dates_with_breaks(dt_RI,flat_RI,break_points, output_path, RI, ind_value=4):
    pixel_date_locators = [i+len(dt_RI) for i in range(0,len(flat_RI), len(dt_RI))]
    id_break_RI = [i for i,c in enumerate(break_points) if c==ind_value]
    a = []
    for ib in range(0, len(id_break_RI)):
         for jb in range(0, len(pixel_date_locators)):
             if (id_break_RI[ib] - pixel_date_locators[jb]>0) & (id_break_RI[ib] - pixel_date_locators[jb]<len(dt_RI)): 
                 a.append(id_break_RI[ib] - pixel_date_locators[jb])
    b = sorted(list(set(a)))
    dates_with_breaks_RI = []
    for ic in b:
        dates_with_breaks_RI.append(datetime.date.fromordinal(dt_RI[ic]))
    with open(output_path + "\\" + RI + "\\"+ 'Dates_with_breaks_%s_%s.txt' % (RI, str(ind_value)), 'w') as fp:
        for item in dates_with_breaks_RI:
            # write each item on a new line
            fp.write("%s\n" % item)
        
    
    
#(dt_NDVI, flat_NDVI, flat_NDVI_estimated_ols, flat_NDVI_a0, flat_NDVI_a1, flat_NDVI_b1, flat_NDVI_c1, flat_NDVI_RMSE, flat_NDVI_break_points,  NDVI_shape) = change_detection_algorithm(NDVI_files)
#array_to_tiff('NDVI', NDVI_files, dt_NDVI, flat_NDVI_estimated_ols, flat_NDVI_a0, flat_NDVI_a1, flat_NDVI_b1, flat_NDVI_c1, flat_NDVI_RMSE, flat_NDVI_break_points, NDVI_shape, output_path)
#dates_with_breaks(dt_NDVI,flat_NDVI,flat_NDVI_break_points, output_path, 'NDVI', ind_value=4)


# (dt_SIPI, flat_SIPI, flat_SIPI_estimated_ols, flat_SIPI_a0, flat_SIPI_a1, flat_SIPI_b1, flat_SIPI_c1, flat_SIPI_RMSE, flat_SIPI_break_points,  SIPI_shape) = change_detection_algorithm(SIPI_files)
# array_to_tiff('SIPI', SIPI_files, dt_SIPI, flat_SIPI_estimated_ols, flat_SIPI_a0, flat_SIPI_a1, flat_SIPI_b1, flat_SIPI_c1, flat_SIPI_RMSE, flat_SIPI_break_points, SIPI_shape, output_path)
# dates_with_breaks(dt_SIPI,flat_SIPI,flat_SIPI_break_points, output_path, 'SIPI', ind_value=4)


(dt_NDWI, flat_NDWI, flat_NDWI_estimated_ols, flat_NDWI_a0, flat_NDWI_a1, flat_NDWI_b1, flat_NDWI_c1, flat_NDWI_RMSE, flat_NDWI_break_points,  NDWI_shape) = change_detection_algorithm(NDWI_files)
array_to_tiff('NDWI', NDWI_files, dt_NDWI, flat_NDWI_estimated_ols, flat_NDWI_a0, flat_NDWI_a1, flat_NDWI_b1, flat_NDWI_c1, flat_NDWI_RMSE, flat_NDWI_break_points, NDWI_shape, output_path)
dates_with_breaks(dt_NDWI,flat_NDWI,flat_NDWI_break_points, output_path, 'NDWI', ind_value=4)



