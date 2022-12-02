# -*- coding: utf-8 -*-
"""
Created on Tue Feb 15 10:07:00 2022

@author: wcamaro
"""

from osgeo import gdal, ogr, osr
import os 
import tarfile

os.environ['PROJ_LIB'] = r'C:\Users\wcamaro\Anaconda3\Library\share\proj'

os.environ['GDAL_DATA'] =  r'C:\Users\wcamaro\Anaconda3\Library\share\gdal'



# POLYGON_AOIS = {'Dyfi':'POLYGON ((-4.2 52.7, -3.8 52.7, -3.8 52.4, -4.2 52.4, -4.2 52.7))',
#                 'Ballyteige':'POLYGON ((-4.2 52.7, -3.8 52.7, -3.8 52.4, -4.2 52.4, -4.2 52.7))'}

POLYGON_AOIS = {'Dyfi':'POLYGON ((-4.015 52.54, -3.97 52.54, -3.97 52.515, -4.015 52.515, -4.015 52.7))',
                'Ballyteige':'POLYGON ((-6.9 52.32, -6.3 52.32, -6.3 52.16, -6.9 52.16, -6.9 52.32))'}




#Input and output paths
#inputPath = '../Images/'
inputPath = r'C:\Users\wcamaro\Documents\ECHOES\06Datasets\WP5\LandsatImages\Raw_files_Tar\Landsat4_5'
# inputPath = r'C:\Users\wcamaro\Documents\ECHOES\06Datasets\WP5\LandsatImages\Raw_files_Tar\Landsat7'
# inputPath = r'C:\Users\wcamaro\Documents\ECHOES\06Datasets\WP5\LandsatImages\Raw_files_Tar\Landsat8_9'
# inputPath = r'C:\Users\wcamaro\Documents\ECHOES\06Datasets\WP5\LandsatImages\ttt'
Landsatfiles = os.listdir(inputPath)
# outputPath = r'C:\Users\wcamaro\Documents\ECHOES\06Datasets\WP5\LandsatImages\Clipfiles\Dyfi\LT04_LT05'
# outputPath = r'C:\Users\wcamaro\Documents\ECHOES\06Datasets\WP5\LandsatImages\Clipfiles\Dyfi\LT07'
# outputPath = r'C:\Users\wcamaro\Documents\ECHOES\06Datasets\WP5\LandsatImages\Clipfiles\Dyfi\LT08_LT09'
outputPath = r'C:\Users\wcamaro\Documents\ECHOES\06Datasets\WP5\LandsatImages\Clipfiles\Ballyteige\LT04_LT05'
# outputPath = r'C:\Users\wcamaro\Documents\ECHOES\06Datasets\WP5\LandsatImages\Clipfiles\Ballyteige\LT07'
# outputPath = r'C:\Users\wcamaro\Documents\ECHOES\06Datasets\WP5\LandsatImages\Clipfiles\Ballyteige\LT08_LT09'


temp = outputPath + '\\' + 'temp'




if os.path.exists(outputPath):
    print ('Output directory already exists')
else:    
    os.makedirs(outputPath)
    
if os.path.exists(temp):
    print ('temp directory already exists')
else:    
    os.makedirs(temp)
    
    


#Vector Data
shp_driver = ogr.GetDriverByName('Esri Shapefile')
# temp_shp = temp + '\\' + 'Dyfi.shp'
temp_shp = temp + '\\' + 'Ballyteige.shp'

if os.path.exists(temp_shp):
    print('temp shp already exists')
else:
    ds_shp = shp_driver.CreateDataSource(temp_shp)
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326)
    # layer = ds_shp.CreateLayer("%s" % ('DYfi'), srs, ogr.wkbPolygon)
    layer = ds_shp.CreateLayer("%s" % ('Ballyteige'), srs, ogr.wkbPolygon)
    
    
    # Add an ID field
    idField = ogr.FieldDefn("id", ogr.OFTInteger)
    layer.CreateField(idField)
    
    # Create the feature and set values
    featureDefn = layer.GetLayerDefn()
    feature = ogr.Feature(featureDefn)
    feature.SetField("id", 1)
    # polygon = ogr.CreateGeometryFromWkt(POLYGON_AOIS['Dyfi'])
    polygon = ogr.CreateGeometryFromWkt(POLYGON_AOIS['Ballyteige'])
    feature.SetGeometry(polygon)
    
    layer.CreateFeature(feature)
    
    feature = None
    ds_shp = None


#Landsat files 
LandsatBands = {'L5Bs':['B1', 'B2', 'B3','B4', 'QA_PIXEL'],
                'L7Bs':['B1', 'B2', 'B3','B4', 'QA_PIXEL'],
                'L89Bs':['B2', 'B3', 'B4','B5', 'QA_PIXEL']
                }

for f in Landsatfiles:
    if f[:4] == 'LC08' or f[:4] == 'LC09' :
       LB = LandsatBands['L89Bs']
    elif f[:4] == 'LT05':
       LB = LandsatBands['L5Bs']
    else:
       LB = LandsatBands['L7Bs'] 
    bandlist = []
    for b in LB:
        tarf = tarfile.open(inputPath + '\\' + f)
        bandlist.append([band for band in tarf.getnames() if band[-(len(b)+4):] == '%s.TIF' %b][0])
        for b_tf in bandlist:
            tarf.extract(b_tf, temp)
    bandfiles = [temp + '\\' + n_file for n_file in bandlist]   
    VRT = temp + '\\'+ 'OutputImage.vrt'
    gdal.BuildVRT(VRT, bandfiles, separate=True, callback=gdal.TermProgress_nocb)
    InputImage = gdal.Open(VRT, 0)
    options = gdal.WarpOptions(cutlineDSName=temp_shp,cropToCutline=True)
    # outRaster = gdal.Warp(destNameOrDestDS=outputPath + '\\' + bandlist[0].split("\\")[-1][:-6]+
    #                       ''.join(LB)+'_Dyfi'+bandlist[0].split("\\")[-1][-4:],
    #                     srcDSOrSrcDSTab= VRT,
    #                     options=options)
    outRaster = gdal.Warp(destNameOrDestDS=outputPath + '\\' + bandlist[0].split("\\")[-1][:-6]+
                          ''.join(LB)+'_Ballyteige'+bandlist[0].split("\\")[-1][-4:],
                        srcDSOrSrcDSTab= VRT,
                        options=options)
 
 
    outRaster = None
    InputImage = None
    tarf = None
    bandfiles = None
    if os.path.exists(VRT):
        os.remove(VRT)
    

if os.path.exists(temp_shp):
         shp_driver.DeleteDataSource(temp_shp)
         
for temp_f in os.listdir(temp):
    if os.path.exists(temp + '\\' + temp_f ):
        os.remove(temp + '\\' + temp_f) 

    
if os.path.exists(temp):
    os.rmdir(temp) 
