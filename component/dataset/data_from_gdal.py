from osgeo import gdal
import os
import numpy as np

# 读入TIF，并接收地理信息
def read_image(path ,name):
    imagepath = os.path.join(path,name)
    dataset = gdal.Open(imagepath)
    # imformation read
    im_width = dataset.RasterXSize
    im_height = dataset.RasterYSize
    im_bands = dataset.RasterCount
    # geo_imformation read
    im_geotrans = dataset.GetGeoTransform()
    im_proj = dataset.GetProjection()
    # numpy write
    im_data = dataset.ReadAsArray(0, 0, im_width, im_height)
    print("gdal ReadAsArray: ", im_data.shape)
    return im_data,im_geotrans,im_proj
# 同时写入地理信息，保存tif
def write_image(image, path, name):
    if 'int8' in image.dtype.name:
        datatype=gdal.GDT_Byte
    elif 'int16' in image.dtype.name:
        datatype = gdal.GDT_UInt16
    else:
        datatype=gdal.GDT_Float32

    if len(image.shape)==3:
        im_bands,im_height,im_width=image.shape
    else:
        im_bands,(im_height, im_width)= 1,image.shape
    # print(f'Writing... name: {name}, shape: {im_bands,(im_height, im_width)}, dtype: {datatype}')
    #writeimage
    save_dir=path
    driver = gdal.GetDriverByName("GTiff")
    pathway = os.path.join(save_dir, name)
    dataset = driver.Create(pathway,im_width, im_height, im_bands, datatype)
    #print('im_width:',dataset.RasterXSize)
    #print('im_height:',dataset.RasterYSize)
    if im_bands == 1:
        dataset.GetRasterBand(1).WriteArray(image)
    else:
        for i in range(im_bands):
            dataset.GetRasterBand(i+1).WriteArray(image[i])
    #dataset.SetGeoTransform(im_geotrans)
    #dataset.SetProjection(im_proj)
# 读入文件，但是不拾取其地理坐标，读入文件为tif
def load_TIF_NoGEO(path,name):
    imagepath = os.path.join(path,name)
    dataset = gdal.Open(imagepath)
    # imformation read
    im_bands = dataset.RasterCount
    im_width = dataset.RasterXSize
    im_height = dataset.RasterYSize
    # print("RasterCount, XSize, YSize: ", im_bands, im_width, im_height)
    # numpy write
    im_data = dataset.ReadAsArray(0, 0, im_width, im_height)
    # print(f"gdal ReadAsArray, shape: {im_data.shape}, dtype: {im_data.dtype}, np.float32: {im_data.dtype == np.float32}")

    return im_data

#define GDT_Byte           1
#define GDT_UInt16         2
#define GDT_Int16          3
#define GDT_UInt32         4
#define GDT_Int32          5
#define GDT_Float32        6
#define GDT_Float64        7
#define GDT_CInt16         8
#define GDT_CInt32         9
#define GDT_CFloat32      10
#define GDT_CFloat64      11
