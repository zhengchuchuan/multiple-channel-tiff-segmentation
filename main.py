import math
import os
import xml.etree.ElementTree as ET
import numpy as np
from osgeo import gdal, gdal_array
from tqdm import tqdm

'''
对多通道的tif图像进行分割,
默认尺寸:640*640,边缘添加白边补整,
切割顺序:从左往右,从上往下
子图命名规范:原图名称_000001开始递增
'''


def read_tif(path):
    """
    读取tif文件,并将其转换为ndarray格式
    返回:数组数据,通道数,高度,宽度
    """
    try:
        dataset = gdal.Open(path, gdal.GA_Update)
        if dataset is None:
            raise IOError("无法打开TIF文件，请检查文件路径。")
    except IOError as e:  # IO异常
        print(str(e))
    except Exception as e:  # 其他异常
        print("其他异常：", str(e))
    # 获取TIF文件的通道数
    channels = dataset.RasterCount
    # 获取TIF文件的宽度和高度
    width = dataset.RasterXSize
    height = dataset.RasterYSize
    # 将tif文件读取为ndarray数组
    # 数据的格式(通道,高度,宽度)
    arr_dataset = dataset.ReadAsArray()  # 转为numpy格式
    del dataset
    return arr_dataset, channels, height, width


def write_tif(path, dataset, data_type, channels, width, height):
    """
    写入tif文件,
    参数:导出子图的路径,子图的数组,保存文件的类型,通道数,拆分的图像宽度,拆分的图像高度
    """
    driver = gdal.GetDriverByName("GTiff")
    # 创建tif文件
    output_dataset = driver.Create(path, width, height, channels,
                                   gdal_array.GDALTypeCodeToNumericTypeCode(data_type))
    # 将子图数据写入TIF文件
    '''
    注意写入顺序,不同的图像可能需要更改,此次处理的图像尺寸格式为C,H,W,顺序错误图像会无法正确显示
    问题:似乎切割出来的图像色相有一定的偏差
    代办:自动识别图像的尺寸写入格式
    '''
    for channel in range(channels):
        output_dataset.GetRasterBand(channel + 1).WriteArray(dataset[channel, :, :])
    # 关闭TIF文件
    output_dataset = None


def read_voc_xml(xml_file_path):
    """
    读取VOC格式的xml文件，提取标签和包围框信息。

    参数：
    xml_file_path：xml文件的路径。

    返回值：
    labels：包含所有标签的列表。
    bboxes：包含所有包围框坐标的列表，每个包围框是一个四元组(xmin, ymin, xmax, ymax)。
    """
    tree = ET.parse(xml_file_path)
    root = tree.getroot()

    labels = []
    bboxes = []

    for obj in root.findall('object'):
        label = obj.find('name').text
        bbox = obj.find('bndbox')
        xmin = int(bbox.find('xmin').text)
        ymin = int(bbox.find('ymin').text)
        xmax = int(bbox.find('xmax').text)
        ymax = int(bbox.find('ymax').text)
        bboxes.append((xmin, ymin, xmax, ymax))
        labels.append(label)

    return labels, bboxes


def tif_segmentation(read_file_root_path, read_file_name, tif_write_path='./output/segmented_tif/',
                     xml_write_path='./output/segmented_xml/',
                     mark_png_write_path='./output/markedPNG/', split_width=640, split_height=640):
    # 读取TIF文件
    arr_dataset, channels, height, width = read_tif(read_file_root_path + read_file_name)
    # 获取行列切片的数量
    row = math.ceil(height / split_height)
    col = math.ceil(width / split_width)

    # 获取未切割的文件名（去除扩展名部分）
    base_file_name = read_file_name[:-4]

    # 创建与未切割文件同名的文件夹（如果不存在）
    output_folder = os.path.join(tif_write_path, base_file_name)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    # 保存切割后的子图并显示进度
    total_images = row * col
    with tqdm(total=total_images) as pbar:  # 使用tqdm的ProgressBar创建进度条对象并设置总进度
        for r in range(row):
            for c in range(col):
                # 计算切割区域
                start_x = c * split_width
                end_x = start_x + split_width
                start_y = r * split_height
                end_y = start_y + split_height

                # 计算实际能切出来的子图的宽度和高度，如果超过原图尺寸则进行修正
                sub_width = split_width if end_x <= width else width - start_x
                sub_height = split_height if end_y <= height else height - start_y
                '''
                为了保证图像的相对位置不改变,切割子图时,不改变原图尺寸,
                将切割后不能满足指定长宽的子图补全,且应该只在图像的右边或者下面补上白边
                '''
                # 创建子图,CHW!!
                sub_image = np.zeros((channels, split_height, split_width), dtype=arr_dataset.dtype)  # 初始化为纯黑色的子图背景
                sub_image[:channels, :sub_height, :sub_width] = arr_dataset[:, start_y:start_y + sub_height, start_x:start_x + sub_width]  # 求交集
                # sub_image[:channels, :sub_height, :sub_width] = arr_dataset[:, 3000:3000+sub_height, 3000:3000 + sub_width] #测试用例

                # 生成子图文件路径,从000001开始
                file_number = r * col + c + 1
                sub_file_name = f"{base_file_name}_{file_number:06d}.tif"
                output_file = os.path.join(output_folder, sub_file_name)
                # 写入tif文件
                write_tif(output_file, sub_image, arr_dataset.dtype, channels, split_width, split_height)
                # 更新进度条，表示处理了一个图像
                pbar.update(1)


if __name__ == '__main__':
    # 指定TIF文件路径
    root_path = "data/"
    file_name = 'SET011.tif'
    tif_write_path = './output/segmented_tif/'
    xml_write_path = './output/segmented_xml/'
    mark_png_write_path = './output/markedPNG/'
    tif_segmentation(read_file_root_path=root_path, read_file_name=file_name, tif_write_path=tif_write_path,
                     xml_write_path=xml_write_path, mark_png_write_path=mark_png_write_path)
