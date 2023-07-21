import numpy as np
import math
import time
from tqdm import tqdm
from osgeo import gdal, gdal_array
from PIL import Image

'''
对多通道的tif图像进行分割,
默认尺寸:640*640,边缘添加白边补整,
切割顺序:从左往右,从上往下
子图命名规范:原图名称_000001开始递增
'''


def tif_segmentation(read_file_root_path, read_file_name, tif_write_path='./output/segmented_tif/',
                     xml_write_path='./output/segmented_xml/',
                     mark_png_write_path='./output/markedPNG/', split_width=640, split_height=640):
    # 打开TIF文件
    try:
        dataset = gdal.Open(read_file_root_path + read_file_name, gdal.GA_Update)
        if dataset is None:
            raise IOError("无法打开TIF文件，请检查文件路径。")
    except IOError as e:  # IO异常
        print(str(e))
    except Exception as e:  # 其他异常
        print("其他异常：", str(e))

    # 获取TIF文件的通道数
    num_channels = dataset.RasterCount
    # 获取TIF文件的宽度和高度
    width = dataset.RasterXSize
    height = dataset.RasterYSize
    # 获取行列切片的数量
    row = math.ceil(height / split_height)
    col = math.ceil(width / split_width)

    # 将tif文件读取为ndarray数组
    # 数据的格式(通道,高度,宽度)
    arr_dataset = dataset.ReadAsArray()

    '''
    保存tif测试
    导出图像尺寸正常,但全黑,查看发现值全为0
    '''
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

                # 计算实际子图的宽度和高度，如果超过原图尺寸则进行修正
                sub_width = split_width if end_x <= width else width - start_x
                sub_height = split_height if end_y <= height else height - start_y
                '''
                为了保证图像的相对位置不改变,切割子图时,不改变原图尺寸,
                将切割后不能满足指定长宽的子图补全,且应该只在图像的右边或者下面补上白边
                '''
                # 创建临时子图
                sub_image_temp = np.zeros((num_channels, split_height, split_width),
                                          dtype=arr_dataset.dtype)    # 初始化为白色
                sub_image_temp[:num_channels, :sub_height, :sub_width] = arr_dataset[:, start_y:end_y, start_x:end_x]

                # 创建子图
                sub_image = np.zeros((num_channels, split_height, split_width), dtype=arr_dataset.dtype) # 初始化为白色
                sub_image[:num_channels, :sub_height, :sub_width] = sub_image_temp[:, :sub_height, :sub_width]

                # 生成文件名
                file_number = r * col + c + 1
                file_name = read_file_name[:-4] + f"_{file_number:06d}.tif"

                # 创建子图的输出文件
                output_file = tif_write_path + file_name
                driver = gdal.GetDriverByName("GTiff")
                output_dataset = driver.Create(output_file, split_width, split_height, num_channels,
                                               gdal_array.GDALTypeCodeToNumericTypeCode(arr_dataset.dtype))

                # 将子图数据写入TIF文件
                if num_channels == 1:
                    output_dataset.GetRasterBand(1).WriteArray(sub_image[:, :, 0])
                else:
                    for channel in range(num_channels):
                        output_dataset.GetRasterBand(channel + 1).WriteArray(sub_image[:, :, channel])

                # 关闭TIF文件
                output_dataset = None

                pbar.update(1)  # 更新进度条，表示处理了一个图像
    # 关闭原始TIF文件
    dataset = None


if __name__ == '__main__':
    # 指定TIF文件路径
    root_path = "data/"
    file_name = 'SET011.tif'
    tif_write_path = './output/segmented_tif/'
    xml_write_path = './output/segmented_xml/'
    mark_png_write_path = './output/markedPNG/'
    tif_segmentation(read_file_root_path=root_path, read_file_name=file_name, tif_write_path=tif_write_path,
                     xml_write_path=xml_write_path, mark_png_write_path=mark_png_write_path)
