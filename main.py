import math
import os
import xml.etree.ElementTree as ET
from xml.dom import minidom
from PIL import Image, ImageFont, ImageDraw
import numpy as np
from osgeo import gdal, gdal_array
from tqdm import tqdm

'''
对多通道的tif图像进行分割,
默认尺寸:640*640,边缘添加白边补整,
切割顺序:从左往右,从上往下
子图命名规范:原图名称_000001开始递增
'''


def normalize_array(array):
    """

    :param array: 待归一化的数组
    :return: 归一化后转换为8位的数组
    """
    # 检查数组是否包含无效值
    has_invalid_values = np.any(np.isnan(array)) or np.any(np.isinf(array))

    # 如果数组中有无效值，则将其替换为0
    if has_invalid_values:
        array = np.nan_to_num(array)

    # 计算数组的极差
    ptp = np.ptp(array)

    # 如果极差为0，则将数组全部设置为0，避免除以0的错误
    if ptp == 0:
        normalized_array = np.zeros_like(array)
    else:
        # 归一化数组
        normalized_array = (array - np.min(array)) / ptp * 255

    return normalized_array.astype(np.uint8)


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
    # 栅格数据的空间参考信息
    geo_trans = dataset.GetGeoTransform()
    # 投影信息
    geo_proj = dataset.GetProjection()
    # 将tif文件读取为ndarray数组,数据的格式(通道,高度,宽度)
    arr_dataset = dataset.ReadAsArray()  # 转为numpy格式
    del dataset
    return arr_dataset, channels, height, width, geo_trans, geo_proj


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


def write_voc_xml(xml_file_path, labels, bboxes):
    """
    将标签和包围框信息保存为VOC格式的xml文件。

    参数：
    xml_file_path：xml文件的保存路径。
    labels：包含所有标签的列表。
    bboxes：包含所有包围框坐标的列表，每个包围框是一个四元组(xmin, ymin, xmax, ymax)。
    """
    # 创建XML根元素
    root = ET.Element("annotation")

    for label, bbox in zip(labels, bboxes):
        obj = ET.SubElement(root, "object")
        name = ET.SubElement(obj, "name")
        name.text = label
        bndbox = ET.SubElement(obj, "bndbox")
        xmin, ymin, xmax, ymax = bbox
        ET.SubElement(bndbox, "xmin").text = str(xmin)
        ET.SubElement(bndbox, "ymin").text = str(ymin)
        ET.SubElement(bndbox, "xmax").text = str(xmax)
        ET.SubElement(bndbox, "ymax").text = str(ymax)

    # 将XML文件格式化
    xml_string = ET.tostring(root, encoding="utf-8")
    dom = minidom.parseString(xml_string)
    pretty_xml_string = dom.toprettyxml(indent="    ")

    # 将格式化后的XML字符串保存为文件
    with open(xml_file_path, "w", encoding="utf-8") as f:
        f.write(pretty_xml_string)


def convert_tif_array_to_png(tif_array):
    """
    读取CHW类型的tif数据,后去其前三个通道转换为png文件的格式后返回
    :param tif_array:tif图像的数组形数据
    :return:png格式数据
    """
    # 取tif前3通道BGR
    png_array = tif_array[:3, :, :]
    # 值映射到255,uint8
    png_array = normalize_array(png_array)
    # 变换为RGB通道
    png_array[[0, 2], :, :] = png_array[[2, 0], :, :]
    return png_array


def draw_bboxes_on_image(png_image, labels, bboxes):
    """
    在图像上绘制包围框和标签。

    参数：
    png_image：要绘制的图像，PIL格式的图像对象。
    labels：标签列表。
    bboxes：包围框坐标列表，每个包围框是一个四元组(xmin, ymin, xmax, ymax)。

    返回值：
    绘制了包围框和标签的图像，PIL格式的图像对象。
    """
    # 创建绘图对象
    draw = ImageDraw.Draw(png_image)

    # 设置字体和字体大小
    font = ImageFont.truetype("arial.ttf", 12)

    for label, bbox in zip(labels, bboxes):
        line_width = 3
        # 提取包围框坐标
        xmin, ymin, xmax, ymax = bbox
        # 计算标签文本的包围框
        label_bbox = draw.textbbox((xmin, ymin), label, font=font)
        # 获取包围框的左下角坐标
        label_x = label_bbox[0]
        label_y = label_bbox[1] - font.size - line_width / 2

        # 绘制矩形框
        draw.rectangle([xmin, ymin, xmax, ymax], outline="red", width=line_width)

        # 绘制标签文本,绘制在标签左上角
        draw.text((label_x, label_y), label, fill="red", font=font)

    return png_image


def tif_segmentation(read_file_root_path, tif_write_path='./output/segmented_tif/',
                     xml_write_path='./output/segmented_xml/',
                     mark_png_write_path='./output/markedPNG/', split_width=640, split_height=640,black_threshold = 1):
    """

    :param read_file_root_path:
    :param tif_write_path:
    :param xml_write_path:
    :param mark_png_write_path:
    :param split_width:
    :param split_height:
    :param black_threshold:
    :return:
    """
    tif_files = [f for f in os.listdir(read_file_root_path) if f.endswith('.tif')]

    with tqdm(total=len(tif_files), desc="Processing Images") as pbar_folder:
        for tif_file in tif_files:
            # 读取TIF文件
            arr_dataset, channels, height, width, geo_trans, geo_proj = read_tif(read_file_root_path + tif_file)
            # 获取行列切片的数量
            row = math.ceil(height / split_height)
            col = math.ceil(width / split_width)

            # 获取未切割的文件名（去除扩展名部分）
            base_tif_file_name = tif_file[:-4]

            # 读取对应的XML文件
            xml_file_path = os.path.join(read_file_root_path, base_tif_file_name + ".xml")
            labels, bboxes = read_voc_xml(xml_file_path)

            # 保存切割后的子图并显示进度
            total_images = row * col
            with tqdm(total=total_images, position=0) as pbar:  # 使用tqdm的ProgressBar创建进度条对象并设置总进度
                for r in range(row):
                    for c in range(col):
                        # 子图编号
                        file_number = r * col + c + 1
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
                        将切割后不能满足指定长宽的子图补全,且应该只在图像的右边或者下面补上黑边
                        '''
                        # 创建子图,CHW!!
                        sub_image = np.zeros((channels, split_height, split_width), dtype=arr_dataset.dtype)  # 初始化为纯黑色的子图背景
                        sub_image[:, :sub_height, :sub_width] = arr_dataset[:, start_y:start_y + sub_height,
                                                                start_x:start_x + sub_width]  # 求交集

                        # 新建一个列表，用于存储与子图有重叠区域的标签和包围框
                        sub_labels = []
                        sub_bboxes = []
                        # 更新XML文件中的包围框坐标（相对于子图的坐标）
                        for i in range(len(bboxes)):
                            xmin, ymin, xmax, ymax = bboxes[i]
                            # 计算包围框相对于子图的坐标
                            relative_xmin = max(0, xmin - start_x)
                            relative_ymin = max(0, ymin - start_y)
                            relative_xmax = min(sub_width, xmax - start_x)
                            relative_ymax = min(sub_height, ymax - start_y)
                            # 检查是否有重叠区域
                            if relative_xmin < sub_width and relative_ymin < sub_height and relative_xmax > 0 and relative_ymax > 0:
                                # 将有重叠区域的标签和包围框添加到列表中
                                sub_labels.append(labels[i])
                                sub_bboxes.append((relative_xmin, relative_ymin, relative_xmax, relative_ymax))

                        # 计算5个通道全为黑色的像素个数
                        black_pixel_num = len(sub_image[sub_image == 0])
                        if len(sub_labels) != 0 or black_pixel_num/sub_image.size < black_threshold:
                            # 生成子图文件路径,从000001开始
                            # 创建与未切割文件同名的文件夹（如果不存在）
                            tif_output_folder_path = os.path.join(tif_write_path, base_tif_file_name)
                            if not os.path.exists(tif_output_folder_path):
                                os.makedirs(tif_output_folder_path)

                            sub_tif_file_name = f"{base_tif_file_name}_{file_number:06d}.tif"
                            sub_tif_output_path = os.path.join(tif_output_folder_path, sub_tif_file_name)
                            # 写入tif文件
                            write_tif(sub_tif_output_path, sub_image, arr_dataset.dtype, channels, split_width, split_height)

                            # tif转png格式
                            sub_png_arr = convert_tif_array_to_png(sub_image)
                            # 将数组转换为PIL图像 CHW->WHC
                            sub_png = Image.fromarray(sub_png_arr.transpose(1, 2, 0))
                            sub_png = draw_bboxes_on_image(sub_png, sub_labels, sub_bboxes)
                            # 保存png图像
                            png_output_folder_path = os.path.join(mark_png_write_path, base_tif_file_name)
                            if not os.path.exists(png_output_folder_path):
                                os.makedirs(png_output_folder_path)
                            sub_png_file_name = f"{base_tif_file_name}_{file_number:06d}.png"
                            sub_png_output_path = os.path.join(png_output_folder_path, sub_png_file_name)
                            sub_png.save(sub_png_output_path)

                            # 保存更新后的XML文件
                            xml_output_folder_path = os.path.join(xml_write_path, base_tif_file_name)
                            if not os.path.exists(xml_output_folder_path):
                                os.makedirs(xml_output_folder_path)
                            sub_xml_output_path = os.path.join(xml_output_folder_path, f"{base_tif_file_name}_{file_number:06d}.xml")
                            write_voc_xml(sub_xml_output_path, sub_labels, sub_bboxes)

                        # 更新每张图像的进度条，表示处理了一个图像
                        pbar.update(1)
            # 更新文件夹剩余图像数量的进度条
            pbar_folder.update(1)


if __name__ == '__main__':
    # 指定TIF文件路径
    root_path = "data/"
    tif_write_path = './output/segmented_tif/'
    xml_write_path = './output/segmented_xml/'
    mark_png_write_path = './output/markedPNG/'
    split_width = 640
    split_height = 640
    tif_segmentation(read_file_root_path=root_path, tif_write_path=tif_write_path,
                     xml_write_path=xml_write_path, mark_png_write_path=mark_png_write_path,
                     split_width=split_width, split_height=split_height,black_threshold = 0.1)
