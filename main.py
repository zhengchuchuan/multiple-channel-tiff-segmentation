import math
import os
import xml.etree.ElementTree as ET
from xml.dom import minidom
from PIL import Image, ImageFont, ImageDraw
import numpy as np
from osgeo import gdal, gdal_array
from tqdm import tqdm

base_size = 640
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


def calculate_split_box(row, col, step, width, height):
    start_x = col * step
    end_x = start_x + width
    start_y = row * step
    end_y = start_y + height
    return start_x, start_y, end_x, end_y


def generate_file_save_path(file_write_path, file_name, file_number, file_type):
    output_folder_path = os.path.join(file_write_path, file_name)
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)
    file_name = f"{file_name}_{file_number}{file_type}"
    file_output_path = os.path.join(output_folder_path, file_name)
    return file_output_path


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


def write_tif(tif_write_path, base_tif_file_name, file_number, dataset, channels, width, height):
    """
    写入tif文件,
    参数:导出子图的路径,子图的数组,保存文件的类型,通道数,拆分的图像宽度,拆分的图像高度
    """
    # 生成子图文件路径,从000001开始
    # 创建与未切割文件同名的文件夹（如果不存在）
    tif_output_folder_path = os.path.join(tif_write_path, base_tif_file_name)
    if not os.path.exists(tif_output_folder_path):
        os.makedirs(tif_output_folder_path)
    # 文件名
    sub_tif_file_name = f"{base_tif_file_name}_{file_number}.tif"
    # 保存路径
    sub_tif_output_path = os.path.join(tif_output_folder_path, sub_tif_file_name)
    driver = gdal.GetDriverByName("GTiff")
    # 创建tif文件
    output_dataset = driver.Create(sub_tif_output_path, width, height, channels,
                                   gdal_array.GDALTypeCodeToNumericTypeCode(dataset.dtype))
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


def write_voc_xml(xml_write_path, base_tif_file_name, file_number, labels, bboxes, image_width, image_height, image_depth):
    """
    将标签和包围框信息保存为VOC格式的xml文件。

    参数：
    xml_file_path：xml文件的保存路径。
    labels：包含所有标签的列表。
    bboxes：包含所有包围框坐标的列表，每个包围框是一个四元组(xmin, ymin, xmax, ymax)。
    image_width: 图像宽度
    image_height: 图像高度
    image_depth: 图像通道数
    """
    sub_xml_output_path = generate_file_save_path(xml_write_path, base_tif_file_name, file_number, '.xml')
    # xml_output_folder_path = os.path.join(xml_write_path, base_tif_file_name)
    # if not os.path.exists(xml_output_folder_path):
    #     os.makedirs(xml_output_folder_path)
    # sub_xml_output_path = os.path.join(xml_output_folder_path, f"{base_tif_file_name}_{file_number:06d}.xml")
    # 创建XML根元素
    root = ET.Element("annotation")
    # 添加图像信息
    folder = ET.SubElement(root, "folder")
    folder.text = "VOC2007"  # 替换为您的图像所在的文件夹名称
    filename = ET.SubElement(root, "filename")
    filename.text = f"{base_tif_file_name}_{file_number}.png"  # 替换为您的图像文件名和扩展名
    size = ET.SubElement(root, "size")
    width = ET.SubElement(size, "width")
    width.text = str(image_width)
    height = ET.SubElement(size, "height")
    height.text = str(image_height)
    depth = ET.SubElement(size, "depth")
    depth.text = str(image_depth)

    for label, bbox in zip(labels, bboxes):
        obj = ET.SubElement(root, "object")
        name = ET.SubElement(obj, "name")
        name.text = label
        pose = ET.SubElement(obj, "pose")
        pose.text = "Unspecified"
        truncated = ET.SubElement(obj, "truncated")
        truncated.text = "0"
        difficult = ET.SubElement(obj, "difficult")
        difficult.text = "0"
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
    with open(sub_xml_output_path, "w", encoding="utf-8") as f:
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


def write_png(file_write_path, file_name, file_number, file_type, img, labels, bboxes):
    # tif转png格式
    sub_png_arr = convert_tif_array_to_png(img)
    # 将数组转换为PIL图像 CHW->WHC
    sub_png = Image.fromarray(sub_png_arr.transpose(1, 2, 0))
    sub_png = draw_bboxes_on_image(sub_png, labels, bboxes)
    # 保存png图像
    sub_png_output_path = generate_file_save_path(file_write_path=file_write_path, file_name=file_name,
                                                  file_number=file_number, file_type=file_type)
    sub_png.save(sub_png_output_path)


def update_sub_xml(width, height, labels, bboxes, start_x, start_y):
    sub_labels = []
    sub_bboxes = []
    # 更新XML文件中的包围框坐标（相对于子图的坐标）
    for i in range(len(bboxes)):
        xmin, ymin, xmax, ymax = bboxes[i]
        # 计算包围框相对于子图的坐标
        relative_xmin = max(0, xmin - start_x)
        relative_ymin = max(0, ymin - start_y)
        relative_xmax = min(width, xmax - start_x)
        relative_ymax = min(height, ymax - start_y)
        # 检查是否有重叠区域
        if relative_xmin < width and relative_ymin < height and relative_xmax > 0 and relative_ymax > 0:
            # 将有重叠区域的标签和包围框添加到列表中
            sub_labels.append(labels[i])
            sub_bboxes.append((relative_xmin, relative_ymin, relative_xmax, relative_ymax))
    return sub_labels, sub_bboxes


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


# def save_files(base_tif_file_name,tif_write_path,xml_write_path,mark_png_write_path):

def tif_segmentation(read_file_root_path, tif_write_path='./output/segmented_tif/',
                     xml_write_path='./output/segmented_xml/',
                     mark_png_write_path='./output/markedPNG/', split_width=640, split_height=640, step=0, black_threshold=1):
    """

    :param step:
    :param read_file_root_path:
    :param tif_write_path:
    :param xml_write_path:
    :param mark_png_write_path:
    :param split_width:
    :param split_height:
    :param black_threshold:
    :return:
    """
    # 获取数据存储的文件下所有的tif文件
    tif_files = [f for f in os.listdir(read_file_root_path) if f.endswith('.tif')]
    # 文件处理个数进度条
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

            # 切割子图,显示子图处理进度
            total_images = row * col
            with tqdm(total=total_images, position=0) as pbar:  # 使用tqdm的ProgressBar创建进度条对象并设置总进度
                for r in range(row):
                    for c in range(col):
                        # 子图编号
                        file_number = str(r * col + c + 1).zfill(6)
                        # 计算切割区域
                        start_x, start_y, end_x, end_y = calculate_split_box(r, c, base_size, split_width, split_height)

                        # 计算实际能切出来的子图的宽度和高度,后续可能会在切出的图像再进行移位切分
                        temp_width = split_width if end_x <= width else width - start_x
                        temp_height = split_height if end_y <= height else height - start_y
                        '''
                        为了保证图像的相对位置不改变,切割子图时,不改变原图尺寸,
                        将切割后不能满足指定长宽的子图补全,且应该只在图像的右边或者下面补上黑边
                        '''
                        # 创建子图,CHW!!
                        temp_image = np.zeros((channels, split_height, split_width), dtype=arr_dataset.dtype)  # 初始化为纯黑色的子图背景
                        temp_image[:, :temp_height, :temp_width] = arr_dataset[:, start_y:start_y + temp_height,
                                                                   start_x:start_x + temp_width]  # 图像覆盖在黑色背景上
                        # 更新XML文件中的包围框坐标（相对于子图的坐标）
                        temp_labels, temp_bboxes = update_sub_xml(temp_width, temp_height, labels, bboxes, start_x, start_y)
                        # 计算5个通道全为黑色的像素个数
                        # 此处有点一刀切,直接对初始切割出来的子图进行了筛选,如果对子图再按步长进行切割时进行筛选,则会保留一部分边缘的无标签图片
                        black_pixel_num = len(temp_image[temp_image == 0])
                        if len(temp_labels) != 0 or (black_pixel_num / temp_image.size) < black_threshold:
                            if step == 0:
                                sub_image = temp_image
                                sub_width = temp_width
                                sub_height = temp_height
                                sub_labels = temp_labels
                                sub_bboxes = temp_bboxes
                                # 写入tif文件
                                write_tif(tif_write_path, base_tif_file_name, file_number, sub_image, channels, sub_width, sub_height)
                                # 保存png
                                write_png(mark_png_write_path, base_tif_file_name, file_number, '.png', sub_image, sub_labels, sub_bboxes)
                                # 如果拆分出的子图有标签,保存更新后的XML文件
                                if len(sub_labels) != 0:
                                    write_voc_xml(xml_write_path, base_tif_file_name, file_number, sub_labels, sub_bboxes, sub_width, sub_height, 3)
                            # 指定了子图切割的步长
                            elif step > 0:
                                # 暂时不处理不能整除的情况!!!!!!!
                                sub_row = math.ceil((temp_height - base_size) / step)
                                sub_col = math.ceil((temp_width - base_size) / step)
                                if sub_col==0: sub_col+=1
                                sub_width = base_size
                                sub_height = base_size

                                for sub_r in range(sub_row):
                                    for sub_c in range(sub_col):

                                        sub_file_number = str(sub_r * sub_col + sub_c + 1).zfill(3)
                                        sub_start_x, sub_start_y, sub_end_x, sub_end_y = calculate_split_box(sub_r, sub_c, step, sub_width, sub_height)
                                        sub_labels, sub_bboxes = update_sub_xml(sub_width, sub_height, temp_labels, temp_bboxes, sub_start_x ,sub_start_y)
                                        sub_image = temp_image[:, sub_start_y:sub_start_y + sub_height,
                                                    sub_start_x:sub_start_x + sub_width]
                                        sub_file_number = file_number + '_' + sub_file_number
                                        sub_black_pixel_num = len(sub_image[sub_image == 0])
                                        if len(sub_labels) != 0 or (sub_black_pixel_num / sub_image.size) < black_threshold:
                                            # 写入tif文件
                                            write_tif(tif_write_path, base_tif_file_name, sub_file_number,
                                                      sub_image, channels, sub_width, sub_height)
                                            # 保存png
                                            write_png(mark_png_write_path, base_tif_file_name, sub_file_number, '.png',
                                                      sub_image, sub_labels, sub_bboxes)
                                            # 如果拆分出的子图有标签,保存更新后的XML文件
                                            if len(sub_labels) != 0:
                                                write_voc_xml(xml_write_path, base_tif_file_name, sub_file_number,
                                                              sub_labels, sub_bboxes, sub_width, sub_height, 3)
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
    split_height = 1280
    tif_segmentation(read_file_root_path=root_path, tif_write_path=tif_write_path,
                     xml_write_path=xml_write_path, mark_png_write_path=mark_png_write_path,
                     split_width=split_width, split_height=split_height, step=32, black_threshold=0.1)
    print("程序结束")
