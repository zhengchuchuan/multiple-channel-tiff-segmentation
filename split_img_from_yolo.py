import os.path

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

"""
计算原始标签中点,将标签宽度以中点为中心重设为指定大小,
并将标签范围内的png图像切割导出到指定文件夹
"""


def get_dir_files_name(floder_path, file_type=None):
    # 获取文件夹下指定结尾名的所有文件名
    """

    :param floder_path: 文件夹路径
    :param file_type: 查找的文件后缀名
    :return: 文件名列表
    """
    if file_type is not None:
        files_name = [f for f in os.listdir(floder_path) if f.lower().endswith(file_type.lower())]
    # 获取所有的文件
    else:
        files_name = [f for f in os.listdir(floder_path)]
    return files_name


def pil_read_jpg(path):
    # 打开图像
    image = Image.open(path)
    # 获取宽高
    width, height = image.size
    # 转换为numpy
    img_array = np.array(image)

    return height, width, img_array


def cv2_img_read(path):
    # dtype对于tif图像需要切换成uint16

    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), -1)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    height = image.shape[0]
    width = image.shape[1]
    return height, width, image


def cv2_img_write(path, img):
    # 解析文件名
    file_name = os.path.basename(path)
    file_extension = os.path.splitext(file_name)[1]
    # 创建目录(如果不存在)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # RGB->BGR
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imencode(file_extension, img)[1].tofile(path)


def read_yolo_classes_file(path):
    """
    读取 YOLO 标签的 classes 文件，并返回一个包含类别的列表。
    :param path: YOLO classes 文件的路径。
    :return: YOLO classes 文件中的类别列表。
    """
    classes = []
    try:
        with open(path, 'r') as file:
            # yolo的classes.txt文件每行存储一个类别标签
            for line in file:
                # 去除开头和结尾的空白字符，并将类别添加到列表中
                classes.append(line.strip())
    except FileNotFoundError:
        print(f"错误：未找到文件 '{path}'。")
    except Exception as e:
        print(f"读取 '{path}' 时发生错误：{e}")

    return classes


def read_yolo_txt_file(path, classes, width, height):
    """
    读取yolo数据集的txt标签格式
    :param classes: 标签类别列表
    :param path: 读取的标签文件路径
    :param width: 标签对应的图像宽度
    :param height: 标签对应的图像高度
    :return: 列表,每个子元素列表对应值:[标签名,左上角x坐标,左上角y坐标,右下角x坐标,右下角y坐标].值为int形
    """
    yolo_datas = []  # txt文件中标签的列表
    with open(path, 'r') as f:
        # 一次读取一行
        for line in f.readlines():
            txt_line = line.strip().split(' ')
            label_index = int(float(txt_line[0].strip()))
            center_x = round(float(str(txt_line[1]).strip()) * width)
            center_y = round(float(str(txt_line[2]).strip()) * height)
            bbox_width = round(float(str(txt_line[3]).strip()) * width)
            bbox_height = round(float(str(txt_line[4]).strip()) * height)

            x_min = int(center_x - bbox_width / 2)
            y_min = int(center_y - bbox_height / 2)
            x_max = int(center_x + bbox_width / 2)
            y_max = int(center_y + bbox_height / 2)

            # 返回标签名,左上角坐标,右下角坐标
            data_line = [classes[label_index], x_min, y_min, x_max, y_max]
            yolo_datas.append(data_line)
    return yolo_datas


def binary_to_list(bin_str, categories):
    """
    二进制转换为对应类别的列表,1表示有,0表示无
    :param binary_string: 二进制数
    :param categories: 类别列表
    :return: 二进制数对应的种类列表
    """
    result = []

    for i, bit in enumerate(bin_str):
        if bit == '1':
            result.append(categories[i])

    return result


def save_list_to_txt(file_path, content_list):
    """
    将列表中的内容写入txt文件,每个元素占一行
    :param file_path:
    :param content_list:
    :return:
    """
    # 创建目录（如果不存在）
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    # 文件写入
    with open(file_path, 'w') as f:
        for item in content_list:
            f.write(str(item) + '\n')


def resize_rectangle(rect_top_left, rect_bottom_right, target_width, target_height):
    """
    将由两个平面坐标点表示的矩形,计算矩形中心,并根据中心重新设置矩形框的大小
    :param rect_top_left: 矩形左上角坐标
    :param rect_bottom_right: 矩形右下角坐标
    :param target_width: 矩形调整的目标宽度
    :param target_height: 矩形调整的目标高度
    :return:调整后的两个坐标点
    """
    # 计算矩形中心点
    center_x = (rect_bottom_right[0] + rect_top_left[0]) / 2
    center_y = (rect_bottom_right[1] + rect_top_left[1]) / 2
    # 从中心点计算新矩形范围的两个坐标点
    new_rect_top_left = [int(center_x - target_width / 2), int(center_y - target_height / 2)]
    new_rect_bottom_right = [int(center_x + target_width / 2), int(center_y + target_height / 2)]

    # 返回缩放后的矩形区域坐标
    return new_rect_top_left, new_rect_bottom_right


if __name__ == '__main__':
    # 新的矩形尺寸
    resized_width = 224
    resized_height = 224
    # 图像文件夹路径
    img_floder_path = r"D:\Documents\DataSet\ZDRH\poppy_top5_more\img"
    # 标签文件夹路径
    label_floder_path = r"D:\Documents\DataSet\ZDRH\poppy_top5_more\labels"
    # 输出文件路径
    split_img_output_folder = r"D:\Documents\DataSet\ZDRH\poppy_top5_more\split_img"
    label_output_path = r"D:\Documents\DataSet\ZDRH\poppy_top5_more\split_img_labels"
    # 列表中的类别
    categories = ['poppy', 'soil', 'tree', 'grass', 'struct', 'others']

    # 读取classes
    classes_path = os.path.join(label_floder_path, 'classes.txt')
    classes = read_yolo_classes_file(classes_path)
    # 获取文件夹下所有jpg文件
    img_files_arr = get_dir_files_name(img_floder_path, '.jpg')

    with tqdm(total=len(img_files_arr)) as pbar:
        for file_name in img_files_arr:
            img_file_path = os.path.join(img_floder_path, file_name)
            # img_width, img_height, image_array = pil_read_jpg(img_file_path)
            img_height, img_width, img_array = cv2_img_read(img_file_path)
            # 计算图像对应标签路径
            label_name = file_name[:-4] + '.txt'
            label_file_path = os.path.join(label_floder_path, label_name)
            # 读取标签数据
            yolo_label_datas = read_yolo_txt_file(label_file_path, classes, img_width, img_height)
            # 导出labels
            for i, data_line in enumerate(yolo_label_datas):
                # 文件保存路径
                sub_label_name = file_name[:-4] + '_' + str(i) + '.txt'
                sub_image_name = file_name[:-4] + '_' + str(i) + '.jpg'
                sub_label_path = os.path.join(label_output_path, sub_label_name)
                sub_image_path = os.path.join(split_img_output_folder, sub_image_name)

                # 重设标签大小
                yolo_label_datas[i][1:3], yolo_label_datas[i][3:] = resize_rectangle(yolo_label_datas[i][1:3], yolo_label_datas[i][3:], resized_width,
                                                                                     resized_height)
                # 裁切后图像范围
                split_img_array = img_array[yolo_label_datas[i][2]:yolo_label_datas[i][4], yolo_label_datas[i][1]:yolo_label_datas[i][3], :]
                # 保存图像
                # Image.fromarray(split_img_array)  # PIL方法
                cv2_img_write(sub_image_path, split_img_array)  # OpenCV方法

                # 二进制转换为实际的标签
                sub_classes = binary_to_list(bin_str=data_line[0], categories=categories)
                # 保存txt文件
                save_list_to_txt(sub_label_path, sub_classes)
            pbar.update(1)

    print("end")
