import os.path

from PIL import Image
import numpy as np
import cv2 as cv

def read_jpg_file(file_path):
    # 打开图像
    image = Image.open(file_path)
    # 获取宽高
    width, height = image.size
    # 转换为numpy
    img_array = np.array(image)

    return width, height, img_array


def read_yolo_classes_file(file_path):
    """
    读取 YOLO 标签的 classes 文件，并返回一个包含类别的列表。
    :param file_path: YOLO classes 文件的路径。
    :return: YOLO classes 文件中的类别列表。
    """
    classes = []
    try:
        with open(file_path, 'r') as file:
            # yolo的classes.txt文件每行存储一个类别标签
            for line in file:
                # 去除开头和结尾的空白字符，并将类别添加到列表中
                classes.append(line.strip())
    except FileNotFoundError:
        print(f"错误：未找到文件 '{file_path}'。")
    except Exception as e:
        print(f"读取 '{file_path}' 时发生错误：{e}")

    return classes


def read_yolo_txt_file(file_path, classes, width, height):
    """
    读取yolo数据集的txt标签格式
    :param classes: 标签类别列表
    :param file_path: 读取的标签文件路径
    :param width: 标签对应的图像宽度
    :param height: 标签对应的图像高度
    :return: 列表,每个子元素列表对应值:[标签名,左上角x坐标,左上角y坐标,右下角x坐标,右下角y坐标].值为int形
    """
    yolo_datas = []  # txt文件中标签的列表
    with open(file_path, 'r') as f:
        # 一次读取一行
        for line in f.readlines():
            txt_line = line.strip().split(' ')
            label = int(float(txt_line[0].strip()))
            center_x = round(float(str(txt_line[1]).strip()) * width)
            center_y = round(float(str(txt_line[2]).strip()) * height)
            bbox_width = round(float(str(txt_line[3]).strip()) * width)
            bbox_height = round(float(str(txt_line[4]).strip()) * height)

            x_min = int(center_x - bbox_width / 2)
            y_min = int(center_y - bbox_height / 2)
            x_max = int(center_x + bbox_width / 2)
            y_max = int(center_y + bbox_height / 2)

            data_line = [classes[label], x_min, y_min, x_max, y_max]
            yolo_datas.append(data_line)
    return yolo_datas


def resize_rectangle(rect_top_left, rect_bottom_right, target_width, target_height):
    """
    将由两个平面坐标点表示的矩形,计算矩形中心,并根据中心重新设置矩形框的大小
    :param rect_top_left: 矩形左上角坐标
    :param rect_bottom_right: 矩形右下角坐标
    :param target_width: 矩形调整的目标宽度
    :param target_height: 矩形调整的目标高度
    :return:调整后的两个坐标点
    """
    center_x = (rect_bottom_right[0] + rect_top_left[0]) / 2
    center_y = (rect_bottom_right[1] + rect_top_left[1]) / 2
    # 根据缩放比例调整矩形区域的坐标
    new_rect_top_left = [int(center_x - target_width / 2), int(center_y - target_height / 2)]
    new_rect_bottom_right = [int(center_x + target_width / 2), int(center_y + target_height / 2)]

    # 返回缩放后的矩形区域坐标
    return new_rect_top_left, new_rect_bottom_right


if __name__ == '__main__':
    resized_width = 64
    resized_height = 64
    output_path = "./output/output.jpg"
    # 读取图像
    img_dir_path = r"D:\Documents\DataSet\ZDRH\罂粟标注非罂粟植物\images"
    img_name = "430528新宁县057.JPG"
    img_file_path = os.path.join(img_dir_path, img_name)
    img_width, img_height, image_array = read_jpg_file(img_file_path)
    # 计算图像对应标签路径
    label_dir_path = r"D:\Documents\DataSet\ZDRH\罂粟标注非罂粟植物\labels"
    label_name = "430528新宁县057.txt"
    label_file_path = os.path.join(label_dir_path, label_name)
    # 读取classes
    classes_path = os.path.join(label_dir_path, 'classes.txt')
    classes = read_yolo_classes_file(classes_path)
    yolo_label_datas = read_yolo_txt_file(label_file_path, classes, img_width, img_height)
    # 重设矩形大小并导出jpg
    for i in range(len(yolo_label_datas)):
        yolo_label_datas[i][1:3], yolo_label_datas[i][3:] = resize_rectangle(yolo_label_datas[i][1:3], yolo_label_datas[i][3:], resized_width,
                                                                             resized_height)
        temp_output = Image.fromarray(image_array[])

    print("end")
