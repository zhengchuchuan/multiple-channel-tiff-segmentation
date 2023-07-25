# 多通道高光谱图像tiff文件分割
## 功能介绍
1. tif文件和xml文件放入data目录下
2. 运行程序后会对tif文件和xml文件进行拆分输出文件夹如下：
- ./output/markedPNG/ 存储切割后带标记的子图(没打标签)
- ./output/segmented_tif/ 存储切割后的tif子图像,相同图像切割出的tif子图保存在同一个文件夹
- ./output/segmented_xml/ 存储切割后的tif子图的xml信息,相同图像切割出的xml保存在同一个文件夹
 