import cv2
import os
import xml.etree.ElementTree as ET


def crop_and_filter_objects(img_path, xml_path):

    filter_names = ["电流表和电压表指针(pointer)_a", "电流表最小刻度(ammeter_min_scale)"]
    object_name = "电流表(ammeter)"
    tail = img_path[-12:-4]
    output_img_path = "./data-end2end-triple/crop_images/" + tail + '.jpg'
    output_xml_path = "./data-end2end-triple/crop_Annotation" + tail + '.xml'

    img = cv2.imread(img_path)
    tree = ET.parse(xml_path)
    root = tree.getroot()

    for obj in root.findall('object'):
        name = obj.find('name').text
        if name == object_name:
            # 获取bounding box坐标
            bndbox = obj.find('bndbox')
            xmin = int(bndbox.find('xmin').text)
            ymin = int(bndbox.find('ymin').text)
            xmax = int(bndbox.find('xmax').text)
            ymax = int(bndbox.find('ymax').text)
            # 裁剪
            object_img = img[ymin:ymax, xmin:xmax]
            # 更新bounding box坐标
            bndbox.find('xmin').text = str(0)
            bndbox.find('ymin').text = str(0)
            bndbox.find('xmax').text = str(xmax - xmin)
            bndbox.find('ymax').text = str(ymax - ymin)

    for obj in root.findall('object'):
        name = obj.find('name').text
        if name not in filter_names:
            root.remove(obj)
        # 保存新的图片和XML文件
    cv2.imwrite(output_img_path, object_img)
    tree.write(output_xml_path)




