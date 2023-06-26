# -*- coding: utf-8 -*-
 
import os
import random
import cv2 as cv
import matplotlib.pyplot as plt
 
 
 
 
# labels = ["M6_bolt", "M8_bolt", "M10_bolt", "hexagon_bolt", "cross_bolt", "star_bolt"]
# color_list = [(0, 0, 255), (255, 0, 0), (0, 255, 0), (0, 255, 255), (255, 255, 0), (255, 255, 255)]
labels = ["bolt"]
color_list = [(0, 0, 255)]
img_dir = "/home/ur/Desktop/results/YOLO_Output/data/img"
yolo_txt_dir = "/home/ur/Desktop/results/YOLO_Output/data/txt"
save_dir="/home/ur/Desktop/results/YOLO_Output/data/test/img_%s.jpg"
# result_dst_dir = "/home/youyheng/DJIdata/robomaster_Final_Tournament/check_label_result"
scale_percent = 200
# rates that represent the imgs of all datasets
# 1 for all imgs, 0.5 for half of the imgs
check_rate = 1
random_check = False
 
def cv_imread(file_path):
    img = plt.imread(file_path)
    img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    return img_rgb
 
 
def my_line(img, start, end):
    thickness = 2
    line_type = 8
    cv.line(img,
             start,
             end,
             (0, 0, 0),
             thickness,
             line_type)
 
 
# draw rectangle with the data caught in the data file
# And set the name of the label to it
def draw_label_rec(img, label_index, label_info_list, img_name):
    global labels
 
    img_height = img.shape[0]
    img_width = img.shape[1]
 
    x = float(label_info_list[0])
    y = float(label_info_list[1])
    w = float(label_info_list[2])
    h = float(label_info_list[3])
 
    x_center = x * img_width
    y_center = y * img_height
 
    xmax = int(x_center + w * img_width / 2)
    xmin = int(x_center - w * img_width / 2)
    ymax = int(y_center + w * img_height / 2)
    ymin = int(y_center - w * img_height / 2)
 
    # Set font
    font = cv.FONT_HERSHEY_SIMPLEX
    global color_list
    
    # draw_rectangle
    cv.rectangle(img,  # img to paint on
             (xmin, ymin),  # bottom top
             (xmax, ymax),  # bottom right
             color_list[int(label_index)],  # bgr color
             2)  # line thickness
 
    ###########need perfection
    cv.putText(img, str(img_name), (5, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
 
 
def main():
    global img_dir, yolo_txt_dir, labels, random_check
 
    origin_window = "Origin Window"
 
    # Load all imgs with label info
    img_name_list = os.listdir(img_dir)
    print(len(img_name_list))
    if random_check is True:
        random.shuffle(img_name_list)
 
    check_max_times = int(check_rate * len(img_name_list))
    for index, img_name in enumerate(img_name_list):
        if not img_name.endswith('png'):
            continue
 
        # Checked for max_times and quit
        if index >= check_max_times:
            return
        print("**check img : {0} **".format(os.path.join(img_dir, img_name)))
        # Open IMG
        src_image = cv_imread(os.path.join(img_dir, img_name))
 
        # Open yolo label txt
        if os.path.exists(os.path.join(yolo_txt_dir, img_name.rpartition(".")[0]+".txt")):
            file_reader = open(os.path.join(yolo_txt_dir, img_name.rpartition(".")[0]+".txt"), "r")
        else:
            continue
 
        ## Dada loaded ##
        if src_image is None:
            print("Open image Error")
            return
 
        if file_reader is None:
            print("Open txt error")
            return
 
        # Pre-handling for Img
        src_height = src_image.shape[0]
        src_width = src_image.shape[1]
 
        # percent of original size
        global scale_percent
        width = int(src_width * scale_percent / 100)
        height = int(src_height * scale_percent / 100)
        dim = (width, height)
 
        # Decode the data
        while True:
            line = file_reader.readline()
            if not line:
                break
            label_info_list = line.split()
            print(label_info_list)
            # Get 5 nums in labeled_obj_info_list:
            # labels[label_info_list[0]] obj type : 0 ArmorBlue, 1 ArmorRed, 2 Base, 3 Watcher
            # label_info_list[1] x
            # label_info_list[2] y
            # label_info_list[3] w
            # label_info_list[4] h
            label_index = int(label_info_list[0])
            print(label_index)
            x = label_info_list[1]
            y = label_info_list[2]
            w = label_info_list[3]
            h = label_info_list[4]
 
            ########################
            # need perfection
            draw_label_rec(src_image, label_index, [x, y, w, h], img_name)
 
        resized_src = cv.resize(src_image, dim, interpolation=cv.INTER_CUBIC)
        # cv.imwrite(save_dir % (img_name), resized_src)
 
        # show the result
        cv.imshow(origin_window, resized_src)
        cv.waitKey(0)
 
        # Debug
        # print("src_height = {0}".format(src_height))
        # print("src_width = {0}".format(src_width))
        cv.destroyAllWindows()
 
        file_reader.close()
        print("**check over**")
 
 
if __name__ == "__main__":
    main()