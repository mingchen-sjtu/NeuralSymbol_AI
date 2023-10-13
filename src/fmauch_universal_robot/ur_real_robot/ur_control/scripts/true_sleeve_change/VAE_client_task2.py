#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import socket
import pickle
import struct
import cv2
import numpy as np
from cv_bridge import CvBridge, CvBridgeError
from YOLO_client import YOLO_SendImg
import math
from PIL import Image


class VAE_SendImg():
    def __init__(self, ip_port=('127.0.0.1', 5052)):
        self.ip_port = ip_port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.connect(ip_port)
        # self.bolt_detector=YOLO_SendImg()

    def pack_image(self, frame):
        # encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
        # result, frame = cv2.imencode('.jpg', frame, encode_param)
        # data_encode = np.array(frame)
        # str_encode = data_encode.tostring()
        
        # # 缓存数据保存到本地，以txt格式保存
        # with open('img_encode.txt', 'w') as f:
        #     f.write(str_encode)
        #     f.flush
        
        # with open('img_encode.txt', 'r') as f:
        #     str_encode = f.read()
        
        # nparr = np.fromstring(str_encode, np.uint8)
        # img_decode = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        # cv2.imwrite("img_decode_2.jpg", img_decode)
        # cv2.imshow("img_decode", img_decode)
        # cv2.waitKey()

        data = pickle.dumps(frame, 0)
        size = len(data)
        packed = struct.pack(">L", size) + data
        return packed, data, size

    def get_VAE_result(self):
        data = self.sock.recv(4096)
        result = pickle.loads(data)
        return result

    def finish_VAE_detect(self, img, result):
        
        imgs=[]
        crop_img=[]
        circlesbox=[]
        circlescenter=[]
        frame_img = Image.fromarray(np.array(img))
        # for bolt in result[0].keys():
        if 'bolt' in result[0].keys():
                print('bolt center success')
                circlescenter.extend(result[0]['bolt'])
        # for bolt in result[1].keys():
        if 'bolt' in result[1].keys():
                print('bolt box success')
                circlesbox.extend(result[1]['bolt'])
        for bolt_b in circlesbox:
            print(bolt_b)
            crop_image = frame_img.crop(bolt_b)
            crop_img.append(crop_image)
        # for bolt in result[2].keys():
        #     print('bolt img success')
        #     imgs.append(result[2][bolt])
        # if 'screw' in result[0].keys():
        #     print('screw center success')
        #     circlescenter.extend(result[0]["screw"])
        # if 'screw' in result[1].keys():
        #     print('screw box success')
        #     circlesbox.extend(result[1]["screw"])
        # if 'screw' in result[2].keys():
        #             print('screw img success')
        #             imgs.append(result[2]["screw"])
        # print(imgs[0])
        # for img in imgs[0]:
        #     crop_img.append(img[0])
            # print(img[0])
        # print(crop_img)
        i=0
        aim_bolt=0
        min_dis=100
        for bolt_c in circlescenter:
            dis=math.sqrt(pow(bolt_c[0] - 433 ,2)+pow(bolt_c[1] - 236 ,2))
            if dis<min_dis:
                min_dis=dis
                aim_bolt=i
            i+=1   
        # print(aim_bolt)
        frame=crop_img[aim_bolt]
        frame.show()
        packed, data, size = self.pack_image(frame)
        self.sock.sendall(packed)
        print("send all finished")
        print(result)
        left=circlesbox[aim_bolt][0]
        right=circlesbox[aim_bolt][2]
        top=circlesbox[aim_bolt][1]
        bottom=circlesbox[aim_bolt][3]
        bolt_size=int(abs(right-left)+abs(top-bottom))/2
        print("bolt_size:",bolt_size)
        result = self.get_VAE_result()
        if result == 'in_hex_bolt':
            result='in_hex_bolt'
        elif result == 'star_bolt':
            # result='star_bolt'
            if bolt_size>=33:
                result='cross_hex_bolt_13'
            elif bolt_size>=28:
                result='cross_hex_bolt_10'
            else:
                result='cross_hex_bolt_8'
        elif result == 'out_hex_bolt':
            if bolt_size>=33:
                result='out_hex_bolt_13'
            # elif bolt_size>=33:
            elif bolt_size>=28:
                result='out_hex_bolt_10'
            else:
                result='out_hex_bolt_8'
        elif result == 'cross_hex_bolt':
            if bolt_size>=33:
                result='cross_hex_bolt_13'
            elif bolt_size>=28:
                result='cross_hex_bolt_10'
            else:
                result='cross_hex_bolt_8'
        # elif result == 'm_bolt':
        #     left=circlesbox[aim_bolt][0]
        #     right=circlesbox[aim_bolt][2]
        #     top=circlesbox[aim_bolt][1]
        #     bottom=circlesbox[aim_bolt][3]
        #     bolt_size=int(abs(right-left)+abs(top-bottom))/2
        #     print(bolt_size)
        #     if bolt_size>22:
        #         result='M10_bolt'
        #     elif bolt_size<20:
        #         result='M6_bolt'
        #     else:
        #         result='M8_bolt'
        
        return result


if __name__ == '__main__':
    bolt_type_detector = VAE_SendImg()
    # frame = cv2.imread('src/ur_real_robot/YOLO_v5_detect/imgs/img_decode_2.jpg')
    frame = cv2.imread('/home/ur/Desktop/ur10e_sim/src/fmauch_universal_robot/ur_real_robot/YOLO_v5_detect/imgs/00000307.png')
    frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    result = bolt_type_detector.bolt_detector.finish_YOLO_detect(frame)
    result = bolt_type_detector.finish_VAE_detect(result)
    # frame1 = cv2.imread('src/ur_real_robot/YOLO_v5_detect/imgs/3.jpg')
    # result1 = bolt_detector.finish_YOLO_detect(frame1)
    # bolt_detector.sock.close()
    # print(result['bolt0'])
    # print(result)
    # print(result1)
