

import os
import pickle
import socket
import cv2
import struct
import numpy as np
import random
from PIL import Image
import select
import sys
import os
import random
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt


import torch
from torchvision import datasets
import torchvision.transforms as transforms

def unpack_image(conn):
    recv_data = b""
    data = b""
    print("unpack_image")
    payload_size = struct.calcsize(">l")
    while len(data) < payload_size:
        # print ('payload_size')
        recv_data += conn.recv(4096)
        # print (recv_data)
        if not recv_data:
            return None
        data += recv_data
    packed_msg_size = data[:payload_size]
    data = data[payload_size:]
    msg_size = struct.unpack(">l", packed_msg_size)[0]
    if msg_size < 0:
        return None
    print('unpack_image len(data): %d, msg_size %d' % (len(data), msg_size))
    while len(data) < msg_size:
        data += conn.recv(4096)

    frame_data = data[:msg_size]
    frame = pickle.loads(frame_data, fix_imports=True, encoding="bytes")
    # frame = cv2.imdecode(frame, cv2.IMREAD_COLOR)

    # print('cv2')
    return frame
def infer_checkpoint(img):
    attribute_in_model = torch.load('./checkpoint/net_in_img_97.pkl',map_location=torch.device(device))
    attribute_out_model = torch.load('./checkpoint/net_out_img_95.pkl',map_location=torch.device(device))
    attribute_in_dic={0:'cross_groove',1:'hex_groove',2:'plane',3:'star_groove'}
    attribute_out_dic={0:'hex',1:'round',2:'other1',3:'other2'}
    attribute_in2index={'cross_groove':0,'hex_groove':1,'plane':2,'star_groove':3}
    attribute_out2index={'hex':0,'round':1,'other1':2,'other2':3}
    attribute_in=np.zeros((4))
    attribute_out=np.zeros((4))
    
    for i in range (4):
        in_op_list = [
            {'op': 'objects', 'param': ''},
            {'op':'filter_nearest_obj', 'param': ''},
            {'op':'obj_attibute', 'param':attribute_in2index[attribute_in_dic[i]]}
        ]
        out_op_list = [
            {'op': 'objects', 'param': ''},
            {'op':'filter_nearest_obj', 'param': ''},
            {'op':'obj_attibute', 'param':attribute_out2index[attribute_out_dic[i]]}
        ]
        img_file_path=""
        y_pred_in = attribute_in_model(in_op_list, img,img_file_path, mode='test')
        if y_pred_in[0].data==1:
            attribute_in[i]=1
        y_pred_out = attribute_out_model(out_op_list, img,img_file_path, mode='test')
        if y_pred_out[0].data==1:
            attribute_out[i]=1
    print("attribute_in:",attribute_in,"attribute_out:",attribute_out)
    return attribute_in,attribute_out

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
ip_port = ('127.0.0.1', 5052)
server.bind(ip_port)
server.listen(5)
transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])


while True:
    conn, addr = server.accept()
    print(conn, addr)
    while True:
        try:
            frame = unpack_image(conn)
            if frame is None:
                print("client request stop")
                break
            
            frame_im = Image.fromarray(np.array(frame))
            # frame_im.show()
            print(frame_im.mode)
            # img = transform(frame_im)
            # img = torch.unsqueeze(img, 0)
            # img = img.to(device)
            attribute_in,attribute_out=infer_checkpoint(frame_im)
            # img_file="/home/xps/Desktop/ur10e_sim/src/fmauch_universal_robot/ur_real_robot/VAE_detect/true_mul_bolt_crops/cross_hex_bolt/0.jpg"
            # frame_im=Image.open(img_file)
            # print(frame_im.mode)
            # attribute_in,attribute_out=infer_checkpoint(frame_im)
            bolt_type=""
            if attribute_in[1]==1 and attribute_out[1]==1:
                bolt_type="in_hex_bolt"
                print("bolt_type:",bolt_type)
            elif attribute_in[3]==1 and attribute_out[1]==1:
                bolt_type="star_bolt"
                print("bolt_type:",bolt_type)
            elif attribute_in[0]==1 and attribute_out[0]==1:
                bolt_type="cross_hex_bolt"
                print("bolt_type:",bolt_type)
            elif attribute_in[2]==1 and attribute_out[0]==1:
                bolt_type="out_hex_bolt"
                print("bolt_type:",bolt_type)
            else:
                print("No matching bolt type")
            array_str = pickle.dumps(bolt_type, protocol=2)
            conn.sendall(array_str)

        except ConnectionResetError as e:
            print('the connection is lost')
            break
    conn.close()
