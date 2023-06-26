#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import socket
import pickle
import struct
import cv2

def pack_image(frame):
    # encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
    # result, frame = cv2.imencode('.jpg', frame, encode_param)
    data = pickle.dumps(frame, 0)
    size = len(data)
    packed = struct.pack(">L", size) + data
    return packed, data, size


def get_predicate_result():
    data = sock.recv(4096)
    result = pickle.loads(data)
    return result


if __name__ == '__main__':
    ip_port = ('127.0.0.1', 5052)
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect(ip_port)
    dirpath = "/home/ur/Desktop/ur10e_sim/src/fmauch_universal_robot/ur_real_robot/VAE_detect/crop/hexagon_bolt/00000227_1.png"
    img = cv2.imread(dirpath)

    packed, data, size = pack_image(img)
    sock.sendall(packed)
    print("send all finished")
    result=(get_predicate_result())
    print(result)
