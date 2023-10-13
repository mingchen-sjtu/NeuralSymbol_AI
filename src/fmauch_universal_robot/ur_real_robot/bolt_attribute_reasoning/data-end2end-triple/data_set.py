import xml.etree.ElementTree as ET
import pickle
import os
from os import listdir, getcwd
from os.path import join
import random

sets=[('dataset', 'train'), ('dataset', 'test')]
# images_type=['images_m','images_cross','images_star','images_hex']
images_type=['images_m','images_cross','images_star','images_hex']


wd = getcwd()
file = '/home/ur/Desktop/attribute_infer/bolt/data-end2end-triple/true_mul_bolt_crops'
bolt_class = [cla for cla in os.listdir(file) if ((".txt" not in cla)and(".tsv" not in cla) )]
n=0
for cla in bolt_class:
    cla_path = file + '/' + cla + '/'
    print(cla_path)
    images_type[n]= os.listdir(cla_path)
    # print(images_type[n])
    n+=1
images_path=['cross_hex_bolt','in_hex_bolt','out_hex_bolt','star_bolt']#change according to read order

list_file = open('/home/ur/Desktop/attribute_infer/bolt/data-end2end-triple/true_mul_bolt_crops/attribute_out_base_in_train.tsv', 'w')
for i in range(350):
    for j in range(4):
        k=0
        if j==0 or j==2 :
           k=0
        else:
            k=1
        # image = random.sample(images_type[j], k=1)
        list_file.write('%s/%s %s\n'%(images_path[j],''+str(i).zfill(3)+'.jpg',k))
list_file.close()
list_file = open('/home/ur/Desktop/attribute_infer/bolt/data-end2end-triple/true_mul_bolt_crops/attribute_out_base_in_test.tsv', 'w')
for i in range(50):
    for j in range(4):
        k=0
        if j==0 or j==2 :
           k=0
        else:
            k=1
        # image = random.sample(images_type[j], k=1)
        list_file.write('%s/%s %s\n'%(images_path[j],''+str(i+350).zfill(3)+'.jpg',k))
list_file.close()
