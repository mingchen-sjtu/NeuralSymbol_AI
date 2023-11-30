import argparse
import imp
from operator import imod, index, mod
from re import I
from turtle import forward
import torch
import numpy as np
import os, json, cv2, random
import xml.etree.ElementTree as ET
import torch.nn as nn
import torchvision
import torch.utils.data as Data
import json
import random
import time
from torch.autograd import Variable
# from graphviz import Digraph


from crop_pic_sin import crop_and_filter_objects
# from timm.data import Mixup

from models.rel_models import OnClassify_v1
# from demo import ImageTool
from models.reasoning_out_and_in import Reasoning
from models.reasoning_out_and_in import id2rel
from models.reasoning_out_and_in import ImageTool
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import transforms
import PIL
from torchviz import make_dot

DATA_INPUT = '/home/ur/Desktop/attribute_infer/bolt/data-end2end-triple/true_mul_bolt_crops/'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# device = torch.device("cpu")
imgtool = ImageTool()


# def make_dot(var, params=None):
#     if params is not None:
#         assert isinstance(params.values()[0], Variable)
#         param_map = {id(v): k for k, v in params.items()}
#
#     node_attr = dict(style='filled',
#                      shape='box',
#                      align='left',
#                      fontsize='12',
#                      ranksep='0.1',
#                      height='0.2')
#     dot = Digraph(node_attr=node_attr, graph_attr=dict(size="12,12"))
#     seen = set()
#
#     def size_to_str(size):
#         return '(' + (', ').join(['%d' % v for v in size]) + ')'
#
#     def add_nodes(var):
#         if var not in seen:
#             if torch.is_tensor(var):
#                 dot.node(str(id(var)), size_to_str(var.size()), fillcolor='orange')
#             elif hasattr(var, 'variable'):
#                 u = var.variable
#                 name = param_map[id(u)] if params is not None else ''
#                 node_name = '%s\n %s' % (name, size_to_str(u.size()))
#                 dot.node(str(id(var)), node_name, fillcolor='lightblue')
#             else:
#                 dot.node(str(id(var)), str(type(var).__name__))
#             seen.add(var)
#             if hasattr(var, 'next_functions'):
#                 for u in var.next_functions:
#                     if u[0] is not None:
#                         dot.edge(str(id(u[0])), str(id(var)))
#                         add_nodes(u[0])
#             if hasattr(var, 'saved_tensors'):
#                 for t in var.saved_tensors:
#                     dot.edge(str(id(t)), str(id(var)))
#                     add_nodes(t)
#
#     add_nodes(var.grad_fn)
#     return dot


def get_args_parser():
    parser = argparse.ArgumentParser('MAE fine-tuning for image classification', add_help=False)
    parser.add_argument('--model', default='vit_base_patch16', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--cutmix_minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--finetune', default='/home/ur/Desktop/attribute_infer/bolt/model/mae_pretrain_vit_base.pth',
                        help='finetune from checkpoint')
    parser.add_argument('--drop_path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')
    parser.add_argument('--global_pool', action='store_true')
    parser.set_defaults(global_pool=True)
    parser.add_argument('--cls_token', action='store_false', dest='global_pool',
                        help='Use class token instead of global pool for classification')
    parser.add_argument('--eval', action='store_true',
                        help='Perform evaluation only')
    parser.add_argument('--dist_eval', action='store_true', default=False,
                        help='Enabling distributed evaluation (recommended during training for faster monitor')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')

    return parser


def train(model):

    train_set = TripleDataset(DATA_INPUT + 'attribute_out_base_in_train.tsv')
    test_set = TripleDataset(DATA_INPUT + 'attribute_out_base_in_test.tsv')  # use triple

    lr = 0.001
    epoch_num = 100
    loss_function = nn.MSELoss()
    # loss_function = nn.CrossEntropyLoss()
    # loss_function = nn.BCELoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    torch.set_grad_enabled(True)
    model.train()

    model.to(device)

    running_loss = 0

    index_list = list(range(train_set.len))
    random.shuffle(index_list)
    # print(index_list)

    for epoch in range(epoch_num):
        print('------- Epoch', epoch, '-------')
        ans_pos = 0
        ans_neg = 0
        train_loss = 0
        loss_pos = 0
        loss_neg = 0
        acc = 0
        train_pos_cnt = 0
        train_neg_cnt = 0
        test_pos_cnt = 0
        test_neg_cnt = 0
        # 观测度量
        test_loss = 0
        pred_tot = 0
        pred_pos = 0
        pred_neg = 0
        # 性能度量
        acc = 0
        acc_pos = 0
        acc_neg = 0
        TP = 0
        FP = 0
        TN = 0
        FN = 0
        best_score = 0.3
        for i in index_list:
            (img, op_lists, answers,img_file_path) = train_set[i]
            for i in range(2):
                optimizer.zero_grad()
                # start_time = time.time()
                # print("start time begin")
                y_pred = model(op_lists[i], img,img_file_path, mode='train')
                model.concept_matrix2zero()
                

                loss = loss_function(y_pred, answers[i])
                loss.requires_grad_(True)
                # make_dot(loss).view()
                train_loss += loss.data
                loss.backward()
                optimizer.step()

        # ------------------- test model ---------------
        # test_set = train_set
        for i in range(test_set.len):
            (img, op_lists, answers,img_file_path) = test_set[i]
            for i in range(2):
                y_pred = model(op_lists[i], img,img_file_path, mode='train')
                model.concept_matrix2zero()
                loss = loss_function(y_pred, answers[i])
                test_loss += loss.data
                y_pred = model(op_lists[i], img,img_file_path,  mode='test')
                model.concept_matrix2zero()
                print(y_pred)

                print(answers[i])
            # acc compute
                if y_pred.equal(answers[i]) :
                    acc += 1


        # print('[INFO] pos cnt', train_pos_cnt, 'neg cnt', train_neg_cnt)
        # print('[INFO] pos ans', ans_pos, 'neg ans', ans_neg)
        print('[INFO] ---train---')
        print('[INFO]---- train loss ----:', train_loss /( train_set.len*2))
        # print('[INFO]---- train pos loss ----:', loss_pos / train_pos_cnt)
        # print('[INFO]---- train neg loss ----:', loss_neg / train_neg_cnt)
        print('[INFO] ---test---')
        # print('[INFO]---- pred avg ----:', pred_tot / test_set.len)
        # print('[INFO]---- pred avg pos----:', pred_pos / test_pos_cnt)
        # print('[INFO]---- pred avg neg----:', pred_neg / test_neg_cnt)
        print('[INFO]---- test loss ----:', test_loss / (test_set.len*2))
        print('[INFO] ---eval---')
        print('[INFO]---- test acc ----:', acc / (test_set.len*2))
        # print('[INFO]---- test acc pos----:', acc_pos / test_pos_cnt)
        # print('[INFO]---- test acc neg----:', acc_neg / test_neg_cnt)
        # print('[INFO]---- test P ----:', P)
        # print('[INFO]---- test R ----:', R)
        # print('[INFO]---- test F1 ----:', F1)

        # if F1 >= best_score:
        if round(acc / (test_set.len*2), 2) >= best_score:

            # best_score = round(F1, 2)
            best_score = round(acc / (test_set.len*2), 2)
            name_str = './checkpoint/model_outbasein_best_4neg.pkl'.replace('best', str(best_score))
            torch.save(model, name_str)
            # infer_checkpoint(model)
            # break


def iii():
    attribute_out_dic={0:'hex1',1:'round1',2:'hex2',3:'round2'}
    # model = torch.load('./checkpoint/reason_model_zero_1.0_4neg.pkl', map_location=torch.device(device))
    model = torch.load('./checkpoint/model_outbasein_0.95_4neg.pkl',map_location=torch.device(device))
    for i in range(len(attribute_out_dic)):

        infer_checkpoint(model,attribute_out_dic[i])


def infer_checkpoint(model,atribute_out):

    attribute_out2bolt={'hex1':'out_hex_bolt','round1':'in_hex_bolt','hex2':'cross_hex_bolt','round2':'star_bolt'}
    attribute_out2index={'hex1':0,'round1':1,'hex2':0,'round2':1}
    op_list = [
            {'op': 'objects', 'param': ''},
            {'op':'filter_nearest_obj', 'param': ''},
            {'op':'obj_attibute', 'param': [1,attribute_out2index[atribute_out]]}
    ]
    img_list = list(range(408, 409))
    # img_list = [1]
    print('[INFO]---------- 评分测试 ---------')
    tol_num=0
    val_num=0
    for img_id in img_list:
        # if img_id == 43:
        #     continue
        # print('[INFO]---case', img_id, '----')
        img_file_path = DATA_INPUT + attribute_out2bolt[atribute_out]+'/' + str(img_id).zfill(3) + '.jpg'
        # ann_file = DATA_INPUT + 'Annotation/shut-' + str(img_id).zfill(3) + '.xml'
        # print(img_file)
        img = imgtool.load_img(img_file_path)
        # img_file_path=""
        y_pred = model(op_list, img,img_file_path, mode='test')
        model.concept_matrix2zero()
        # max_val,index = torch.max(y_pred,dim=1)
        # print(index)
        tol_num+=1
        print( y_pred[0].data)
        if y_pred[0].data==1:
            val_num+=1
        # print('[INFO] image:', img_id, 'attribute',y_pred)
    print('[INFO] attributu', atribute_out,'total num:', tol_num, 'correct num',val_num, 'accuracy',val_num/tol_num)
    # img_file_path ="/home/ur/Desktop/attribute_infer/bolt/data-end2end-triple/true_mul_bolt_crops/in_hex_bolt/003.jpg"

    # img = imgtool.load_img(img_file_path)

    # y_pred = model(op_list, img,img_file_path, mode='test')
    
    # print( y_pred[0].data)

class TripleDataset(Dataset):
    def __init__(self, file):

        # self.eye_matrix = torch.eye(1)
        self.imgtool = ImageTool()

        # with open(file) as f:
        #     triples = json.load(f)
        # f.close()
        triples_pos = []
        with open(file) as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                # if i == 0:
                #     continue
                (img_path, attribute_in) = line.split(' ')  # dataset triple 格式
                triples_pos.append((img_path, int(attribute_in)))

        # # 负采样
        # rate = 2  # 负采样比例
        # triples_neg = []
        # for triple in triples_pos:
        #     (img_id, obj_id, sub_id, rel_id) = triple
        #
        #     # 读取图片中物体数量
        #     img_file = DATA_INPUT + 'images/shut-' + str(img_id).zfill(3) + '.jpg'
        #     ann_file = DATA_INPUT + 'Annotation/shut-' + str(img_id).zfill(3) + '.xml'
        #     # print(img_file)
        #     img, ann = imgtool.load_img(img_file, ann_file)
        #     obj_num = len(ann)
        #
        #     cnt = 0
        #
        #     while (cnt < rate):
        #         # 打乱头尾实体，构造负例
        #         neg_sub_id = random.randint(0, obj_num - 1)
        #         neg_obj_id = random.randint(0, obj_num - 1)
        #         # neg_obj_id = obj_id
        #         neg_triple = (img_id, neg_obj_id, neg_sub_id, rel_id)
        #         if (neg_triple not in triples_pos) and (neg_triple not in triples_neg):
        #             triples_neg.append(neg_triple)
        #             cnt += 1
        #             # print(neg_triple)
        # print('[INFO] neg sample finish')
        # # print(len(triples_pos), len(triples_neg))

        self.triples_pos = triples_pos
        # self.triples_neg = triples_neg
        # self.triples = triples_pos + triples_neg
        self.triples = triples_pos
        self.len = len(self.triples)

        # convert triple to QA
        questions_pos = []
        for triple in triples_pos:
            # question = {
            #     "image_id": triple[0],
            #     "op_list": [
            #         {"op": "objects", "param": ""},  # 获取所有物体
            #         {"op": "filter_index", "param": triple[1]},  # 通过filter找到 triple中object的物体
            #         {"op": "relate", "param": id2rel[triple[3]]},  # 以object物体为起点，通过relate操作查询于其具有triple中rel关系的物体
            #         {"op": "filter_index", "param": triple[2]}],  # 通过filter判定该物体是否是triple中subject物体
            #     "answer": triple[2],  # 答案就是triple中的subject
            #     "type": "index"
            # }
            attribute_list=torch.zeros((1,4))
            attribute_list[0][triple[1]]=1#[[1,0,0,0]]
            attribute_list=torch.reshape(attribute_list, (1, -1))
            # print(" attribute_list:", attribute_list)
            question =[] 
            for i in range(2):
                answer=torch.zeros((1))
                answer[0]=attribute_list[0][i]#[0]or[1]
                q={
                    "image_path": triple[0],
                    "op_list": [
                        {"op": "objects", "param": ""},  # 获取所有物体
                        {"op": "filter_nearest_obj", "param": ""},  # 找到最近的物体
                        {"op": "obj_attibute", "param":[1,i]}],  # 通过filter判定该物体是否是triple中subject物体
                    "answer": answer,  # 答案就是triple中的subject
                }
                question.append(q)
            questions_pos.append(question)

        #questions_neg = []
        # for triple in triples_neg:
        #     question = {
        #         "image_id": triple[0],
        #         "op_list": [
        #             {"op": "objects", "param": ""},
        #             {"op": "filter_index", "param": triple[1]},
        #             {"op": "relate", "param": id2rel[triple[3]]},
        #             {"op": "filter_index", "param": triple[2]}],
        #         "answer": triple[2],
        #         "type": "index_neg"
        #     }
        #     questions_neg.append(question)

        # self.questions = questions_pos + questions_neg
        self.questions = questions_pos
        self.len = len(self.questions)

    def __getitem__(self, index):
        img_path = self.questions[index][0]['image_path']#取出第一个op_list的图像作为图像

        img_file_path = DATA_INPUT + img_path
        # ann_file = DATA_INPUT + 'Annotation/shut-' + str(img_id).zfill(3) + '.xml'
        # print(img_file)
        img = imgtool.load_img(img_file_path)
        op_lists=[]
        for i in range(2):
            op_list = self.questions[index][i]['op_list']
            op_lists.append(op_list)
        answers=[]
        for i in range(2):
            answer =self.questions[index][i]['answer']
            answers.append(answer)

        return (img, op_lists, answers,img_file_path)


    # def __getitem__(self, index):
    #     img_id = self.questions[index]['image_id']
    #     img_id_1 = self.questions[index+1]['image_id']
    #
    #     img_file = DATA_INPUT + 'images/shut-' + str(img_id).zfill(3) + '.jpg'
    #     ann_file = DATA_INPUT + 'Annotation/shut-' + str(img_id).zfill(3) + '.xml'
    #
    #     img_file_1 = DATA_INPUT + 'images/shut-' + str(img_id_1).zfill(3) + '.jpg'
    #     ann_file_1 = DATA_INPUT + 'Annotation/shut-' + str(img_id_1).zfill(3) + '.xml'
    #
    #     # print(img_file)
    #     img, ann = imgtool.load_img(img_file, ann_file)
    #     op_list = self.questions[index]['op_list']
    #     type = self.questions[index]['type']
    #     name_t = 'shut-' + str(img_id).zfill(3)
    #
    #     img_1, ann_1 = imgtool.load_img(img_file_1, ann_file_1)
    #     op_list_1 = self.questions[index+1]['op_list']
    #     type_1 = self.questions[index+1]['type']
    #     name_t_1 = 'shut-' + str(img_id_1).zfill(3)
    #
    #     if type == "index_neg":
    #         answer = torch.zeros(60)  # 构建一个所有物体都不被选中为0的向量
    #     else:
    #         answer = self.eye_matrix[self.questions[index]['answer']]  # 构建一个仅有答案obj为1其他均为0的向量
    #
    #     pic = '/yq/ddd/intel_amm_2/data-end2end-triple/images/shut-' + str(img_id).zfill(3) + '.jpg'
    #     xml_path = '/yq/ddd/intel_amm_2/data-end2end-triple/Annotation/shut-' + str(img_id).zfill(3) + '.xml'
    #     crop_and_filter_objects(pic, xml_path)
    #
    #     return (img, ann, op_list, answer, type, name_t)



    def __len__(self):
        return self.len


if __name__ == "__main__":

    args = get_args_parser()
    args = args.parse_args()
    # Model Load
    # model_in = Reasoning(args)
    # for name, param in model_in.named_parameters():
    #     print(name, param.size(), type(param))
    # model_out = torch.load('./checkpoint/net_in_0.9862.pkl',map_location=torch.device(device))

    # train(model_out) # 训练流程
    
    iii()  # 测试流程