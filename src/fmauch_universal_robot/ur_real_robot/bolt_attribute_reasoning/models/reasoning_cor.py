import torch
import numpy as np
import os, json, cv2, random
import xml.etree.ElementTree as ET
import torch.nn as nn
from PIL.ImageDraw import ImageDraw
from PIL.ImageFont import ImageFont
from timm.data.mixup import Mixup

import time
from crop_pic_sin import crop_and_filter_objects
from models.vit_attribute_model import OnClassify_v3
from models.models_vit import VisionTransformer

from models.rel_models import OnClassify_v1
from models.engine_finetune import train_one_epoch
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import transforms
from PIL import Image
from torchviz import make_dot
from  models import models_vit

max_attribute_num = 3

max_obj_num = 1
max_concept_num = 4
max_rel_num = 10

concept_matrix = None
relate_matrix = None

attribute2id = {'in_contour': 0, 'out_contour': 1, 'corrosion': 2}
# concept2id_name = {'电流表最小刻度(ammeter_min_scale)': 0, '电流表和电压表指针(pointer)_a': 1,  '电流表(ammeter)': 2, '其他(others)': 3}
concept2id_name = {'in_contour': {'plane': 0,'hex_groove': 1,'cross_groove': 2,'star_groove': 3},
'out_contour': {'hex': 0,'round': 1,'others_1': 2,'others_2': 3},
'corrosion': {'no_corrosion': 0,'small_corrosion': 1,'medium_corrosion': 2,'severe_corrosion': 3}}
# concept2id_name = {'bolt': 0,'其他(others)': 1}
rel2id = {'zero': 0, 'unzero': 1}
id2rel = {0: 'zero', 1: 'unzero'}

colors = [[48, 48, 255]]  # 255 48 48 RGB


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")


class Reasoning(nn.Module):
    '''
    Reasoning负责整体逻辑规则的顺序计算及逻辑判断
    '''

    def __init__(self, args):
        super(Reasoning, self).__init__()
        self.exec = Executor(args)
        self.imgtool = ImageTool()

    def forward(self, dcl_op_list, img, mode):
        '''
        说明：对于每个operation的执行结果，统一都是采用一个一维的行向量进行表示，行向量中每个元素表示其所对应的物体在该步骤之后被选中的概率
        （具体见The neuro-symbolic concept learner, Sec 3.1 Quasi-symbolic program execution部分解释）
        '''

        obj_name_list = []  # 列表包含每张图片中所有物体的name,若为开关柄和基座则正常输入name,其他物体则name置为"其他"
        # for label in ann:
        #     obj_name_list.append(label['name'])
        # print(obj_name_list)

        exec = self.exec
        exec.init(img, mode)

        buffer = []
        step = 0
        flag_neg = False
        for tmp in dcl_op_list:
            step += 1
            op = tmp['op']
            param = tmp['param']
            # if(mode == 'infer'):
            #     print('[INFO] current op', op, param)
            if op == 'objects':
                buffer.append(torch.ones(max_obj_num, requires_grad=False))
                # img, ann = self.imgtool.load_img('./data-demo/image/img1.jpg', './data-demo/annotation/img1_annotation.xml')
                # img = self.imgtool.addbox(img, buffer[-1], ann)
                # file = './data-demo/image/out0.jpg'.replace('0', str(step))
                # self.imgtool.save_img(file, img)
                # continue
            elif op == 'filter_nearest_obj':
                buffer.append(exec.filter_nearest_obj(buffer[-1], param))
            elif op == 'obj_attibute':
                buffer.append(exec.obj_attibute(buffer[-1], param))
            elif op == 'attibute2sleeve':
                buffer.append(exec.attibute2sleeve(buffer[-1], param))
            elif op == 'filter_name':
                buffer.append(exec.filter_obj_name(buffer[-1], param))
                # img, ann = self.imgtool.load_img('./data-demo/image/img1.jpg', './data-demo/annotation/img1_annotation.xml')
                # img = self.imgtool.addbox(img, buffer[-1], ann)
                # file = './data-demo/image/out0.jpg'.replace('0', str(step))
                # self.imgtool.save_img(file, img)
            elif op == 'filter_index':
                buffer.append(exec.filter_obj_index(buffer[-1], param))
                # img, ann = self.imgtool.load_img('./data-demo/image/img1.jpg', './data-demo/annotation/img1_annotation.xml')
                # img = self.imgtool.addbox(img, buffer[-1], ann)
                # file = './data-demo/image/out0.jpg'.replace('0', str(step))
                # self.imgtool.save_img(file, img)
            elif op == 'relate':
                buffer.append(exec.relate(buffer[-1], param))
                # img, ann = self.imgtool.load_img('./data-demo/image/img1.jpg', './data-demo/annotation/img1_annotation.xml')
                # img = self.imgtool.addbox(img, buffer[-1], ann)
                # file = './data-demo/image/out0.jpg'.replace('0', str(step))
                # self.imgtool.save_img(file, img)
            elif op == 'count':
                buffer.append(exec.count(buffer[-1]))
            elif op == 'intersect':
                buffer.append(exec.intersect(buffer[-1], buffer[param]))
            elif op == 'union':
                buffer.append(exec.union(buffer[-1], buffer[param]))
            elif op == 'and':
                # print(buffer[-1], )
                # print(buffer)
                buffer.append(exec.and_bool(buffer[-1], buffer[param]))
            elif op == 'or':
                buffer.append(exec.or_bool(buffer[-1], buffer[param]))
            elif op == 'exist':
                # img, ann = self.imgtool.load_img('./data-demo/image/img1.jpg', './data-demo/annotation/img1_annotation.xml')
                # img = self.imgtool.addbox(img, buffer[-1], ann)
                # file = './data-demo/image/out0.jpg'.replace('0', str(step))
                # self.imgtool.save_img(file, img)
                buffer.append(exec.exist(buffer[-1]))
            else:
                print('[!!!ERROR!!!] operator not exist!')
            if (mode == 'infer'):
                # print('[INFO]', op, param,  buffer[-1].data)
                print('[INFO]', op, param, end=' ')
                step_score = exec.exist(buffer[-1])
                if step_score <= 0.5 and flag_neg == False:
                    print('<---- Bad Step!', end=' ')
                    flag_neg = True
                    # print('[INFO] 错误操作', op, param)
                    # return exec.exist(buffer[-1])
                print('')
        answer = buffer[-1]
        return answer


class Predicator(nn.Module):
    '''
    Predicator集成一些谓词及相关计算
    '''

    def __init__(self, args):
        super(Predicator, self).__init__()

        # self.net_zero = models_vit.__dict__[args.model](
        #     num_classes=2,
        #     drop_path_rate=0.1,
        #     global_pool=args.global_pool,
        # )
        self.net_cor = OnClassify_v3(args)
        # self.net_out = OnClassify_v3(args)
        # self.net_cor = OnClassify_v3(args)


        # self.net_zero = VisionTransformer()

        # self.net_con = OnClassify_v1()
        # self.net_uncon = OnClassify_v1()
        # self.net_on = OnClassify_v1()
        # self.net_down = OnClassify_v1()
        # self.net_on = OnClassify_v1()
        # self.net_right = OnClassify_v1()
        # self.net_balance = OnClassify_v1()

    # def box_corner_to_center(self, box):
    #     """从（左上，右下）转换到（中间，宽度，高度）"""
    #     x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
    #     cx = (x1 + x2) / 2
    #     cy = (y1 + y2) / 2
    #     w = x2 - x1
    #     h = y2 - y1
    #     return (cx, cy, w, h)

    # def IoU(self, boxA, boxB):
    #     boxA = [int(x) for x in boxA]
    #     boxB = [int(x) for x in boxB]

    #     xA = max(boxA[0], boxB[0])
    #     yA = max(boxA[1], boxB[1])
    #     xB = min(boxA[2], boxB[2])
    #     yB = min(boxA[3], boxB[3])

    #     interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    #     boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    #     boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    #     iou = interArea / float(boxAArea + boxBArea - interArea)

    #     return iou

    # def on(self, aobj, bobj):
    #     '''
    #     if a on b
    #     implement by IOU
    #     '''
    #     xmin_a, ymin_a, xmax_a, ymax_a = aobj['box']
    #     xmin_b, ymin_b, xmax_b, ymax_b = bobj['box']
    #     if (ymax_a + ymin_a <= ymax_b + ymin_b):
    #         value = self.IoU(aobj['box'], bobj['box'])
    #         return value
    #     else:
    #         return 0


    # def con_classify(self, aobj, bobj):
    #     '''
    #     if a connected b
    #     implement by network
    #     '''
    #     xmin_a, ymin_a, xmax_a, ymax_a = aobj['box']
    #     xmin_b, ymin_b, xmax_b, ymax_b = bobj['box']
    #     x = [xmin_a, ymin_a, xmax_a - xmin_a, ymax_a - ymin_a, xmin_b, ymin_b, xmax_b - xmin_b,
    #          ymax_b - ymin_b]               # 拼接特征 feature: [x1, y1, w1, h1, x2, y2, w2, h2]
    #     x = np.array(x)
    #     x = torch.from_numpy(x)
    #     x = x.unsqueeze(dim=0)              # [,8][1, 8]
    #     # print(x)
    #     # print(x.shape)
    #     y_pred = self.net_con(x.to(device, torch.float))
    #     # pred = y_pred.softmax(dim=1)
    #     # return pred
    #     return y_pred

    def attributes_classify(self, img):
        '''
        if ammeter's pointer return to 0
        implement by network
        '''


        # xmin_a, ymin_a, xmax_a, ymax_a = aobj['box']
        # xmin_b, ymin_b, xmax_b, ymax_b = bobj['box']
        # x = [xmin_a, ymin_a, xmax_a - xmin_a, ymax_a - ymin_a, xmin_b, ymin_b, xmax_b - xmin_b,
        #      ymax_b - ymin_b]               # 拼接特征 feature: [x1, y1, w1, h1, x2, y2, w2, h2]
        # x = np.array(x)
        # x = torch.from_numpy(x)
        # x = x.unsqueeze(dim=0)              # [,8][1, 8]
        # print(x)
        # print(x.shape)

        # pic = '/yq/ddd/intel_amm_2/data-end2end-triple/images/' + name_t + '.jpg'
        # xml_path = '/yq/ddd/intel_amm_2/data-end2end-triple/Annotation/' + name_t + '.xml'

        # mixup_fn = Mixup(
        #     mixup_alpha=0.8, cutmix_alpha=1.0, cutmix_minmax=None,
        #     prob=1.0, switch_prob=0.5, mode='batch',
        #     label_smoothing=0.1, num_classes=2)

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        # start_time_classify = time.time()


        # crop_and_filter_objects(pic, xml_path)
        # pic_1_path = "/yq/ddd/intel_amm_2/data-end2end-triple/crop_images/" + name_t + '.jpg'
        pic_1 = Image.open(img)
        pic_ten = transform(pic_1)
        pic_ten = pic_ten.unsqueeze(0)


        # if mixup_fn is not None:
        #     pic_ten = mixup_fn(pic_ten)

        y_pre_in = torch.zero((1,4))
        # y_pre_out = self.net_out(pic_ten.to(device, torch.float))
        # y_pre_cor = self.net_cor(pic_ten.to(device, torch.float))
        y_pre_out = torch.zero((1,4))
        y_pre_cor = self.net_in(pic_ten.to(device, torch.float))

        # y_pre[ y_pre>0.5]=1
        # y_pre[ y_pre<0.5]=0

        # loss_1 = y_pre
        # make_dot(loss_1).view()
        # dot = make_dot(loss_1)
        # dot.format = 'png'
        # dot.render(filename='graph_1')
        # max_val = torch.max(y_pre)
        # loss_1 = max_val
        # make_dot(loss_1).view()
        # dot = make_dot(loss_1)
        # dot.format = 'png'
        # dot.render(filename='graph_2')
        # y_pred = torch.tensor([[max_val]])
        y_pred=torch.cat((y_pre_in, y_pre_out,y_pre_cor),dim=0)
        y_pred = torch.reshape(y_pred, (3, -1))

        # loss_1 = y_pred
        # make_dot(loss_1).view()
        # dot = make_dot(loss_1)
        # dot.format = 'png'
        # dot.render(filename='graph_3')

        # end_time_classify = time.time()  # 时间计算
        # classify_time = end_time_classify - start_time_classify
        # run_time = str(classify_time)
        # print("classify_time = " + run_time)

        # pred = y_pred.softmax(dim=1)
        # return pred
        return y_pred




    # def balance_classify(self, aobj):
    #     x = 0
    #     y_pred = self.net_balance(x.to(device, torch.float))
    #     return y_pred

    # def touch(self, aobj, bobj):
    #     '''
    #     if a touch b
    #     implement by IOU
    #     '''
    #     value = self.IoU(aobj['box'], bobj['box'])
    #     return value


class ImageTool():
    '''
     Image读取相关方法实现
    '''

    def __init__(self):
        self.transform = transforms.Compose([
            transforms.Resize(224),  # 缩放图片(Image)，保持长宽比不变，最短边为224像素
            transforms.CenterCrop(224),  # 从图片中间切出224*224的图片
            transforms.ToTensor(),  # 将图片(Image)转成Tensor，归一化至[0, 1]（直接除以255）
            # transforms.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5]) # 标准化至[-1, 1]，规定均值和标准差
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 常用标准化
        ])

    def load_img(self, img_file):
        img = cv2.imread(img_file)
        # tmp = PIL.Image.open(img_file)
        # img = self.transform(tmp)

        # image_path, labels = self.parse_xml(anta_file)

        return img

    def parse_xml(self, xml_file):
        """
        解析 xml_文件
        输入：xml文件路径
        返回：图像路径和对应的label信息
        """
        # 使用ElementTree解析xml文件
        tree = ET.parse(xml_file)
        root = tree.getroot()
        image_path = ''
        labels = []
        DATA_PATH = ''

        for item in root:
            if item.tag == 'filename':
                image_path = os.path.join(DATA_PATH, 'VOC2007/JPEGImages', item.text)
            elif item.tag == 'object':
                obj_name = item[0].text
                # 将objetc的名称转换为ID
                # obj_num = classes_num[obj_name]
                # 依次得到Bbox的左上和右下点的坐标
                xmin = int(item[4][0].text)
                ymin = int(item[4][1].text)
                xmax = int(item[4][2].text)
                ymax = int(item[4][3].text)

                # if obj_name == '电流表最小刻度(ammeter_min_scale)' or obj_name == '电流表和电压表指针(pointer)_a' or obj_name == '电流表(ammeter)':
                if obj_name == 'bolt':
                    obj_name = obj_name
                else:
                    obj_name = '其他(others)'

                labels.append({'box': [xmin, ymin, xmax, ymax], 'name': obj_name})


        return image_path, labels

    # def img_ann_show(self, img, ann):
    def drawOneBox(img, bbox, color, label):
        '''对于给定的图像与给定的与类别标注信息，在图片上绘制出bbox并且标注上指定的类别
        '''
        colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(concept2id_name))]
        img_PIL = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        # locate *.ttc
        font = ImageFont.truetype("NotoSansCJK-Bold.ttc", 20, encoding='utf-8')

        x1, y1, x2, y2 = bbox
        draw = ImageDraw.Draw(img_PIL)
        position = (x1, y1 - 30)
        draw.text(position, label, tuple(color), font=font)
        img = cv2.cvtColor(np.asarray(img_PIL), cv2.COLOR_RGB2BGR)
        cv2.rectangle(img, (x1, y1), (x2, y2), colors[0], 2)
        return img

    def save_img(self, img_file, img):
        cv2.imwrite(img_file, img)

    def addbox(self, img, buffer, ann):
        for obj_index in range(min(max_obj_num, len(ann))):
            xmin, ymin, xmax, ymax = ann[obj_index]['box']
            name = ann[obj_index]['name']
            if buffer[obj_index] > 0:
                # cv2.rectangle(img, (x1,y1), (x2, y2), colors[0], 2)
                cv2.rectangle(img, (xmin, ymin), (xmax, ymax), colors[0], 2)
                text = str(name) + str(obj_index) + str(buffer[obj_index].data)
                Font = cv2.FONT_HERSHEY_SIMPLEX
                # cv2.putText(img, text, (x,y), Font, Size, (B,G,R), Thickness)
                cv2.putText(img, text, (xmin, ymin - 10), Font, 0.5, colors[0], 1)

        return img


class Executor(nn.Module):
    '''
    Executor负责具体实现每个operator的运算过程
    计算公式具体见concept learner page 18
    '''

    '''
    attribute2id = {'name': 0, 'index': 1, 'balance': 2}
    concept2id_name = {'开关整体(switch)': 0, '开关柄(switch_handle)': 1}
    rel2id = {'up': 0, 'down': 1}
    id2rel = {0: 'up', 1: 'down'}
    '''
    def __init__(self, args):
        super(Executor, self).__init__()
        self.predicator = Predicator(args)
        self.concept_matrix = None
        self.relate_matrix = None

        # self._make_concept_matrix(ann)
        # # print(concept_matrix)
        # self._make_relate_matrix(ann)

    def init(self,img, mode):
        self.concept_matrix = self._make_concept_matrix(img, mode)
        # print(concept_matrix)
        # self.relate_matrix = self._make_relate_matrix(ann, name_t, id_a, id_b, mode)

    def filter_obj_name(self, selected, concept):
        '''
        '''
        concept_idx = concept2id_name[concept]
        mask = self._get_concept_mask(0, concept_idx)  # 0 is attribute name------即为name
        # mask = torch.min(selected, mask)
        mask = selected * mask

        return mask

    def filter_obj_index(self, selected, index):     #这个方法可以不要,index暂时还没实现
        '''
        filter by local index
        '''
        # mask = self._get_concept_mask(1, index)  # 0 is attribute name
        # # mask = torch.min(selected, mask)
        # mask = selected * mask
        mask = torch.zeros(max_obj_num, requires_grad=False)
        mask[index] = 1
        # mask = torch.min(selected, mask)
        mask = selected * mask

        return mask

    def relate(self, selected, rel):
        '''

        '''
        rel_idx = rel2id[rel]
        mask = self._get_relate_mask(rel_idx)
        mask = (mask * selected.unsqueeze(-1)).sum(dim=-2)
        # mask = torch.unsqueeze(selected, -1).sum(-2)
        return mask

    def filter_nearest_obj(self, selected):
        '''
        '''
        return 0
    def obj_attibute(self, selected):
        '''
        '''
        attibute_vec=torch.zeros(12, requires_grad=False)
        i=0
        for attritube_index in range(min(max_attribute_num, len(attribute2id))):
            for concept_index in range(min(max_concept_num, len(concept2id_name[attritube_index]))):
                attibute_vec[i]= self._get_concept_obj_mask(attritube_index,concept_index,selected)
                i+=1
        return attibute_vec
    def exist(self, selected):
        '''
        '''
        return selected.max()

    def count(self, selected):
        '''
        '''
        return selected.sum()

    def and_bool(self, selected1, selected2):
        '''
        '''
        # print(selected1, selected2)
        return torch.min(selected1, selected2)

    def or_bool(self, selected1, selected2):
        '''
        '''
        return torch.max(selected1, selected2)

    def intersect(self, selected1, selected2):
        return torch.min(selected1, selected2)

    def union(self, selected1, selected2):
        return torch.max(selected1, selected2)

    def _make_concept_matrix(self, img, mode):
        # global concept_matrix
        concept_matrix = torch.zeros((max_attribute_num, max_concept_num, max_obj_num), requires_grad=False)   #max_obj_num=60
        # 0 dim is for 'name' concept
        index=0
        res = self.predicator.attributes_classify(img)
        for attritube_index in range(min(max_attribute_num, len(attribute2id))):
            for concept_index in range(min(max_concept_num, len(concept2id_name[attritube_index]))):
                for obj_index in range(max_obj_num):
                    # print(ann[obj_index]['name'], concept2id_name[ann[obj_index]['name']])
                    # print(ann[obj_index]['name'])
                   
                    if mode == 'test':
                                # res = res.argmax(dim=1)
                                # relate_matrix[0][a_index][b_index] = res[0][0]
                                if (res[attritube_index][concept_index].data >= 0.5):
                                    concept_matrix[attritube_index][concept_index][obj_index] = 1 # [a_index][b_index]:a_index索引物体相对于b_index索引物体为up关系
                                else:
                                    concept_matrix[attritube_index][concept_index][obj_index] = 0

                    elif mode == 'train' or mode == 'infer':
                        concept_matrix[attritube_index][concept_index][obj_index] = res[attritube_index][concept_index]
                    # if concept2id_name[ann[obj_index]['name']] == concept_index:
                    #     concept_matrix[0][concept_index][obj_index] = 1
        # 1 dim for 'obj local index'
        # concept_matrix[1] = torch.eye(max_obj_num)

        # # 2 dim is for 'balance' concept
        # for obj_index in range(min(max_obj_num, len(ann))):
        #     concept_matrix[2][0][obj_index] = 1
        # print(concept_matrix)

        return concept_matrix

    def _make_relate_matrix(self, mode):
        start_time_relate = time.time()
        count = 0
        # global relate_matrix
        relate_matrix = torch.zeros((max_attribute_num, max_concept_num, max_sleeve_num))  # torch.zeros((10, 60, 60))
        # 0 dim for 'up' relation
        for a_index in range(min(max_obj_num, len(ann))):
            for b_index in range(min(max_obj_num, len(ann))):
                if a_index != b_index:
                    if (a_index == id_a and b_index == id_b) or (a_index == id_b and b_index == id_a):
                        # res_gt = self.predicator.on(ann[a_index], ann[b_index])
                        count = count + 1
                        res = self.predicator.zero_classify(ann[a_index], ann[b_index], name_t)
                        if mode == 'test':
                            # res = res.argmax(dim=1)
                            # relate_matrix[0][a_index][b_index] = res[0][0]
                            if (res[0][0].data >= 0.01):
                                relate_matrix[0][a_index][b_index] = 1 # [a_index][b_index]:a_index索引物体相对于b_index索引物体为up关系
                            else:
                                relate_matrix[0][a_index][b_index] = 0

                        elif mode == 'train' or mode == 'infer':
                            relate_matrix[0][a_index][b_index] = res[0][0]

        # 1 dim for 'down' relation
        for a_index in range(min(max_obj_num, len(ann))):
            for b_index in range(min(max_obj_num, len(ann))):
                if a_index != b_index:
                    relate_matrix[1][a_index][b_index] = relate_matrix[0][b_index][a_index]

        # end_time_relate = time.time()  # 时间计算
        # relate_time = end_time_relate - start_time_relate
        # run_time = str(relate_time)
        # print("relate_time = " + run_time)
        # count1 = str(count)
        # print("数量为" + count1)


        return relate_matrix

    def _get_concept_mask(self, attribute, concept_index):  # (0, concept_index)
        '''

        '''
        return self.concept_matrix[attribute][concept_index]  # 返回的就是这种物体在该图片中出现的情况,如(0, 1, 1)即为图片中共3个物体,第2,3个为此物体
    def _get_concept_obj_mask(self, attribute, concept_index,obj_index):  
        '''

        '''
        return self.concept_matrix[attribute][concept_index][obj_index]

    def _get_relate_mask(self, relate_index):
        return self.relate_matrix[relate_index]
