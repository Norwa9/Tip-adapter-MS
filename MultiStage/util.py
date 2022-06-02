import torch
import numpy as np
import torch.nn as nn 
from time import time
from datetime import datetime
from datetime import timedelta
from datetime import timezone
import logging
import os

# ----------------------------------粗类别----------------------------------
'''
if the output's top1 class contains target class
@ logits: [batch, class_num]
@ target: [class_num]
@ example:
    batch = 4
    top1_indices = [0, 1, 3, 5]
    target = [0, 1, 1, 1]
    return [True, Ture, False, False]
'''
def isTop1(logits:torch.Tensor, target:torch.Tensor):
    top1_indices = logits.topk(1)[1] # indices
    top1_indices = top1_indices.view(-1) # [batch, 1] -> [batch]
    res = (top1_indices == target)
    return res


'''
if the output's top5 class contains target class
@ logits: [batch, class_num]
@ target: [class_num]
@ example:
    batch = 4
    top5_indices = 
    [
        [0, 1, 3, 5, 7]
        [0, 1, 2, 5, 0]
        [0, 6, 3, 5, 0]
        [4, 1, 4, 4, 9]
    ]
    target = [0, 6, 1, 1]
    return [True, False, False, True]
'''
def isTopK(logits:torch.Tensor, target:torch.Tensor,topk:int):
    topK_indices = logits.topk(topk)[1] # indices
    res = []
    for index,top5 in enumerate(topK_indices):
        res.append(target[index] in top5)
    res = torch.tensor(res)
    return res

'''
属于topK,但是不属于top1的预测类别的下标
'''
def is_corase_class(logits:torch.Tensor, target:torch.Tensor,topk:int = 5):
    logits = logits.to(torch.float32) # 转精度

    is_Top1 = isTop1(logits,target)
    is_TopK = isTopK(logits,target,topk)
    not_Top1 = np.logical_not(is_Top1)
    corase_class_flags = np.logical_and(not_Top1,is_TopK)
    # 返回是粗类别的类别下标
    res = []
    for index,flag in enumerate(corase_class_flags):
        if flag:
            res.append(target[index])
    res = torch.tensor(res)
    return  res

'''
返回值:某样本对应top5最相似的类别'名称'
'''
def find_topk_classes(logits:torch.Tensor, target:torch.Tensor, classNames:list, zeroshot_weights_dict, k:int = 5):
    logits = logits.to(torch.float32) # 转精度
    indices = logits.topk(k)[1] # [batch , k]

    offset = k

    # 1. new target
    new_target = [] # [batch]
    # 对于一个样本，如果其topk中有GT，则将其新标签定义为topK中GT的下标
    # 如果其topK没有GT，那么就在[0,K)中随机赋值一个新标签给该样本 TODO
    for index, topK_i in enumerate(indices):
        target_i = target[index]
        if target_i in topK_i:
            new_target_i = ((topK_i == target_i).nonzero(as_tuple=True)[0]).item()
        else:
            new_target_i = target_i + offset
        new_target.append(new_target_i)

    new_target = torch.tensor(new_target).cuda()
    
    # 2. class embedding  
    class_embeddings = [] # [batch, k, 1024]
    for index, indices_i in enumerate(indices):
        embeddings = []
        for i in indices_i:
            name = classNames[i]
            embeddings.append(zeroshot_weights_dict[name])
        
        
        if new_target[index] >= k: 
            # 1. topK中没有GT
            # 把GT的类的class embedding作为正类
            extra_index = new_target[index] - offset
            new_target[index] = k 
        else: 
            # 2. topK中有GT
            # 随机从TopK的类别以外取一个作为类的class embedding作负类
            while(1):
                extra_index = np.random.randint(0,len(classNames))
                if extra_index not in indices_i:
                    # print(f'index:{index},neg, extra_index:{extra_index}')
                    break
        neg_name = classNames[extra_index]
        embeddings.append(zeroshot_weights_dict[neg_name])
        embeddings = torch.stack(embeddings) # [k+1, 1024]
        class_embeddings.append(embeddings)
    class_embeddings = torch.stack(class_embeddings)


    return class_embeddings, new_target


def find_topk_classes_V2(logits:torch.Tensor, classNames:list, zeroshot_weights_dict, k:int = 5):
    logits = logits.to(torch.float32) # 转精度
    indices = logits.topk(k)[1] # [batch , k]

    class_embeddings = [] 
    for index, indices_i in enumerate(indices):
        embeddings = []
        for i in indices_i:
            name = classNames[i]
            embeddings.append(zeroshot_weights_dict[name])
        embeddings = torch.stack(embeddings)
        class_embeddings.append(embeddings)
    class_embeddings = torch.stack(class_embeddings)


    return class_embeddings # [batch, k, 1024]


'''
topk_indices : [batch, k]
return : [batch, class_num] 
'''
def topK_indices_to_mask(batch_topk_indices:torch.tensor, class_num = 1000, shot = 16):
    batch_size = batch_topk_indices.shape[0]
    masks_class = torch.zeros(batch_size, class_num)

    for index, indices in enumerate(batch_topk_indices):
        for i in indices:
            masks_class[index][i.astype('int64')] = 1
    
    
    masks_sample = masks_class.unsqueeze(2)
    masks_sample = masks_sample.repeat(1,1,shot)
    masks_sample = masks_sample.view(batch_size, -1)

    masks_sample = masks_sample.cuda()
    masks_class = masks_class.cuda()

    return masks_sample, masks_class


def topK_indices_to_mask_V2(logits:torch.tensor, target:torch.tensor, class_num = 1000, topK = 5, shot = 16):
    batch_topk_indices = logits.topk(topK)[1]
    batch_size = batch_topk_indices.shape[0]
    masks_class = torch.zeros(batch_size, class_num)

    for index, indices in enumerate(batch_topk_indices):
        for i in indices:
            masks_class[index][i] = 1
            masks_class[index][target[index]] = 1 
    
    
    masks_sample = masks_class.unsqueeze(2)
    masks_sample = masks_sample.repeat(1,1,shot)
    masks_sample = masks_sample.view(batch_size, -1)


    return masks_sample, masks_class

'''
输入:一个样本的预测logits
输出:其最接近的topK+1个prototypes的下标:
    1.new_target:[batch], topK+1分类的下标.
    2.origin_target:[batch,topK+1], 每个样本肯定包含其target的topK+1类下标
'''
def find_topk_plus_one(logits:torch.Tensor, target:torch.Tensor, topK:int):
    cls_num = logits.shape[1]
    batch = logits.shape[0]

    batch_topk_indices = logits.topk(topK)[1] # [batch , topK]
    
    # 1. new target
    new_target = [] # [batch]  , 取值0~topK
    origin_target = [] # [batch, topK+1] , 取值0~cls_num-1
    # 对于一个样本
    # 如果其topk中有GT，则将其新标签定义为topK中GT的下标，然后再添加一个负prototype
    # 如果其topK没有GT，则将其新标签定义为topK+1，然后添加GT
    for index, topK_i in enumerate(batch_topk_indices):
        origin_target_i = topK_i.cpu().numpy() # [topK]
        target_i = target[index]
        if target_i in topK_i:
            new_target_i = ((topK_i == target_i).nonzero(as_tuple=True)[0]).item() # 0~topK
            # 添加负prototype,选择第topK+1相似的prototype的下标
            topKp1 = logits[index].topk(topK+1)[1][-1]
            neg_index = topKp1.cpu().numpy()
            origin_target_i = np.append(origin_target_i, neg_index)
        else:
            new_target_i = topK # 0~topK
            origin_target_i = np.append(origin_target_i, target_i.cpu().numpy()) # [topK] -> [topK+1]
        
        new_target.append(new_target_i)
        origin_target = np.append(origin_target, origin_target_i)
    
    origin_target = origin_target.reshape(batch,-1)
    new_target = torch.tensor(new_target).cuda()

    return new_target, origin_target
    
'''
输入:
    proto_indices:[batch,topK+1]
    orgin_protos [cls_num, 1024]
    orgin_zeroshot_weights [1024, cls_num]
输出:
    根据proto_indices选取的prototypes: topK_plusone_protos:[batch,topK+1,1024]
'''
def get_topK_plusone_protos(proto_indices, orgin_protos:torch.Tensor, orgin_zeroshot_weights:torch.Tensor,):
    cls_num = orgin_protos.shape[0]
    orgin_zeroshot_weights = orgin_zeroshot_weights.T
    # proto_indices 转one-hot:
    # proto_indices: [batch,topK+1]
    # one_hot_proto_indices: [batch,topK+1, 1000]
    one_hot_proto_indices = torch.zeros(proto_indices.shape[0],proto_indices.shape[1], cls_num)
    for i,indices in enumerate(proto_indices):
        for j,index in enumerate(indices):
            one_hot_proto_indices[i][j][index.astype('int64')] = 1
    # print(one_hot_proto_indices)
    # topK_plusone_protos 
    one_hot_proto_indices = one_hot_proto_indices.cuda()
    topK_plusone_protos = one_hot_proto_indices @ orgin_protos # [batch,topK+1, 1000] @ [1000, 1024] -> [batch,topK+1, 1024]
    topK_plusone_zeroshot_weights = one_hot_proto_indices @ orgin_zeroshot_weights

    return topK_plusone_protos, topK_plusone_zeroshot_weights

def get_topK_ProtosAndZeroshotWeight(logits:torch.Tensor, orgin_protos:torch.Tensor, orgin_zeroshot_weights:torch.Tensor,topK:int):
    indices = logits.topk(topK)[1] # [batch,topK]
    cls_num = orgin_protos.shape[0]
    one_hot_indices = torch.zeros(indices.shape[0],indices.shape[1], cls_num) # [batch, topK, 1000]
    for i,indices in enumerate(indices):
        for j,index in enumerate(indices):
            one_hot_indices[i][j][index] = 1

    one_hot_indices = one_hot_indices.cuda()
    topK_protos = one_hot_indices @ orgin_protos # [batch,topK, 1000] @ [1000, 1024] -> [batch,topK, 1024]
    topK_zeroshot_weights = one_hot_indices @ orgin_zeroshot_weights.T

    return topK_protos,topK_zeroshot_weights

# ----------------------------------------metrics---------------------------------------------
'''
训练阶段使用的准确率算法
'''
def accuracy(output, target, topk=(1,)):
    pred = output.topk(max(topk), 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy()) for k in topk]




# ----------------------------------------other---------------------------------------------






def get_logger(log_dir):
    # set log path
    SHA_TZ = timezone(
        timedelta(hours=8),
        name='Asia/Shanghai',
    )
    utc_now = datetime.utcnow().replace(tzinfo=timezone.utc)
    beijing_now = utc_now.astimezone(SHA_TZ)
    cur_time = beijing_now.strftime('%Y%m%d-%H.%M.%S')
    log_file_dir = f'logs/{log_dir}/{cur_time}'
    make_dir(log_file_dir)
    log_file_path = os.path.join(log_file_dir,'log.log')

    # set logging
    logger = logging.getLogger() 
    logger.setLevel(logging.INFO)

    for ph in logger.handlers:
        logger.removeHandler(ph)
    # add FileHandler to log file
    formatter_file = logging.Formatter('%(asctime)s:%(levelname)s:%(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    fh = logging.FileHandler(log_file_path)
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter_file)
    logger.addHandler(fh)
    # add StreamHandler to terminal outputs
    formatter_stream = logging.Formatter('%(message)s')
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter_stream)
    logger.addHandler(ch)
    return logger

def make_dir(path):

    folder =  os.path.exists(path)
    
    assert not folder
    
    os.makedirs(path)

# ----------------------------------dict----------------------------------
'''
初始化粗类别计数字典
@ class_num: 类别数
'''
def init_corase_class_list(class_num):
    l = torch.zeros(class_num,dtype=torch.long)
    return l

'''
更新粗类别计数字典
@ dict: 粗类别计数字典
@ indices: 粗类别的类别下标数组
'''
def update_corase_class_list(l, indices):
    for index in indices:
        l[index] += 1
    return l

