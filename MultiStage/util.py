import torch
import numpy as np
import torch.nn as nn 

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

