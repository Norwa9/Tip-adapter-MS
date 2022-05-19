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
        
        # add neg embedding
        if new_target[index] >= k: # 说明topK中没有GT
            # 获取GT的类别下标
            extra_index = new_target[index] - offset
            # print(f'index:{index},pos, extra_index:{extra_index}')
            new_target[index] = k # 把GT作为第k+1个class embedding
        else:
            # 随机从TopK以外的下标中取一个作为负类样本
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

