import torch
from tqdm import tqdm
import numpy as np
import random
from util import *
'''
transformer
loader, if == None,表示一次性测试完整个测试集(可能会爆显存,因此可以输入loader分批测试)
test_features=[50000,1024]
test_prototypes=[50000,topK,1024]
test_zs_weights=[50000,topK,1024]
test_topK_targets=[50000,topK]
test_labels=[500000]
'''
def test_stage2_offline(transformer,loader,test_features,test_prototypes,test_zs_weights,test_topK_targets,test_labels,alpha=None,beta=None):
    top1, n = 0., 0.
    if loader == None:
        with torch.no_grad():
            
            new_logits = transformer(test_features, test_prototypes, test_zs_weights, alpha, beta)
            
            '''test_topK_targets:[batch,topK],test_labels:[batch],如果new_logits中预测的top1是test_labels,则算预测正确'''
            acc1_num = accuracy_test(new_logits, test_topK_targets, test_labels) # new_logits=[batch, proto_num], target=[batch,cls_num]
            top1 = acc1_num
            n = test_features.shape[0]
            top1 = (top1 / n) * 100
    else:
        with torch.no_grad():
            for i, (images, target) in enumerate(tqdm(loader)):
                batch = images.shape[0]
                batch_target = target.cuda()
                batch_test_features = test_features[i*batch:(i+1)*batch]
                batch_test_topK_targets = test_topK_targets[i*batch:(i+1)*batch] # [batch, topK] , 表示每个测试样本提取的topK的标签
                topK_protos, topK_zeroshot_weights = test_prototypes[i*batch:(i+1)*batch], test_zs_weights[i*batch:(i+1)*batch]
                
                new_logits = transformer(batch_test_features, topK_protos, topK_zeroshot_weights, alpha, beta)
                
                acc1_num = accuracy_test(new_logits, batch_test_topK_targets, batch_target) # new_logits=[batch, proto_num], target=[batch,cls_num]
                top1 += acc1_num
                n += batch
            top1 = (top1 / n) * 100
    return top1

'''
online:
从数据集的图片过一遍第一阶段的adapter,然后提取topK个prototypes和zs weight 进行第二阶段
与offline的区别在于prototypes和zs weight是否有存储准备好
'''
def test_stage2_online(args, transformer,adapter, model, test_loader, alpha=None,beta=None):
    correct_all,n = 0,0
    for i, (images, target) in enumerate(tqdm(test_loader)):
        images = images.cuda()
        target = target.cuda()
        with torch.no_grad():
            image_features = model.encode_image(images)
            image_features /= image_features.norm(dim=-1, keepdim=True) # [batch, image_dim]

        logits, _ = adapter(image_features, target)
        topK_protos, topK_zeroshot_weights = get_topK_ProtosAndZeroshotWeight(logits, adapter.proto, adapter.zero_shots_weight, args.topK)
        new_logits = transformer(image_features,topK_protos, topK_zeroshot_weights, alpha, beta)
        new_target = logits.topk(args.topK)[1]
        
        correct = accuracy_test(new_logits, new_target, target)
        correct_all += correct[0]
        n += len(new_logits)
    top1  = (correct / n) * 100
    return top1
'''
输入:
    1.一个样本的预测logits
    2.该样本的标签target
    3.需要选取多少个原型topK
输出:
    包含GT prototype和topK个random prototypes的下标:
    1.new_target:[batch], topK+1分类的下标.
    2.origin_target:[batch,topK+1], 每个样本肯定包含其target的topK+1类下标
'''
def sapmle_topk1_prototypes(logits:torch.Tensor, targets:torch.Tensor, topK:int):
    cls_num = logits.shape[1]
    batch = logits.shape[0]

    origin_target = []
    new_target = torch.zeros(batch,1).view(-1).to(torch.long)
    for i, target in enumerate(targets):
        random_targets = random.sample(range(cls_num),topK+1)
        if target in random_targets:
            new_target[i] = random_targets.index(target)
        else:
            random_targets[0] = target.cpu().numpy() # tensor->numpy
        origin_target =  np.append(origin_target, random_targets)
    
    origin_target = origin_target.reshape(batch,-1)
    new_target = new_target.cuda()

    return new_target, origin_target


# ---------------------------- metrics ----------------------------
def one_hot(y, cls_num):
    one_hot_target = torch.zeros((y.shape[0], cls_num))
    for i,target in enumerate(y):
            one_hot_target[i][target] = 1
    return one_hot_target


'''
测试阶段使用的准确率算法,计算一个batch中top1正确个数
test_logits:[batch,topK]
test_topK_targets:[batch,topK],里面不一定有正确的测试标签
test_target=[batch],测试标签
'''
def accuracy_test(test_logits:torch.Tensor, test_topK_targets,test_target):
    acc1 = 0.
    top1_indices = test_logits.topk(1)[1]
    for i,target in enumerate(test_target):
        j = top1_indices[i][0]
        if target == test_topK_targets[i][j]:
            acc1 += 1
        
    return acc1