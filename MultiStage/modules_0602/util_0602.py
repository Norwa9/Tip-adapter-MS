import torch
from tqdm import tqdm

'''
transformer
loader, if == None,表示一次性测试完整个测试集(可能会爆显存,因此可以输入loader分批测试)
test_features=[50000,1024]
test_prototypes=[50000,topK,1024]
test_zs_weights=[50000,topK,1024]
test_topK_targets=[50000,topK]
test_labels=[500000]
'''
def test_stage2(transformer,loader,test_features,test_prototypes,test_zs_weights,test_topK_targets,test_labels,alpha=None,beta=None):
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

def search():
    pass

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