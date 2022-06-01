import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from modules.transformer import TransformerEncoder


class ProtoTransformer(nn.Module):
    # 用训练集初始化tip_adapter(linear1)
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.alpha = args.transformer_alpha
        self.beta = args.transformer_beta
        self.embed_dim  = args.embed_dim
        self.num_heads = args.num_heads
        self.layers = args.layers
        self.attn_dropout = args.attn_dropout
        self.relu_dropout = args.relu_dropout 
        self.res_dropout = args.res_dropout
        self.embed_dropout = args.embed_dropout
        self.attn_mask = args.attn_mask

        self.SALayers = TransformerEncoder(embed_dim=self.embed_dim,
                                  num_heads=self.num_heads,
                                  layers=self.layers,
                                  attn_dropout=self.attn_dropout,
                                  relu_dropout=self.relu_dropout,
                                  res_dropout=self.res_dropout,
                                  embed_dropout=self.embed_dropout,
                                  attn_mask=self.attn_mask)

    # x : [batch, 1024]
    # new_target : [batch], 取值0~topK
    # prototypes : [batch, topK+1, 1024]
    # zeroshot_weights : [batch, topK+1, 1024]
    def forward(self, x, prototypes, zeroshot_weights, alpha=None,beta=None):
        if alpha: # search 阶段
            self.alpha = alpha
            self.beta = beta
        '''
        1.输入prototypes经过自注意力交互, 取第一个输出当做GT的偏移量
        '''
        prototypes = prototypes.permute(1,0,2) # [batch, topK+1, 1024] -> [topK+1, batch, 1024]
        batch_offset = self.SALayers(prototypes)[0] # [batch, 1024] , 表示每个样本的GT prototype应该加上的偏移量
        batch_offset = F.normalize(batch_offset, dim=-1)

        '''
        2.计算logits
        将GT 对应的prototype加上offset,得到新protots后,输入x与它们计算相似度输出预测概率分布
        类似adapter,预测概率分布由直接点乘的logits和zeroshot prompts的logits两部分组成
        '''
        prototypes = prototypes.permute(1,0,2)
        batch_offset = batch_offset.unsqueeze(1)
        prototypes = prototypes + batch_offset
        prototypes = F.normalize(prototypes, dim=-1)
        x = x.unsqueeze(2) # [batch, 1024] -> [batch, 1024, 1]
        sim =  (prototypes @ x ).squeeze(2) # [batch, topK+1, 1024] @ [batch, 1024, 1] = [batch, topK+1, 1]
        new_knowledges = ((-1) * (self.alpha - self.alpha * sim)).exp() * self.beta # [batch, topK+1]
        zero_shot_logits = (zeroshot_weights @ x ).squeeze(2)   #  [batch,topK+1,1024] @ [batch,1024,1]  =  [batch,topK+1,1]
        logits = new_knowledges + zero_shot_logits # [batch,topK+1]
        # TODO 去掉 zero_shot_logits
        
        return logits
    


    
if __name__ == '__main__':
    class test:
        embed_dim  = 1024
        num_heads = 4
        layers = 1
        attn_dropout = 0.1
        relu_dropout = 0.1
        res_dropout = 0.1
        embed_dropout = 0.1
        attn_mask = True
        alpha = 1.0
        beta = 1.0

    query = torch.rand(16,1024)
    prototyes = torch.rand(16,6,1024)
    zeroshot_weights = torch.rand(16,6,1024)
    args = test()
    net = ProtoTransformer(args)
    logits = net(query,prototyes,zeroshot_weights)

    print(logits.shape)
    print("Model parameters:", f"{np.sum([int(np.prod(p.shape)) for p in net.parameters()]):,}")

