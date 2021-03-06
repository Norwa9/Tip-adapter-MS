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

        self.linear = nn.Sequential(
            nn.Linear(1024,256),
            nn.ReLU(),
            nn.Linear(256,1024)
        )

    '''
    x : [batch, 1024]
    new_target : [batch], 取值0~topK
    prototypes : [batch, proto_num, 1024]
    zeroshot_weights : [batch, proto_num, 1024]
    '''
    def forward(self, x, prototypes, zeroshot_weights, new_target=None, alpha=None,beta=None):
        if alpha != None:
            self.alpha = alpha
            self.beta = beta
        
        in_feature = (prototypes + zeroshot_weights + x.unsqueeze(1)) / 3 # [batch, proto_num, 1024]
        offset = self.SALayers(in_feature.permute(1,0,2))[0] # [batch, 1024]
        # offset = F.normalize(offset,dim=-1) # [batch,1024]
        offset_prototypes = prototypes + offset.unsqueeze(1)
        offset_prototypes = F.normalize(offset_prototypes,dim=-1)

        sim =  (offset_prototypes @ x.unsqueeze(2)).squeeze(2) # [batch, proto_num, 1024] @ [batch, 1024, 1] = [batch, proto_num, 1] -> [batch, proto_num]
        new_knowledges = ((-1) * (self.alpha - self.alpha * sim)).exp() * self.beta # [batch, proto_num]
        zero_shot_logits = (1. * zeroshot_weights @ x.unsqueeze(2)).squeeze(2)
        logits = new_knowledges + zero_shot_logits
        # logits = new_knowledges
        

        recons_loss = None
        mse_loss = nn.MSELoss()
        if new_target != None:
            # is Training 
            target_proto = prototypes[range(x.shape[0]),new_target] # [batch, 1024]
            recons_proto = offset_prototypes[range(x.shape[0]),new_target] # [batch, 1024]
            recons_loss = mse_loss(target_proto,recons_proto)
        
        return logits, recons_loss

def att_rank(y):
    exp_y = torch.exp(y)
    sum = torch.sum(exp_y,dim=-1).unsqueeze(-1)
    y = exp_y / sum
    return y
    
def one_hot(y, cls_num):
    one_hot_target = torch.zeros((y.shape[0], cls_num))
    for i,target in enumerate(y):
            one_hot_target[i][target] = 1
    return one_hot_target
    
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
    p = torch.rand(16,6,1024)
    zeroshot_weights = torch.rand(16,6,1024)
    args = test()
    net = ProtoTransformer(args)
    logits = net(query,p,zeroshot_weights)

    print(logits.shape)
    print("Model parameters:", f"{np.sum([int(np.prod(p.shape)) for p in net.parameters()]):,}")

