import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformer import TransformerEncoder


class ProtoTransformer(nn.Module):
    # 用训练集初始化tip_adapter(linear1)
    def __init__(self, args):
        super().__init__()
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

        pass

    # query : [batch, 1024]
    # prototypes : [batch, topK+1, 1024]
    def forward(self, query, prototypes):
        prototypes = prototypes.permute(1,0,2) # [batch, topK+1, 1024] -> [topK+1, batch, 1024]
        out_prototypes = self.SALayers(prototypes).permute(1,0,2) # [batch, topK+1, 1024]
        out_prototypes = F.normalize(out_prototypes,dim=-1)
        logits = ( out_prototypes @ query.unsqueeze(2) ).squeeze(2)
        
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

    query = torch.rand(16,1024)
    prototyes = torch.rand(16,6,1024)
    args = test()
    net = ProtoTransformer(args)
    logits = net(query,prototyes)

    print(logits.shape)
    print("Model parameters:", f"{np.sum([int(np.prod(p.shape)) for p in net.parameters()]):,}")

