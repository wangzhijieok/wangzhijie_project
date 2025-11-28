import copy
import math
import torch
import torch.nn as nn

def attention(query, key, value, mask=None, dropout=None):
    # 将query矩阵的最后一个维度值作为d_k
    d_k = query.size(-1)

    # 将key的最后两个维度互换（转置），才能与query矩阵相乘
    # 完成了矩阵乘法后再除以d_k的平方根，实现缩放点积注意力（Scaled Dot-Product Attention）
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

    # 如果存在mask的内容，则将那些为0的位置替换成一个很大的负数（使softmax后趋近于0）
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)

    # 对mask后的attention矩阵按最后一个维度进行softmax
    p_attn = F.softmax(scores, dim=-1)

    # 如果dropout参数非空，则对注意力分布进行dropout操作
    if dropout is not None:
        p_attn = dropout(p_attn)

    # 返回加权后的value矩阵（注意力结果），以及注意力权重矩阵
    return torch.matmul(p_attn, value), p_attn



def clones(module: nn.Module, N: int) -> nn.ModuleList:
    """
    复制同一个子模块N份（参数互不共享），并放入ModuleList中。
    用于Transformer里重复的线性层/子层等。
    """
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        # 确保可整除
        assert d_model % h == 0
        # 得到一个head的attention表示维度
        self.d_k = d_model // h
        # head数量
        self.h = h
        # 定义4个全连接层参数，供后面作为WQ, WK, WV矩阵和最后一个多头注意力矩阵concat之后进行变换的矩阵
        self.linears = clones(nn.Linear(d_model, d_model), N=4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    # qkv可以来自不同层
    def forward(self, query, key, value, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)
        # query的第一个维度应为batch size
        nbatches = query.size(0)

        # 将embedding层乘以WQ, WK, WV矩阵（均为全连接）
        # 并将结果拆成h头，然后将第二个和第三个维度互换（具体过程见上述解析）
        query, key, value = [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linears, (query, key, value))]

        # 调用上面定义的attention函数计算每个head的注意力矩阵与value的乘积，以及注意力矩阵
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)

        # 将多个头注意力的结果concat起来（注意要先把h维放回到第三维的位置）
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)

        # 使用self.linears中的最后一个全连接函数做变换并返回结果
        return self.linears[-1](x)
