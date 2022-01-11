#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Author: SinGaln
# @time: 2022/1/11 9:20

"""
    模型搭建：该模型主要的思路就是将实体的类别作为注意力头数，每个头的设为固定维数，最终将hidden_states映射到num_head * head_dim * 2
            这里还加入了RoPE的位置信息，详情见：https://spaces.ac.cn/archives/8373
"""
import torch
import torch.nn as nn
from .utils import loss_fun
from transformers import BertModel, BertPreTrainedModel

class GlobalPointer(BertPreTrainedModel):
    def __init__(self, config, args, num_labels, RoPE=True):
        """
        :param args: 配置参数，如dropout_rate等
        :param config: Bert默认的参数
        :param num_labels: 实体类别数量(注意力头数量)
        :param head_size: 注意力头的维度
        :param RoPE: 是否启用博采众长位置编码(默认为True)
        """
        super(GlobalPointer, self).__init__(config)
        self.args = args
        self.num_labels = num_labels
        self.head_size = args.head_size
        self.hidden_size = config.hidden_size

        self.encoder = BertModel(config=config)
        # 在linear中乘以2的目的和multi-heads-attention的思想一样，便于后续切分为qw,kw,
        # 当然，在multi-heads-attention中是乘以3，因为要切分为query,key,value三个
        self.linear = nn.Linear(self.hidden_size, self.num_labels * self.head_size * 2)

        self.RoPE = RoPE

    def sinusoidal_position_embedding(self, batch_size, seq_len, output_dim):
        position_ids = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(-1)

        indices = torch.arange(0, output_dim // 2, dtype=torch.float)
        indices = torch.pow(10000, -2 * indices / output_dim)
        embeddings = position_ids * indices
        embeddings = torch.stack([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
        embeddings = embeddings.repeat((batch_size, *([1]*len(embeddings.shape))))
        embeddings = torch.reshape(embeddings, (batch_size, seq_len, output_dim))
        return embeddings

    def forward(self, input_ids, attention_mask, token_type_ids, labels_id=None):
        # [batch_size, seq_len, hidden_states]
        sequence_output, _ = self.encoder(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)

        batch_size = sequence_output.size()[0] # batch_size
        seq_length = sequence_output.size()[1] # sequence_length

        # [batch_size, seq_len, num_labels * head_size * 2]
        outputs = self.linear(sequence_output)
        # 按照head_size * 2对最后一维进行切分 [batch_size, seq_len, num_labels, head_size * 2]
        outputs = torch.split(outputs, self.head_size * 2, dim=-1)
        # 按-2维进行拼接,扩展一维
        outputs = torch.stack(outputs, dim=-2)
        # 对输出进行切分为qw,kw [batch_size, seq_len, num_labels, head_size],
        # 类似于multi-heads-attention的处理方式，https://spaces.ac.cn/archives/8373中说的线性变换应该是self.linear操作,
        # 不是对输出进行线性变换，输出按head_size进行切分即可得到qw,kw(本人的理解)
        qw, kw = outputs[..., :self.head_size], outputs[..., self.head_size:]

        # 博采众长旋转位置编码，详情见:https://spaces.ac.cn/archives/8265
        if self.RoPE:
            # [batch_size, seq_len, head_size]
            pos_embed = self.sinusoidal_position_embedding(batch_size=batch_size, seq_len=seq_length, output_dim=self.head_size)
            # cos_pos,sin_pos: (batch_size, seq_len, 1, inner_dim)
            cos_pos = pos_embed[..., None, 1::2].repeat_interleave(2, dim=-1)
            sin_pos = pos_embed[..., None, ::2].repeat_interleave(2, dim=-1)
            qw2 = torch.stack([-qw[..., 1::2], qw[..., ::2]], -1)
            qw2 = qw2.reshape(qw.shape)
            qw = qw * cos_pos + qw2 * sin_pos
            kw2 = torch.stack([-kw[..., 1::2], kw[..., ::2]], -1)
            kw2 = kw2.reshape(kw.shape)
            kw = kw * cos_pos + kw2 * sin_pos

        # [batch_size, num_labels, seq_len, seq_len]
        logits = torch.einsum('bmhd, bnhd->bhmn', qw, kw)

        padding_mask = attention_mask.unsqueeze(1).unsqueeze(1).expand(batch_size, self.num_labels, seq_length, seq_length)
        logits = logits * padding_mask - (1-padding_mask) * 1e12

        # 排除下三角
        mask = torch.tril(torch.ones_like(logits), -1)
        logits = logits - mask * 1e12
        output = (logits,)
        # 计算loss
        if labels_id is not None:
            loss = loss_fun(labels_id, logits)
            output += (loss,)
        return output  # (logits, loss)