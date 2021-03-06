#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Author: SinGaln
# @time: 2022/1/11 10:33
import os
import torch
import logging
import numpy as np
from utils import Metrics
from tqdm import tqdm, trange
from model import GlobalPointer
from utils import loss_fun
from torch.utils.data import SequentialSampler, DataLoader
from transformers import BertConfig, AdamW

logger = logging.getLogger(__name__)


class Trainer(object):
    def __init__(self, args, entity_label_lst, train_data=None, dev_data=None):
        self.args = args
        self.train_data = train_data
        self.dev_data = dev_data
        self.entity_label_lst = entity_label_lst
        self.num_labels = len(entity_label_lst)

        # 模型初始化
        self.config = BertConfig.from_pretrained(args.pretrained_model_path)
        self.model = GlobalPointer.from_pretrained(args.pretrained_model_path, config=self.config, args=args,
                                                   num_labels=self.num_labels, RoPE=args.RoPE)

        # 设置GPU or CPU
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if torch.cuda.device_count() > 1:
            self.model = torch.nn.DataParallel(self.model)
        else:
            self.model.to(self.device)

    def train(self, train_dataloader):
        # 只进行预测
        if self.args.do_train:
            logger.info("***** Running evaluation on %s dataset *****")
            logger.info("  Num examples = %d", len(self.train_data))
            logger.info("  Batch size = %d", self.args.eval_batch_size)

            # 优化器和schedule
            optimizer = AdamW(self.model.parameters(), lr=self.args.learning_rate, eps=self.args.adam_epsilon)
            # Training and Evaluate
            tr_loss = 0.
            global_steps = 0.
            self.model.train()
            epoch_iterator = tqdm(train_dataloader, desc="Iteration")
            for step, batch in enumerate(epoch_iterator):
                batch = tuple(t.to(self.device) for t in batch)
                input_ids = batch[0]
                attention_mask = batch[1]
                token_type_ids = batch[2]
                entity_labels_id = batch[3]
                logits = self.model(input_ids=input_ids, attention_mask=attention_mask,
                                          token_type_ids=token_type_ids,
                                          labels_id=entity_labels_id)
                loss = loss_fun(logits, entity_labels_id)
                tr_loss += loss.item()

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                # scheduler.step()

            self.save_model()

    def evaluate(self, model):
        dataset = self.dev_data
        eval_sampler = SequentialSampler(dataset)
        eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=self.args.eval_batch_size)

        # Eval!
        logger.info("***** Running evaluation on %s dataset *****")
        logger.info("  Num examples = %d", len(dataset))
        logger.info("  Batch size = %d", self.args.eval_batch_size)
        eval_loss = 0.0
        nb_eval_steps = 0
        entity_preds = None
        out_entity_labels_ids = None

        model.eval()

        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            batch = tuple(t.to(self.device) for t in batch)
            with torch.no_grad():
                inputs = {'input_ids': batch[0],
                          'attention_mask': batch[1],
                          'token_type_ids': batch[2],
                          'labels_id': batch[3]}
                entity_logits, entity_loss = model(**inputs)

                # entity prediction
                if entity_preds is None:
                    entity_preds = entity_logits.detach().cpu().numpy()
                    out_entity_labels_ids = inputs["labels_id"].detach().cpu().numpy()
                else:
                    entity_preds = np.append(entity_preds, entity_logits.detach().cpu().numpy(), axis=0)
                    out_entity_labels_ids = np.append(out_entity_labels_ids,
                                                      inputs["labels_id"].detach().cpu().numpy(),
                                                      axis=0)
                eval_loss += entity_loss.mean().item()
            nb_eval_steps += 1

        eval_loss = eval_loss / nb_eval_steps
        results = {
            "loss": eval_loss
        }

        # entity result
        entity_preds = np.argmax(entity_preds, axis=2)
        entity_label_map = {i: label for i, label in enumerate(self.entity_label_lst)}
        out_entity_label_list = [[] for _ in range(out_entity_labels_ids.shape[0])]
        entity_preds_list = [[] for _ in range(out_entity_labels_ids.shape[0])]

        for i in range(out_entity_labels_ids.shape[0]):
            for j in range(out_entity_labels_ids.shape[1]):
                if out_entity_labels_ids[i, j] != 0:
                    out_entity_label_list[i].append(entity_label_map[out_entity_labels_ids[i][j]])
                    entity_preds_list[i].append(entity_label_map[entity_preds[i][j]])
        metrics = Metrics()
        total_result = metrics.get_evaluate_fpr(y_pred=entity_preds_list, y_true=out_entity_label_list)
        results.update(total_result)

        logger.info("***** Eval results *****")
        for key in sorted(results.keys()):
            logger.info("  %s = %s", key, str(results[key]))

        return results

    def save_model(self):
        # Save models checkpoint (Overwrite)
        save_path = os.path.join(self.args.save_path, "finetune_bert_model")
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
        model_to_save.save_pretrained(save_path)

        # Save training arguments together with the trained models
        torch.save(self.args, os.path.join(save_path, 'training_args.bin'))
        logger.info("Saving models checkpoint to %s", save_path)
