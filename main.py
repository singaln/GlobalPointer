#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Author: SinGaln
# @time: 2022/1/11 11:11
import os
import argparse
from trainer import Trainer
from utils import init_logger, set_seed, write
from data_loader import EntityProcess, EntityDataset, get_vocab


def main(args):
    init_logger()
    set_seed(args)

    # train_data
    ep = EntityProcess(args=args)
    train_path = os.path.join(args.data_path, "train.txt")
    contents, labels = ep._read_input_file(train_path)
    word2id, tag2id = get_vocab(contents, labels)
    write(args.data_path, word2id, tag2id)
    train_data = ep.get_example(contents, labels, word2id, tag2id)
    entity_label_lst = list(tag2id.keys())
    ner_train_data = EntityDataset(examples=train_data, tag2id=tag2id)

    # dev_data
    dev_path = os.path.join(args.data_path, "train.txt")
    dev_contents, dev_labels = ep._read_input_file(dev_path)
    dev_data = ep.get_example(dev_contents, dev_labels, word2id, tag2id)
    ner_dev_data = EntityDataset(examples=dev_data, tag2id=tag2id)

    trainer = Trainer(args=args, entity_label_lst=entity_label_lst, train_data=ner_train_data, dev_data=ner_dev_data)

    if args.do_train:
        trainer.train(collect_fn=ner_train_data.collect_fn)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_path", type=str, default="./data/", help="The path of data.")
    parser.add_argument("--pretrained_model_path", type=str, default="./pytorch_bert_path", help="The path of pretrained bert models.")
    parser.add_argument("--head_size", type=int, default=64, help="The dimension of each head size.")
    parser.add_argument("--RoPE", action="store_true", help="Whether to enable location encoding.")

    parser.add_argument("--train_batch_size", type=int, default=32, help="The size of train for every batch.")
    parser.add_argument("--eval_batch_size", default=64, type=int, help="Batch size for evaluation.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--num_train_epochs", default=50.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
    parser.add_argument("--dropout_rate", default=0.1, type=float, help="Dropout for fully-connected layers")

    parser.add_argument('--logging_steps', type=int, default=200, help="Log every X updates steps.")
    parser.add_argument('--save_steps', type=int, default=200, help="Save checkpoint every X updates steps.")
    parser.add_argument('--eval_step', type=int, default=200, help="")
    parser.add_argument("--dropout_prob", type=float, default=0.2, help="")
    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--seed", type=int, default=1234, help="The value of set models seed.")
    parser.add_argument("--save_path", type=str, default="./save_path", help="The path of save models.")
    args = parser.parse_args()
    main(args)