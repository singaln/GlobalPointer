#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Author: SinGaln
# @time: 2022/1/11 11:11
import os
import argparse
from .trainer import Trainer
from .utils import init_logger, set_seed, write
from .data_loader import EntityProcess, EntityDataset, get_vocab


def main(args):
    init_logger()
    set_seed(args)

    # train_data
    ep = EntityProcess(args=args)
    train_path = os.path.join(args.data_path, "train.txt")
    contents, labels = ep._read_input_file(train_path)
    tag2id, word2id = get_vocab(contents, labels)
    write(args.data_path, word2id, tag2id)
    train_data = ep.get_example(contents, labels, word2id, tag2id)
    entity_label_lst = list(tag2id.keys())
    ner_train_data = EntityDataset(examples=train_data, tag2id=tag2id)

    # dev_data
    dev_path = os.path.join(args.data_path, "dev.txt")
    dev_contents, dev_labels = ep._read_input_file(dev_path)
    dev_data = ep.get_example(dev_contents, dev_labels, word2id, tag2id)
    ner_dev_data = EntityDataset(examples=dev_data, tag2id=tag2id)

    trainer = Trainer(args=args, entity_label_lst=entity_label_lst, train_data=ner_train_data, dev_data=ner_dev_data)

    if args.do_train:
        trainer.train(collect_fn=ner_train_data.collect_fn)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
