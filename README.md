## GlobalPointer的pytorch版复现

### 目录结构

```apl
> GlobalPointer
	> data ---数据文件夹
	> pytorch_bert_path  ---预训练模型
	> data_loader.py
	> main.py
	> model.py
	> trainer.py
	> utils.py
	
```

#### data_loader

data_loader部分重写为可以读取通常的实体识别标注文本，转为相应的训练格式。

#### 项目运行

```python
python main.py --data_path --do_train
```

...........还未复现完毕，抽时间再写

### 参考

- 科学空间：https://spaces.ac.cn/archives/8373
- 科学空间：https://spaces.ac.cn/archives/7359
- https://github.com/gaohongkui/GlobalPointer_pytorch