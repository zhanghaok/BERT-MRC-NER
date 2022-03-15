## 基于BERT-MRC的命名实体识别模型

### 原始论文

见paper文件夹

### 原理

原始论文：https://arxiv.org/pdf/1910.11476v6.pdf

原始论文代码：[ShannonAI/mrc-for-flat-nested-ner: Code for ACL 2020 paper `A Unified MRC Framework for Named Entity Recognition` (github.com)](https://github.com/ShannonAI/mrc-for-flat-nested-ner)



在输入文本前加上了实体类型的描述信息，这些实体类型的描述作为先验知识提高了模型抽取的效果，所以BERT-MRC模型在数据量匮乏的场景下，通过在输入文本前面拼接的query获得了一定的先验信息，提升了性能。



BERT-MRC模型是目前实体识别领域的一个SOTA模型，在数据量较小的情况下效果较其他模型要更好，原因是因为BERT-MRC模型可以通过问题加入一些先验知识，减小由于数据量太小带来的问题，在实际实验中，在数据量比较小的情况下，BERT-MRC模型的效果要较其他模型要更好一点。BERT-MRC模型很适合在缺乏标注数据的场景下使用。

### 模型细节

参考：[ 命名实体识别Baseline模型BERT-MRC总结](https://blog.csdn.net/eagleuniversityeye/article/details/109601547)



### 运行

重新训练 :

```shell
nohup python train.py &
```



直接eval（首先下载已经训练好的模型saved_model文件夹有说明）：

```bash
nohup python eval1.py &
```

### 日志

log文件夹有训练和测试的日志记录
