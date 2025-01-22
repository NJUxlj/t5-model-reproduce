# t5-model-reproduce
reproduce google's T5 model: "Text-to-Text Transfer Transformer"
![image](https://github.com/user-attachments/assets/5d2886f5-812a-412a-b602-8e5c715676a7)


## 1. T5模型的基本概念
T5是Google在2019年提出的一个革命性的模型，其最大的创新点在于将所有NLP任务都统一转换为"文本到文本"的形式1。这意味着无论是翻译、分类、问答还是摘要，都被转换成相同的文本生成范式。

## 2. 模型架构详解
T5采用了标准的Encoder-Decoder Transformer架构，主要包含以下关键组件：

### 编码器（Encoder）：
- 由多层Transformer块堆叠而成
- 每个块包含自注意力层（Self-Attention）和前馈神经网络（Feed-Forward Network）
- 使用**相对位置编码**，而不是绝对位置编码

### 解码器（Decoder）：
- 同样由多层Transformer块构成
- 包含掩码自注意力层（Masked Self-Attention）
- 包含编码器-解码器注意力层（Encoder-Decoder Attention）
- 也使用相对位置编码


## 3. T5的特殊设计
### 统一的文本到文本框架：
- 所有任务都通过特定前缀来区分，例如：
  - 翻译任务："translate English to German: "
  - 摘要任务："summarize: "
  - 分类任务："classify: "

### 预训练策略：
- 使用"span-corruption"目标
- 随机掩码连续的文本片段
- 使用特殊标记<extra_id_N>来表示被掩码的片段

- ![image](https://github.com/user-attachments/assets/ed4c5e62-b6a5-43cf-aa97-c6fcbc3dc066)




## About this project
- `t5_using_pytorch` contains the code that we manually implement the T5 without using any advanced libraries.
  - (not finished yet)
- `t5_using_transformers` contains the code that we use the transformers library from the huggingface to implement the T5. 
  - (this one is already finished)





## How to Run
### `t5_using_pytorch`
```bash
python main.py
```

### `t5_using_transformers`
- for details of this version, you can refer to the `README.md` in the `t5_using_transformers` folder.
```bash
cd t5_using_transformers
python Trainer.py
python Inference.py
```




## Training Snapshot for the pytorch version





## Results for the pytorch version



