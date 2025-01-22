import torch  
from transformers import (  
    T5ForConditionalGeneration,  
    T5Tokenizer,  
    Trainer,  
    TrainingArguments,  
    DataCollatorForSeq2Seq  
)  
from datasets import load_dataset  
import numpy as np  
from typing import Dict, List  
import evaluate 

import os
import sys
sys.path.append("../")
from config.config import T5_MODEL_PATH, T5_CHECKPOINT




def inference(text: str, model_path: str = T5_CHECKPOINT)->str:  
    '''
    return decoder_output: str
    '''
    # 加载微调后的模型和分词器  
    model = T5ForConditionalGeneration.from_pretrained(model_path)  
    tokenizer:T5Tokenizer = T5Tokenizer.from_pretrained(model_path)  

    # 打印模型路径和文件列表  
    print(f"Loading model from: {model_path}")  
    print("Files in checkpoint directory:", os.listdir(model_path))  
    
    # 对输入文本进行编码  
    # 这里的 max_length=512 仅仅是对输入文本的限制

    # 一步到位，直接获得所需的所有编码信息  
    inputs = tokenizer(
            text, 
            return_tensors="pt", 
            max_length=512, 
            truncation=True,
            padding=True
        )  
    
    # inputs 是一个字典，包含：  
    # {  
    #     'input_ids': tensor([[...]]),  
    #     'attention_mask': tensor([[...]])  
    # }  

    '''
    # 方法2：使用tokenize()（更底层的方式）  
    # 这需要多个步骤  
    tokens = tokenizer.tokenize(text)  # 只得到tokens  
    token_ids = tokenizer.convert_tokens_to_ids(tokens)  # 转换为ids  
    input_ids = torch.tensor([token_ids])  # 转换为tensor  
    
    '''

    # 生成输出  
    outputs = model.generate(  
        inputs["input_ids"],   # shape = (batch_size=1, sequence_length)
        attention_mask=inputs.attention_mask,
        max_length=128,   # 这个参数用于限制生成文本的最大长度
        num_beams=4,  
        length_penalty=2.0,  
        early_stopping=True,
        do_sample=True,  # 启用采样  
        temperature=0.7,  # 控制生成的随机性  
        top_p=0.9,       # 控制采样的概率阈值  
    )   # shape = (1, 1 ,128) = (batch_size=1, num_return_sequences=1, sequence_length=128)
    
    # 解码输出  
    decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)  
    
    return decoded_output  





if __name__ == '__main__':
    # output = inference("User: who are you? Agent:")

    # print("\n",output)

    context = '''Architecturally, the school has a Catholic character. 
            Atop the Main Building's gold dome is a golden statue of the Virgin Mary. 
            Immediately in front of the Main Building and facing it, is a copper statue of Christ 
            with arms upraised with the legend "Venite Ad Me Omnes". Next to the Main Building is
            the Basilica of the Sacred Heart. Immediately behind the basilica is the Grotto, 
            a Marian place of prayer and reflection.'''
    question  = "To whom did the Virgin Mary allegedly appear in 1858 in Lourdes France?"
    test_inputs = [  
        f"Question:{question} Context:{context}",  
    ]  

    for test_input in test_inputs:  
        output = inference(test_input)  
        print(f"\nInput:\n {test_input}")  
        print(f"Output:\n {output}")  
