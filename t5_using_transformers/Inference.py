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

import sys
sys.path.append("../")
from config.config import T5_MODEL_PATH




def inference(text: str, model_path: str = T5_MODEL_PATH)->str:  
    '''
    return decoder_output: str
    '''
    # 加载微调后的模型和分词器  
    model = T5ForConditionalGeneration.from_pretrained(model_path)  
    tokenizer:T5Tokenizer = T5Tokenizer.from_pretrained(model_path)  
    
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
        early_stopping=True  
    )   # shape = (1, 1 ,128) = (batch_size=1, num_return_sequences=1, sequence_length=128)
    
    # 解码输出  
    decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)  
    
    return decoded_output  





if __name__ == '__main__':
    output = inference("User: who are you? Agent:")

    print("\n",output)
