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




def inference(text: str, model_path: str = "t5-finetuned"):  
    # 加载微调后的模型和分词器  
    model = T5ForConditionalGeneration.from_pretrained(model_path)  
    tokenizer = T5Tokenizer.from_pretrained(model_path)  
    
    # 对输入文本进行编码  
    inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True)  
    
    # 生成输出  
    outputs = model.generate(  
        inputs.input_ids,  
        max_length=128,  
        num_beams=4,  
        length_penalty=2.0,  
        early_stopping=True  
    )  
    
    # 解码输出  
    decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)  
    
    return decoded_output  





if __name__ == '__main__':
    inference()
