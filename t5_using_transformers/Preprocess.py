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

# 设置随机种子  
def set_seed(seed: int = 42):  
    np.random.seed(seed)  
    torch.manual_seed(seed)  
    torch.cuda.manual_seed(seed)  
    
set_seed()  




class DataProcessor:  
    def __init__(self, tokenizer, max_source_length: int = 512, max_target_length: int = 128):  
        self.tokenizer = tokenizer  
        self.max_source_length = max_source_length  
        self.max_target_length = max_target_length  
        
    def preprocess_function(self, examples):  
        # 根据实际任务修改输入输出字段名  
        inputs = examples["source_text"]  
        targets = examples["target_text"]  
        
        model_inputs = self.tokenizer(  
            inputs,  
            max_length=self.max_source_length,  
            padding="max_length",  
            truncation=True,  
        )  
        
        # 设置目标文本的tokenization  
        labels = self.tokenizer(  
            targets,  
            max_length=self.max_target_length,  
            padding="max_length",  
            truncation=True,  
        )  
        
        model_inputs["labels"] = labels["input_ids"]  
        return model_inputs  