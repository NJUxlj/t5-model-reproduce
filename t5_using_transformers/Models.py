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

from config.config import Config, T5_MODEL_PATH




def get_model_and_tokenizer(model_name: str = T5_MODEL_PATH):  
    # 加载预训练模型和分词器  
    tokenizer = T5Tokenizer.from_pretrained(model_name)  
    model = T5ForConditionalGeneration.from_pretrained(model_name)  
    return model, tokenizer  