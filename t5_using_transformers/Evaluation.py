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
from config.config import T5_MODEL_PATH, Config



def compute_metrics(eval_pred):  
    metric = evaluate.load("rouge")  
    predictions, labels = eval_pred  # shape = Tuple[LongTensor[batch_size, seq_len], LongTensor]
    
    # 解码预测结果和标签  
    tokenizer:T5Tokenizer = T5Tokenizer.from_pretrained(T5_MODEL_PATH)  
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)  
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)  
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)  
    
    # 计算ROUGE分数  
    result = metric.compute(predictions=decoded_preds, references=decoded_labels)  
    
    return {  
        "rouge1": result["rouge1"],  
        "rouge2": result["rouge2"],  
        "rougeL": result["rougeL"],  
    }  