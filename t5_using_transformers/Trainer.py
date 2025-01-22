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


from Models import get_model_and_tokenizer
from Preprocess import DataProcessor
from Evaluation import compute_metrics


import sys
sys.path.append("../")
from config.config import T5_MODEL_PATH, Config

from Preprocess import prepare_squad_dataset





def get_training_args(  
    output_dir: str = Config['output_dir'],  
    num_train_epochs: int = 3,  
    per_device_train_batch_size: int = 8,  
    per_device_eval_batch_size: int = 8,  
    warmup_steps: int = 500,  
    weight_decay: float = 0.01,  
    logging_steps: int = 100,  
    evaluation_strategy: str = "steps",  
    eval_steps: int = 500,  
    save_steps: int = 1000,  
    gradient_accumulation_steps: int = 2  
):  
    return TrainingArguments(  
        output_dir=output_dir,  
        num_train_epochs=num_train_epochs,  
        per_device_train_batch_size=per_device_train_batch_size,  
        per_device_eval_batch_size=per_device_eval_batch_size,  
        warmup_steps=warmup_steps,  
        weight_decay=weight_decay,  
        logging_steps=logging_steps,  
        evaluation_strategy=evaluation_strategy,  
        eval_steps=eval_steps,  
        save_steps=save_steps,  
        gradient_accumulation_steps=gradient_accumulation_steps,  
        load_best_model_at_end=True,  
        metric_for_best_model="eval_loss"  
    )  






def train():  
    # 1. 加载模型和分词器  
    model, tokenizer = get_model_and_tokenizer()  
    
    # 2. 加载数据集（需要根据实际任务修改）  
    processed_datasets = prepare_squad_dataset()
    
    # 4. 准备训练参数  
    training_args = get_training_args()  
    
    # 5. 准备数据整理器  
    data_collator = DataCollatorForSeq2Seq(  
        tokenizer=tokenizer,  
        model=model,  
        padding=True  
    )  
    
    # 6. 初始化Trainer  
    trainer = Trainer(  
        model=model,  
        args=training_args,  
        train_dataset=processed_datasets["train"],  
        eval_dataset=processed_datasets["validation"],  
        data_collator=data_collator,  
        tokenizer=tokenizer,  
        compute_metrics=compute_metrics  
    )  
    
    # 7. 开始训练  
    trainer.train()  
    
    # 8. 保存模型  
    trainer.save_model()  
    
if __name__ == "__main__":  
    train()  