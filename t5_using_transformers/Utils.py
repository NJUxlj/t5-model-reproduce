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


from Inference import inference



def check_tokenizer_details(model_path):  
    '''
    检查模型的词表和特殊标记
    '''
    tokenizer = T5Tokenizer.from_pretrained(model_path)  
    
    # 检查词表大小  
    print(f"Vocabulary size: {len(tokenizer)}")  
    
    # 检查特殊标记  
    print(f"Special tokens: {tokenizer.all_special_tokens}")  
    
    # 测试tokenization  
    test_input = "User: Who are you? Agent:"  
    tokens = tokenizer.tokenize(test_input)  
    token_ids = tokenizer.encode(test_input)  
    
    print(f"\nTest input: {test_input}")  
    print(f"Tokenized: {tokens}")  
    print(f"Token IDs: {token_ids}")  
    
    # 检查是否有padding token  
    print(f"\nPadding token: {tokenizer.pad_token}")  
    print(f"EOS token: {tokenizer.eos_token}")  
    print(f"BOS token: {tokenizer.bos_token if hasattr(tokenizer, 'bos_token') else 'No BOS token'}")





def check_model_outputs(model_path):  
    '''
    检查模型的输出logits
    '''
    model = T5ForConditionalGeneration.from_pretrained(model_path)  
    tokenizer = T5Tokenizer.from_pretrained(model_path)  
    
    input_text = "User: Who are you? Agent:"  
    inputs = tokenizer(input_text, return_tensors="pt", padding=True)  

    # 创建decoder_input_ids  
    decoder_input_ids = torch.tensor([[tokenizer.pad_token_id]])  
    
    # 获取模型的原始输出（不使用generate）  
    with torch.no_grad():  
        outputs = model(
            input_ids=inputs["input_ids"],  
            attention_mask=inputs["attention_mask"],  
            decoder_input_ids=decoder_input_ids  
        )  
    
    print(f"Output logits shape: {outputs.logits.shape}")  
    
    # 检查前5个最高概率的token  
    first_token_logits = outputs.logits[0, 0]  
    top_tokens = torch.topk(first_token_logits, 5)  
    
    print("\nTop 5 predicted tokens for first position:")  
    for score, idx in zip(top_tokens.values, top_tokens.indices):  
        token = tokenizer.decode([idx])  
        print(f"Token: {token}, Score: {score:.2f}")

    
    # 测试实际生成  
    print("\nTesting generation:")  
    generated_ids = model.generate(  
        inputs["input_ids"],  
        attention_mask=inputs["attention_mask"],  
        max_length=50,  
        num_beams=4,  
        do_sample=True,  
        temperature=0.7,  
        top_p=0.9,  
        no_repeat_ngram_size=2  
    )  
    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)  
    print(f"Generated text: {generated_text}")  




def check_checkpoint(checkpoint_path):  
    '''
    检查训练检查点
    '''
    import os  
    
    checkpoint_dir = checkpoint_path
    if os.path.exists(checkpoint_dir):  
        print(f"Found checkpoint directory: {checkpoint_dir}")  
        files = os.listdir(checkpoint_dir)  
        print(f"Files in checkpoint: {files}")  
        
        # 检查训练参数  
        if 'trainer_state.json' in files:  
            import json  
            with open(os.path.join(checkpoint_dir, 'trainer_state.json'), 'r') as f:  
                state = json.load(f)  
                print("\nTraining state:")  
                print(f"Best metric: {state.get('best_metric', 'Not found')}")  
                print(f"Number of training steps: {state.get('global_step', 'Not found')}")  
                print(f"Training loss: {state.get('log_history', [])[-1] if state.get('log_history') else 'Not found'}")










if __name__ == '__main__':  
    model_path = "output/t5-finetuned"  
    
    print("\n=== Checking Tokenizer Details ===")  
    check_tokenizer_details(model_path)  
    
    print("\n=== Checking Model Outputs ===")  
    check_model_outputs(model_path)  
    
    print("\n=== Checking Checkpoint ===")  
    check_checkpoint(model_path)  
    
    # 原有的推理代码  
    output = inference("User: who are you? Agent:")  
    print("\nInference output:", output)