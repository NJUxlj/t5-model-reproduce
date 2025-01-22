import torch  
from transformers import (  
    T5ForConditionalGeneration,  
    T5Tokenizer,  
    Trainer,  
    TrainingArguments,  
    DataCollatorForSeq2Seq  
)  
from datasets import load_dataset, Dataset 
import numpy as np  
from typing import Dict, List  
import evaluate  
from dataclasses import dataclass
from typing import List, Dict, Any, Optional  

import sys
sys.path.append('/root/autodl-tmp/t5-model-reproduce')
from config.config import T5_MODEL_PATH, SQUAD_PATH, Config, DEVICE

# 设置随机种子  
def set_seed(seed: int = 42):  
    np.random.seed(seed)  
    torch.manual_seed(seed)  
    torch.cuda.manual_seed(seed)  
    
set_seed()  



@dataclass
class DataProcessor:  
    """基础数据处理类  
    
    Attributes:  
        tokenizer: 分词器对象  
        max_source_length (int): 源文本最大长度，默认512  
        max_target_length (int): 目标文本最大长度，默认128  
    """  
    tokenizer: Any  # tokenizer类型根据实际使用的模型库定义  
    max_source_length: int = 512  
    max_target_length: int = 128  

    # def __init__(self, tokenizer, max_source_length: int = 512, max_target_length: int = 128):  
    #     self.tokenizer = tokenizer  
    #     self.max_source_length = max_source_length  
    #     self.max_target_length = max_target_length  
        
    def preprocess_function(self, examples:Dict[str,List])->Dict[str, List]:
        """处理输入数据的基础方法  
        
        Args:  
            examples: 包含source_text和target_text的字典  
            
        Returns:  
            包含处理后的input_ids和labels的字典  
        """    
        # 根据实际任务修改输入输出字段名  
        inputs = examples["source_text"]  
        targets = examples["target_text"]  
        
        # 对输入文本进行tokenize 
        model_inputs = self.tokenizer(  
            inputs,  
            max_length=self.max_source_length,  
            padding="max_length",  
            truncation=True,  
        )  
        
        # 对目标文本进行tokenize 
        labels = self.tokenizer(  
            targets,  
            max_length=self.max_target_length,  
            padding="max_length",  
            truncation=True,  
        )  
        
        model_inputs["labels"] = labels["input_ids"]  
        return model_inputs  




class SquadProcessor(DataProcessor):  
    def __init__(  
        self,   
        tokenizer,   
        max_source_length: int = 512,   
        max_target_length: int = 128,  
        prefix: str = "question: "  
    ):  
        # 如果不写这一行，就等于重写了父类的方法
        super().__init__(tokenizer, max_source_length, max_target_length) 
        
        # self.tokenizer = tokenizer  
        # self.max_source_length = max_source_length  
        # self.max_target_length = max_target_length   

        self.prefix = prefix  
        
    def preprocess_function(self, examples:Dict[str, List])->Dict[str,List]:  
        """处理SQuAD格式数据的方法  
        
        Args:  
            examples: 包含question、context和answers的字典  
            
        Returns:  
            包含处理后的input_ids和labels的字典  
        """  
        # 组合问题和上下文  
        questions = examples['question']  
        contexts = examples['context']  
        
        # 将问题和上下文组合成输入格式  
        # 格式: "question: {question} context: {context}"  
        inputs: List[str] = [  
            f"{self.prefix}{question} context: {context}"   
            for question, context in zip(questions, contexts)  
        ]  
        
        # 获取答案（SQuAD数据集中answers是一个包含多个可能答案的列表，我们取第一个）  
        targets = [  
            answers['text'][0] if answers['text'] else ""  
            for answers in examples['answers']  
        ]  
        
        # Tokenize输入  
        model_inputs = self.tokenizer(  
            inputs,  
            max_length=self.max_source_length,  
            padding="max_length",  
            truncation=True,  
        )  
        
        # Tokenize目标答案  
        labels = self.tokenizer(  
            targets,  
            max_length=self.max_target_length,  
            padding="max_length",  
            truncation=True,  
        )  
        
        model_inputs["labels"] = labels["input_ids"]  
        
        return model_inputs  


def prepare_squad_dataset()->Dataset:  
    from datasets import load_dataset  
    
    # 加载SQuAD数据集  
    dataset = load_dataset(SQUAD_PATH)  
    
    # 初始化tokenizer  
    tokenizer = T5Tokenizer.from_pretrained(T5_MODEL_PATH)  
    
    # 创建数据处理器  
    processor = SquadProcessor(tokenizer)  
    
    # 处理数据集  
    processed_datasets = dataset.map(  
        processor.preprocess_function,  
        batched=True,  
        remove_columns=dataset["train"].column_names,  
        desc="Processing dataset",  
    )  

    processed_datasets['train'] = processed_datasets['train'].select(range(2000))
    processed_datasets['validation'] = processed_datasets['validation'].select(range(1000))
    # processed_datasets['test'] = processed_datasets['test'].select(range(1000))
    
    return processed_datasets






# test 
if __name__ == '__main__':
    prepare_squad_dataset()
    print('done')
    print(prepare_squad_dataset()['train'][0])
    # print(prepare_squad_dataset()['train'][1])
    # print(prepare_squad_dataset()['train'][2])
    # print(prepare_squad_dataset()['train'][3])