import os
import torch

T5_MODEL_PATH = "/root/autodl-tmp/models/t5-small"

SQUAD_PATH = "/root/autodl-tmp/t5-model-reproduce/data/squad"

Config = {
    "output_dir": "output/t5-finetuned",
	"input_max_length":512,
	"output_max_length":512,
    "epoch": 3,
    "batch_size": 32,
    "optimizer": "adam",
    "learning_rate":1e-3,
    "seed":42,
    # "vocab_size":6219,
    # "vocab_path":"vocab.txt",
    # "train_data_path": r"sample_data.json",
    # "valid_data_path": r"sample_data.json",
    "beam_size":5
}

T5_CHECKPOINT = Config['output_dir']



DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
