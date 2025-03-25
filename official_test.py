from datetime import datetime
import random
import torch
from torch.nn.functional import mse_loss
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizer, AdamW, RobertaForSequenceClassification, AutoTokenizer, \
    AutoModelForSequenceClassification, FlaxLlamaForCausalLM, LlamaForSequenceClassification
import pandas as pd
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data.dataset import random_split
from tqdm import tqdm
import os
import argparse
import json
import torch
from peft import PeftModel,PeftModelForTokenClassification
import os
import torch.nn as nn
from peft import AutoPeftModelForCausalLM,AutoPeftModelForSequenceClassification,AutoPeftModelForTokenClassification
from tools.order_metrics import *
LlamaForSequenceClassification
import transformers.models.qwen2
from sklearn.metrics import ndcg_score
import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
 
os.environ["TOKENIZERS_PARALLELISM"] = "false"
 
seed = 0
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

 
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
def NDCG_k(predictions, labels, k=20):
    if len(predictions) < k:
        return -1  # or handle as preferred
    return ndcg_score([labels], [predictions], k=k)
def save_args_to_json(args, file_path):
     
    args_dict = vars(args)
     
    with open(file_path, 'w') as f:
        json.dump(args_dict, f, indent=4)
def get_args():
    parser = argparse.ArgumentParser(description="Train a transformer model with LoRA adaptation on text classification tasks.")
    
    # Dataset and training configuration
    parser.add_argument('--max_length', type=int, default=1024, help='Maximum length of the tokenized input sequences')
    parser.add_argument('--batch_size', type=int, default=20, help='Batch size for training and validation')


    parser.add_argument('--data_path', type=str, default='ScImpactPredict/data/Data_TNCSI_S_OA_AuthorCite_8242_fix1.csv', help='Path to the dataset CSV file')
    parser.add_argument('--checkpoint', type=str, default='llama3_weight', help='Model checkpoint path')
    parser.add_argument('--weight_dir', type=str, default='runs/Jul12_14-54-26_gpu22', help='Model checkpoint path')
    parser.add_argument('--loss_func', type=str, default='bce',choices=['bce','mse','l1'])
    parser.add_argument('--prompt_style', type=int,default=0)
    parser.add_argument('--test_ratio', type=float,default=1.0)
    # parser.add_argument('--model_save_path', type=str, help='Path to save the trained models')
    parser.add_argument('--device', type=str, default='cuda', help='Device to train the model on (cuda or cpu)')


    parser.add_argument('--num_labels', type=int, default=1, help='Number of labels for sequence classification')
    parser.add_argument('--load_in_8bit', type=bool, default=True, help='Whether to load the model in 8-bit for efficiency')
    
    # LoRA configuration
    parser.add_argument('--lora_r', type=int, default=16, help='Rank of LoRA layers')
    parser.add_argument('--lora_alpha', type=int, default=32, help='Expansion factor for LoRA layers')
    parser.add_argument('--lora_dropout', type=float, default=0.05, help='Dropout rate for LoRA layers')
    parser.add_argument('--lora_bias', type=str, default='none', help='Bias mode for LoRA layers')
    parser.add_argument('--target_modules', type=str, default='q_proj,v_proj', help='Comma-separated list of transformer modules to apply LoRA')
    
    
    
     
    default_tb_dir = datetime.now().strftime("%m-%d-%H-%M")
    parser.add_argument('--runs_dir', type=str, default=os.path.join('ScImpactPredict/inference',default_tb_dir), help='Directory for storing TensorBoard logs')

    return parser.parse_args()
from accelerate import Accelerator
accelerator = Accelerator()
args = get_args()
from offcial_train import TextDataset 
args.eff_gpus = int(torch.cuda.device_count() * args.batch_size)

writer = SummaryWriter(args.runs_dir)


tokenizer = AutoTokenizer.from_pretrained(args.weight_dir)
tokenizer.pad_token = tokenizer.eos_token
device_map={'':torch.cuda.current_device()}


 
model = AutoPeftModelForSequenceClassification.from_pretrained(args.weight_dir, num_labels=args.num_labels, load_in_8bit=args.load_in_8bit,device_map=device_map,) 
model.config.pad_token_id = model.config.eos_token_id
model.loss_func = args.loss_func
# model.score.load_state_dict(torch.load(os.path.join(args.weight_dir,'score.pt')))
# model = model.merge_and_unload()
print(model.score.weight)

full_data = pd.read_csv(args.data_path)
dataset = TextDataset(full_data, tokenizer, max_length=args.max_length,prompt_style=args.prompt_style)
test_loader = DataLoader(dataset, batch_size=16, shuffle=True)
print(f'Test Dataloader has {len(test_loader)} samples in total')
total_val_mse = 0.0
total_val_mae = 0.0

all_pred = []
all_GT = []
model.eval()
model, test_loader = accelerator.prepare(model, test_loader)

with torch.no_grad():
    for batch in tqdm(test_loader):
        outputs = model(**batch)
        predictions = outputs["logits"]
        labels = batch["labels"]

        all_GT.append(labels)
        all_pred.append(predictions.squeeze(1))

        mse = nn.MSELoss()(predictions.squeeze(1), labels).item()
        mae = nn.L1Loss()(predictions.squeeze(1), labels).item()

        total_val_mse += mse
        total_val_mae += mae

    avg_mse_loss = total_val_mse / len(test_loader)
    avg_mae_loss = total_val_mae / len(test_loader)


all_pred = torch.cat(all_pred, dim=0)
all_GT = torch.cat(all_GT, dim=0)

all_pred = accelerator.gather_for_metrics(all_pred)
all_GT = accelerator.gather_for_metrics(all_GT)


all_pred_np = all_pred.cpu().numpy()
all_GT_np = all_GT.cpu().numpy()

if accelerator.is_main_process:
    print(args.weight_dir)
    print(f"MSE: {avg_mse_loss:.4f}, MAE: {avg_mae_loss:.4f}")
    print(f"NDCG@20: {NDCG_k(all_pred_np, all_GT_np):.4f}")
    writer.close()

