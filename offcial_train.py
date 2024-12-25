from sklearn.metrics import ndcg_score
from transformers import Trainer, TrainingArguments
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForSequenceClassification, FlaxLlamaForCausalLM
import pandas as pd
from torch.utils.data.dataset import random_split
import argparse
import json
from accelerate import Accelerator
import os
import torch.nn as nn

from NAID.dataset import TextDataset

os.environ["TOKENIZERS_PARALLELISM"] = "false"
import transformers.models.qwen2
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType

accelerator = Accelerator()


def NDCG_k(predictions, labels, k=20):
    if len(predictions) < k:
        return -1  # or handle as preferred
    return ndcg_score([labels], [predictions], k=k)
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = torch.tensor(predictions).squeeze()
    labels = torch.tensor(labels).squeeze()
    mse = nn.MSELoss()(predictions, labels).item()
    mae = nn.L1Loss()(predictions, labels).item()
    # Convert tensors to numpy arrays for NDCG computation
    predictions = predictions.detach().cpu().numpy()
    labels = labels.detach().cpu().numpy()
    # Calculate NDCG
    ndcg = NDCG_k(predictions, labels)
    return {"mse": mse, "mae": mae, "ndcg": ndcg}


def save_args_to_json(args, file_path):
    args_dict = vars(args)
    with open(file_path, 'w') as f:
        json.dump(args_dict, f, indent=4)


def main(args):
    args.eff_gpus = int(torch.cuda.device_count())
    args.eff_batch_size = args.eff_gpus * args.batch_size

    if args.learning_rate is None:  # only base_lr is specified
        args.learning_rate = args.base_lr * args.eff_batch_size / 256

    # Load your dataset
    df = pd.read_csv(args.data_path)
    df_test = pd.read_csv(args.test_data_path)

    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)

    device_map = {'': torch.cuda.current_device()}
    model = AutoModelForSequenceClassification.from_pretrained(
        args.checkpoint,
        num_labels=args.num_labels,
        load_in_8bit=args.load_in_8bit,
        device_map=device_map,
    )

    model.config.pad_token_id = model.config.eos_token_id
    tokenizer.pad_token = tokenizer.eos_token
    model.loss_func = args.loss_func
    if len(args.target_modules) > 0:
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=args.target_modules.split(','),
            task_type=TaskType.SEQ_CLS,
            inference_mode=False
        )

        model = prepare_model_for_kbit_training(model)
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    else:
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            task_type=TaskType.SEQ_CLS,
            inference_mode=False
        )

        model = prepare_model_for_kbit_training(model)
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    total_dataset = TextDataset(df, tokenizer, args.max_length, args.prompt_style)
    total_size = len(total_dataset)
    train_size = int(0.9 * total_size)
    val_size = total_size - train_size
    train_dataset, val_dataset = random_split(total_dataset, [train_size, val_size])
    test_dataset = TextDataset(df_test, tokenizer, args.max_length) # DO NOT USE FOR PARAMETER SEARCHING

    # Prepare Accelerator
    accelerator = Accelerator()
    if accelerator.is_local_main_process:
        default_tb_dir = datetime.now().strftime("%m-%d-%H-%M-%s")
        if args.runs_dir is None:
            args.runs_dir = os.path.join('official_runs', default_tb_dir)
        os.makedirs(args.runs_dir, exist_ok=True)
        json_file_path = os.path.join(args.runs_dir, 'args.json')
        save_args_to_json(args, json_file_path)

    # Define training arguments
    training_args = TrainingArguments(
        ddp_find_unused_parameters=False,
        output_dir=args.runs_dir,
        learning_rate=args.learning_rate,
        num_train_epochs=args.total_epochs,
        logging_dir=args.runs_dir,
        logging_steps=10,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        weight_decay=args.weight_decay,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        warmup_ratio=args.warmup_ratio,
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    # Train model
    model, tokenizer = accelerator.prepare(model, tokenizer)
    trainer.train()

    if accelerator.is_local_main_process:
        model_last_id = os.path.join(args.runs_dir, 'last')
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(
            model_last_id,
            is_main_process=accelerator.is_main_process,
            save_function=accelerator.save,
        )
        score_state_dict = unwrapped_model.score.state_dict()
        print(score_state_dict)
        torch.save(score_state_dict, os.path.join(model_last_id, 'score.pt'))


def get_args():
    parser = argparse.ArgumentParser(
        description="Train a transformer model with LoRA adaptation on text classification tasks.")

    # Most likely to be adjusted parameters
    parser.add_argument('--checkpoint', type=str, default='llama3_weight', help='Model checkpoint path')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training and validation')
    parser.add_argument('--data_path', type=str, default='NAID/NAID_test_extrainfo_arxiv_id.csv',
                        help='Path to the training dataset CSV file')
    parser.add_argument('--test_data_path', type=str, default='NAID/NAID_train_extrainfo_arxiv_id.csv',
                        help='Path to the testing dataset CSV file')
    parser.add_argument('--runs_dir', type=str, default=None,
                        help='Directory for storing TensorBoard logs and model checkpoints')

    # Dataset and training configuration
    parser.add_argument('--total_epochs', type=int, default=5, help='Total number of epochs to train')
    parser.add_argument('--base_lr', type=float, default=5e-5, help='Base learning rate for the optimizer')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate for the optimizer')
    parser.add_argument('--weight_decay', type=float, default=1e-2, help='Weight decay for the optimizer')

    parser.add_argument('--max_length', type=int, default=1024, help='Maximum length of the tokenized input sequences')
    parser.add_argument('--loss_func', type=str, default='mse', choices=['bce', 'mse', 'l1', 'smoothl1', 'focalmse'],
                        help='Loss function to use')
    parser.add_argument('--num_labels', type=int, default=1, help='Number of labels for sequence classification')
    parser.add_argument('--load_in_8bit', type=bool, default=True,
                        help='Whether to load the model in 8-bit for efficiency')
    parser.add_argument('--device', type=str, default='cuda', help='Device to train the model on (cuda or cpu)')
    parser.add_argument('--lora_r', type=int, default=16, help='Rank of LoRA layers')
    parser.add_argument('--lora_alpha', type=int, default=32, help='Expansion factor for LoRA layers')
    parser.add_argument('--lora_dropout', type=float, default=0.05, help='Dropout rate for LoRA layers')
    parser.add_argument('--target_modules', type=str, default='q_proj,v_proj',
                        help='Comma-separated list of transformer modules to apply LoRA')
    parser.add_argument('--warmup_ratio', type=float, default=0.1, help='Warmup ratio for learning rate scheduler')
    parser.add_argument('--prompt_style', type=int, default=0)  # Modified in NAID/dataset.py as needed

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    main(args)
