import os
import wandb
import torch
from tqdm import tqdm
from src.data_processing.tokenizers import SimpleCharTokenizer
from src.data_processing.datasets import TinyShakespeareDataset
from src.train.trainer import Trainer
from src.model.networks import BigramLanguageModel


torch.manual_seed(42)

# hyperparameters
model_name = "bigram"
train_frac = 0.9 # what fraction of the data will be used for training?
batch_size = 32 # how many independent sequences will we process in parallel?
block_size = 8 # what is the maximum context length for predictions?
max_iters = 3000
eval_interval = 300
learning_rate = 1e-2
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
checkpoint_path = f"/home/haydark/dev/uzbekgpt/results/checkpoints/{model_name}"

if not os.path.exists(checkpoint_path):
    os.makedirs(checkpoint_path)
# -------------------------------------------------------------
wandb.init(
    project="llm-exploration",
    # track hyperparameters and run metadata
    config={
        "architecture": model_name,
        "dataset": "tiny-shakespeare",
        "learning_rate": learning_rate,
        "train_frac": train_frac,
        "batch_size": batch_size,
        "block_size": block_size,
        "max_iters": max_iters,
        "eval_interval": eval_interval,
        "eval_iters": eval_iters,
    }

)
# -------------------------------------------------------------

def main():
    # load data
    fname = "/home/haydark/dev/uzbekgpt/data/tinyshakespeare/input.txt"
    with open(fname, 'r', encoding='utf-8') as f:
        text = f.read()

    # split into training and validation sets
    split_idx = int(train_frac * len(text))
    train_text, val_text = text[:split_idx], text[split_idx:]

    # initialize tokenizer
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    stoi = { ch:i for i,ch in enumerate(chars) }
    itos = { i:ch for i,ch in enumerate(chars) }
    tokenizer = SimpleCharTokenizer(vocab_size, stoi, itos)

    # initialize datasets
    train_dataset = TinyShakespeareDataset(train_text, block_size, tokenizer)
    val_dataset = TinyShakespeareDataset(val_text, block_size, tokenizer)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # initialize model
    model = BigramLanguageModel(vocab_size, block_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # train model
    trainer = Trainer(model, optimizer, train_loader, val_loader, max_iters, eval_iters, eval_interval, checkpoint_path)
    train_losses, val_losses = trainer.train()




if __name__ == '__main__':
    main()
    