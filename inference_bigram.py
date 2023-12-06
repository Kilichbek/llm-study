import os
import torch
from tqdm import tqdm
from src.data_processing.tokenizers import SimpleCharTokenizer
from src.model.networks import BigramLanguageModel


torch.manual_seed(42)

# hyperparameters
model_name = "bigram"
train_frac = 0.9 # what fraction of the data will be used for training?
batch_size = 32 # how many independent sequences will we process in parallel?
block_size = 8 # what is the maximum context length for predictions?
max_iters = 10
eval_interval = 1
learning_rate = 1e-2
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_workers = 0
checkpoint_path = "/home/haydark/dev/uzbekgpt/results/checkpoints/bigram/model-10000.pt"

def main():
    # load data
    fname = "/home/haydark/dev/uzbekgpt/data/raw/input.txt"
    with open(fname, 'r', encoding='utf-8') as f:
        text = f.read()

    # initialize tokenizer
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    stoi = { ch:i for i,ch in enumerate(chars) }
    itos = { i:ch for i,ch in enumerate(chars) }
    tokenizer = SimpleCharTokenizer(vocab_size, stoi, itos)

    # initialize model
    model = BigramLanguageModel(vocab_size)
    # load checkpoint
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()

    # generate text
    seed = "I have a dream "
    tokens = tokenizer.encode(seed)
    tokens = tokens.to(device)
    tokens = tokens.unsqueeze(0)
    generated_tokens = model.generate(tokens, max_new_tokens=500)
    generated_tokens = generated_tokens.cpu().numpy()
    
    # decode tokens
    text = tokenizer.decode(generated_tokens[0])
    print(text)

if __name__ == '__main__':
    main()
    