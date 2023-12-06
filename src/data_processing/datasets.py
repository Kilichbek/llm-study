import torch

class TinyShakespeareDataset(torch.utils.data.Dataset):
    """
    Tiny Shakespeare dataset.
    src: https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
    """
    def __init__(self, text: str, context_length: int, tokenizer):
        self.text = text
        self.tokenizer = tokenizer
        self.context_length = context_length

    def __len__(self):
        return len(self.text) - self.context_length

    def __getitem__(self, idx):
        """
        Returns a single training example.
        """
        x = self.text[idx:idx + self.context_length]
        y = self.text[idx + 1:idx + self.context_length + 1]
        return self.tokenizer.encode(x), self.tokenizer.encode(y)