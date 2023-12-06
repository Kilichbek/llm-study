import torch
import torch.nn.functional as F

# ------------------------------------------------------------------------------
class BaseLanguageModel(torch.nn.Module):
    """
    Base class for language models.
    """
    def __init__(self, vocab_size: int):
        super(BaseLanguageModel, self).__init__()
        self.vocab_size = vocab_size

    def forward(self, x):
        raise NotImplementedError(
            "Subclasses must implement forward method."
        )

    def loss(self, x, y):
        raise NotImplementedError(
            "Subclasses must implement loss method."
        )

    @torch.inference_mode()
    def generate(self, x):
        raise NotImplementedError(
            "Subclasses must implement generate method."
        )

# ------------------------------------------------------------------------------
class BigramLanguageModel(BaseLanguageModel):
    """
    super simple bi-gram language model
    adopted from https://github.com/karpathy/ng-video-lecture/
    """
    def __init__(self, vocab_size: int):
        super(BigramLanguageModel, self).__init__(vocab_size)
        self.bigram = torch.nn.Embedding(vocab_size, vocab_size)

    def forward(self, tokens):
        return self.bigram(tokens)

    @torch.inference_mode()
    def generate(self, x):
        return torch.argmax(self.forward(x), dim=1)

    def loss(self, pred_tokens, target_tokens):
        return F.cross_entropy(pred_tokens, target_tokens)