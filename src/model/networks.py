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
        self.lookup_table = torch.nn.Embedding(vocab_size, vocab_size)

    def forward(self, tokens):

        logits = self.lookup_table(tokens) # [batch_size, context_length, vocab_size]
        return logits
        

    @torch.inference_mode()
    def generate(self, tokens, max_new_tokens=100):
        """
        Generate new tokens from a seed string.
        """
        self.eval()
        for _ in range(max_new_tokens):
            logits = self.forward(tokens)
            logits = logits[:, -1, :] # focus on the last token in the context
            probs = F.softmax(logits, dim=-1) # [batch_size, vocab_size]
            new_token = torch.multinomial(probs, num_samples=1)
            tokens = torch.cat([tokens, new_token], dim=-1) # [batch_size, context_length + 1]

        return tokens

    def loss(self, logits, target_tokens):
        """
        Computes the cross-entropy loss between the predicted tokens and the
        target tokens.
        """
        batch_size, context_length, vocab_size = logits.shape
        logits = logits.view(batch_size * context_length, vocab_size) # [batch_size * context_length]
        target_tokens = target_tokens.view(batch_size * context_length) # [batch_size * context_length]
        loss = F.cross_entropy(logits, target_tokens)
        return loss

