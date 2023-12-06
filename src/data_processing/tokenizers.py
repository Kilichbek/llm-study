import torch


class BaseTokenizer:
    """
    Base class for tokenizers.
    """
    def __init__(self, vocab_size: int = 0):
        self.vocab_size = vocab_size

    def tokenize(self, text: str) -> torch.Tensor:
        raise NotImplementedError(
            "Subclasses must implement tokenize method."
        )

    def decode(self, tokens: torch.Tensor) -> str:
        raise NotImplementedError(
            "Subclasses must implement detokenize method."
        )

#-------------------------------------------------------------

class SimpleCharTokenizer(BaseTokenizer):
    """
    Simple character tokenizer.
    """
    def __init__(self, vocab_size: int, encoder, decoder):
        super().__init__(vocab_size)
        self.encoder = encoder
        self.decoder = decoder

    def encode(self, text: str) -> torch.Tensor:
        """
        Encode a string into a tensor of tokens.
        """
        return torch.tensor([self.encoder[c] for c in text])

    def decode(self, tokens: torch.Tensor) -> str:
        """
        Decode a tensor of tokens into a string.
        """
        return "".join([self.decoder[t] for t in tokens])