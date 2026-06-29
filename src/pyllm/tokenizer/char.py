class CharTokenizer:
    """The simplest possible tokenizer: one number per character.

    Imagine giving every letter a locker number: a=0, b=1, c=2... To "encode" a
    word you write down its locker numbers; to "decode" you read the letters back
    out. Simple and lossless, but the model has to spell everything out one
    letter at a time -- which is why we also build a smarter BPE tokenizer later.
    """

    def __init__(self, text):
        chars = sorted(set(text))
        self.stoi = {char: index for index, char in enumerate(chars)}
        self.itos = {index: char for char, index in self.stoi.items()}

    @property
    def vocab_size(self):
        return len(self.stoi)

    def encode(self, text):
        return [self.stoi[char] for char in text]

    def decode(self, ids):
        return "".join(self.itos[index] for index in ids)
