from collections import Counter


class BPETokenizer:
    """A smarter tokenizer that learns common chunks, like "th" or "ing".

    Reading letter-by-letter is slow. People read in chunks. BPE (Byte Pair
    Encoding) learns chunks automatically: look at the text, find the two
    neighbours that appear together most often, glue them into one new token,
    and repeat. After a while common pieces like "ab" or "the" become single
    tokens, so the model reads in bigger, smarter gulps.
    """

    def __init__(self):
        # merges[i] = the (left, right) pair learned at step i, in order.
        self.merges = []
        self.stoi = {}
        self.itos = {}

    def _rebuild_vocab(self, base_chars):
        tokens = list(base_chars)
        for left, right in self.merges:
            tokens.append(left + right)
        self.stoi = {token: index for index, token in enumerate(tokens)}
        self.itos = {index: token for token, index in self.stoi.items()}

    def train(self, text, num_merges):
        """Learn the chunks: repeatedly glue the most common neighbour pair."""
        base_chars = sorted(set(text))
        symbols = list(text)  # each element is a token string, starting as chars
        self.merges = []
        for _ in range(num_merges):
            pairs = Counter(zip(symbols, symbols[1:], strict=False))
            if not pairs:
                break
            (left, right), count = pairs.most_common(1)[0]
            if count < 2:
                break  # nothing worth merging
            self.merges.append((left, right))
            symbols = _merge_pair(symbols, left, right)
        self._rebuild_vocab(base_chars)

    @property
    def vocab_size(self):
        """How many tokens we know: the single characters plus every learned chunk."""
        return len(self.stoi)

    def encode(self, text):
        """Turn text into token numbers, gluing in the learned chunks as we go."""
        symbols = list(text)
        for left, right in self.merges:
            symbols = _merge_pair(symbols, left, right)
        return [self.stoi[symbol] for symbol in symbols]

    def decode(self, ids):
        """Read the text back out by stitching each token's characters together."""
        return "".join(self.itos[index] for index in ids)


def _merge_pair(symbols, left, right):
    """Walk the list and glue every adjacent (left, right) into one token."""
    merged = []
    i = 0
    while i < len(symbols):
        if i < len(symbols) - 1 and symbols[i] == left and symbols[i + 1] == right:
            merged.append(left + right)
            i += 2
        else:
            merged.append(symbols[i])
            i += 1
    return merged
