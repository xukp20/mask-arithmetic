"""
    Define the toy tokenizer with only vocabulary related to numbers and operations.
"""

MAX_NUMBER=100
DEFAULT_VOCAB = [
    # separated tokens for each number
    str(i) for i in range(MAX_NUMBER + 1)
] + [
    "+", "-", "*", "/", "(", ")", "=", ":", "[MASK]", "[PAD]", "[UNK]"
]

IGNORE_INDEX=-100

class ArithmeticTokenizer:
    def __init__(self, vocab):
        self.vocab = vocab
        self.number_vocab = [token for token in self.vocab if token in [str(i) for i in range(MAX_NUMBER + 1)]]
        self.operation_vocab = [token for token in self.vocab if token in ['+', '-', '*', '/']]

        if '[MASK]' not in self.vocab:
            self.vocab.append('[MASK]')
        if '[PAD]' not in self.vocab:
            self.vocab.append('[PAD]')
        if '[UNK]' not in self.vocab:
            self.vocab.append('[UNK]')
        
        self.token2id = {token: idx for idx, token in enumerate(self.vocab)}
        self.id2token = {idx: token for token, idx in self.token2id.items()}
        
        self.mask_token = '[MASK]'
        self.mask_token_id = self.token2id['[MASK]']
        self.pad_token = '[PAD]'
        self.pad_token_id = self.token2id['[PAD]']
        self.unk_token = '[UNK]'
        self.unk_token_id = self.token2id['[UNK]']

    def encode(self, text, max_length=None, padding=False):
        tokens = text.split()
        ids = [self.token2id.get(token, self.unk_token_id) for token in tokens]
        attention_mask = [1] * len(ids)
        
        if max_length is not None:
            ids = ids[:max_length]
            attention_mask = attention_mask[:max_length]
            if padding:
                ids = ids + [self.pad_token_id] * (max_length - len(ids))
                attention_mask = attention_mask + [0] * (max_length - len(attention_mask))
        
        return {
            'input_ids': ids,
            'attention_mask': attention_mask
        }

    def decode(self, ids):
        ids = [int(id) for id in ids]
        return ' '.join([self.id2token.get(id, self.unk_token if id != IGNORE_INDEX else "[-100]") for id in ids])

    def __len__(self):
        return len(self.vocab)

    def get_vocab_size(self):
        return len(self.vocab)

    def get_number_mask(self, inputs):
        if isinstance(inputs, str):
            inputs = self.encode(inputs)['input_ids']
        
        return [1 if self.id2token[id] in self.number_vocab else 0 for id in inputs]
        
    def get_mask_token_id(self):
        return self.mask_token_id

    def get_mask_token(self):
        return self.mask_token

DEFAULT_TOKENIZER = ArithmeticTokenizer(DEFAULT_VOCAB)