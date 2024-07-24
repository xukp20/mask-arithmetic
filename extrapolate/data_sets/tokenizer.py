# get a word level tokenizer

import os
import json

SPECIAL_TOKENS = {
    "pad_token": "<pad>",
    "bos_token": "<s>",
    "eos_token": "</s>",
    "unk_token": "<unk>",
    "mask_token": "<mask>",
}

def get_tokenizer(vocab_path, max_length, padding_side="left"):
    # read vocab
    with open(vocab_path, 'r') as f:
        vocab = json.load(f)

    vocab = {word.strip(): i for i, word in enumerate(vocab)}
    for token in SPECIAL_TOKENS.values():
        vocab[token] = len(vocab)

    from tokenizers.models import WordLevel
    word_level = WordLevel(vocab=vocab, unk_token="<unk>")

    from tokenizers import pre_tokenizers
    import tokenizers
    pre_tokenizer = pre_tokenizers.Sequence(
        [
            pre_tokenizers.Whitespace(),
        ]
    )

    tokenizer = tokenizers.Tokenizer(word_level)
    tokenizer.pre_tokenizer = pre_tokenizer
    tokenizer.decoder = tokenizers.decoders.BPEDecoder()

    from transformers import PreTrainedTokenizerFast
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer)
    tokenizer.add_special_tokens(SPECIAL_TOKENS)
    tokenizer.model_max_length = max_length
    tokenizer.padding_side = padding_side

    return tokenizer


