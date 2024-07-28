"""
    Generate the baseline data:
    - For each proof, take out all the exprs from source to the one before target.
    - The target are all the target of the proof
    - The tactic is the one to be used
"""

from expr import Expr
from tactic import ALL_TACTICS, class2name
from proof import Proof, ProofStep

import argparse

PROJECT_BASE="../../"
import sys
sys.path.append(PROJECT_BASE)
from extrapolate.data_sets.tokenizer import get_tokenizer

DEFAULT_VOCAB_PATH = f"vocab_with_6_tactics.json"

def parse_args():
    parser = argparse.ArgumentParser(description='Generate the baseline data for mask-arithmetic.')
    parser.add_argument("--output_base", type=str, help="The base path to save the output to.", default="data")
    parser.add_argument("--input_dir", type=str, help='The path to the input directory.', default='proof_num_op2-3-4-5')
    parser.add_argument("--length_range", type=str, help='The range of the proof lengths.', default='10-15')
    
    parser.add_argument("--train_count", type=int, help='The count of training samples.', default=100000)
    parser.add_argument("--valid_count", type=int, help='The count of validation samples.', default=500)
    
    parser.add_argument("--task", type=str, help="The task to generate the data for.", default="tactic", choices=["tactic", "position", "both"])
    return parser.parse_args()


from utils import parse_range
import os, json

def load_data(args):
    input_dir = os.path.join(args.output_base, args.input_dir)
    proofs = {}
    lengths = parse_range(args.length_range)
    for length in lengths:
        with open(os.path.join(input_dir, f"{length}.jsonl"), 'r') as f:
            proofs[length] = []
            for line in f:
                proof = json.loads(line)
                proofs[length].append(proof)
            
        print(f"Loaded {len(proofs[length])} proofs of length {length}")
    
    print("Loaded {} proofs in total.".format(sum([len(proofs[length]) for length in lengths])))

    return proofs


TACTIC_TO_TOKEN={
    name: f"T{i+1}" for i, name in enumerate(ALL_TACTICS)
}

def parse_proof(proof):
    source = Expr.from_dict(proof['source'])
    target = Expr.from_dict(proof['target'])
    steps = [ProofStep.from_dict(step) for step in proof['steps']]

    # generate all the possible tactic data
    data = []
    for i, step in enumerate(steps):
        data.append({
            'source': str(source),
            'target': str(target),
            'tactic': TACTIC_TO_TOKEN[class2name(step.tactic)],
            'position': str(step.position),
        })
        source = step.expr
    
    return data

# tokenizer the data
# MAX LENGTH is 1 + 1 + (5 + 4) + (5 + 4) = 20
TACTIC_LENGTH=1
POSITION_LENGTH=1
EXPR_LENGTH=10
MAX_LENGTH=TACTIC_LENGTH + POSITION_LENGTH + EXPR_LENGTH + EXPR_LENGTH
MASK_TOKEN="<mask>"

from transformers import PreTrainedTokenizerFast
def get_sample_ids(sample, tokenizer: PreTrainedTokenizerFast, mask_id=None, task="tactic"):
    source = sample['source']
    target = sample['target']
    tactic = sample['tactic']
    position = sample['position']

    source = tokenizer.encode(str(source), add_special_tokens=False, truncation=True, padding='max_length', max_length=EXPR_LENGTH)
    target = tokenizer.encode(str(target), add_special_tokens=False, truncation=True, padding='max_length', max_length=EXPR_LENGTH)
    tactic = tokenizer.encode(tactic, add_special_tokens=False, truncation=True, padding='max_length', max_length=TACTIC_LENGTH)
    position = tokenizer.encode(str(position), add_special_tokens=False, truncation=True, padding='max_length', max_length=POSITION_LENGTH)
    # tactic and position are masks
    
    # concat the ids
    if task == "tactic":
        return [mask_id] * TACTIC_LENGTH + [tokenizer.pad_token_id] * POSITION_LENGTH + source + target
    elif task == "position":
        return tactic + [mask_id] * POSITION_LENGTH + source + target
    else:
        return [mask_id] * TACTIC_LENGTH + [mask_id] * POSITION_LENGTH + source + target

IGNORE_INDEX=-100
def get_labels(sample, tokenizer: PreTrainedTokenizerFast, task="tactic"):
    # predict tactic and position at the same time
    tactic_id = tokenizer.encode(sample['tactic'], add_special_tokens=False, truncation=True, padding='max_length', max_length=TACTIC_LENGTH)[0]
    position_id = tokenizer.encode(str(sample['position']), add_special_tokens=False, truncation=True, padding='max_length', max_length=POSITION_LENGTH)[0]

    if task == "tactic":
        return [tactic_id] * TACTIC_LENGTH + [IGNORE_INDEX] * POSITION_LENGTH + [IGNORE_INDEX] * (EXPR_LENGTH + EXPR_LENGTH)
    elif task == "position":
        return [IGNORE_INDEX] * TACTIC_LENGTH + [position_id] * POSITION_LENGTH + [IGNORE_INDEX] * (EXPR_LENGTH + EXPR_LENGTH)
    else:
        return [tactic_id] * TACTIC_LENGTH + [position_id] * POSITION_LENGTH + [IGNORE_INDEX] * (EXPR_LENGTH + EXPR_LENGTH)

def get_attn_mask(sample, tokenizer: PreTrainedTokenizerFast, task="tactic"):
    # get actual length of the two exprs
    source = tokenizer(sample['source'], add_special_tokens=False, truncation=True, padding='max_length', max_length=EXPR_LENGTH)
    target = tokenizer(sample['target'], add_special_tokens=False, truncation=True, padding='max_length', max_length=EXPR_LENGTH)
    source_mask = source['attention_mask']
    target_mask = target['attention_mask']

    if task == "tactic":
        return [1] * TACTIC_LENGTH + [0] * POSITION_LENGTH + source_mask + target_mask
    else:
        return [1] * (TACTIC_LENGTH + POSITION_LENGTH) + source_mask + target_mask


def generate_data(args):
    # load the data
    proofs = load_data(args)

    # get the tokenizer
    vocab_path = os.path.join(args.output_base, DEFAULT_VOCAB_PATH)
    tokenizer = get_tokenizer(vocab_path, MAX_LENGTH, padding_side="right")
    mask_id = tokenizer.convert_tokens_to_ids(MASK_TOKEN)

    # generate the data
    train_data = []
    valid_data = []
    # flatten 
    proofs = [proof for length in proofs for proof in proofs[length]]
    data = []

    for proof in proofs:
        data.extend(parse_proof(proof))

    import random
    random.shuffle(data)

    for sample in data:
        ids = get_sample_ids(sample, tokenizer, mask_id, args.task)
        labels = get_labels(sample, tokenizer, args.task)
        attn_mask = get_attn_mask(sample, tokenizer, args.task)

        if len(train_data) < args.train_count:
            train_data.append({
                'input_ids': ids,
                'labels': labels,
                'attention_mask': attn_mask
            })
        elif len(valid_data) < args.valid_count:
            valid_data.append({
                'input_ids': ids,
                'labels': labels,
                'attention_mask': attn_mask
            })
        else:
            break

    # print top 3 samples
    print("Top 3 samples:")
    for i in range(3):
        # decode the input_ids
        sample = train_data[i]
        input_ids = sample['input_ids']
        print(input_ids)
        print(tokenizer.decode(input_ids, skip_special_tokens=False))
        print(sample['labels'])
        print(sample['attention_mask'])
        print()

    return train_data, valid_data, tokenizer


def save_data(train_data, valid_data, args):
    train_path = os.path.join(args.output_base, "train_baseline_{0}.jsonl".format(args.task))
    valid_path = os.path.join(args.output_base, "eval_baseline_{0}.jsonl".format(args.task))

    with open(train_path, 'w') as f:
        for sample in train_data:
            f.write(json.dumps(sample) + '\n')
    
    with open(valid_path, 'w') as f:
        for sample in valid_data:
            f.write(json.dumps(sample) + '\n')
    

def main():
    args = parse_args()
    train_data, valid_data, tokenizer = generate_data(args)
    save_data(train_data, valid_data, args)
    # save the tokenizer
    tokenizer_path = os.path.join(args.output_base, "6_tactics_tokenizer")
    tokenizer.save_pretrained(tokenizer_path)

if __name__ == "__main__":
    main()