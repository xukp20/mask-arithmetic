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
    parser.add_argument("--input_dir", type=str, help='The path to the input directory.', default=None)
    parser.add_argument("--values", type=str, help='The values to generate the exprs for.', default=None)
    parser.add_argument("--num_operands", type=str, help='The range of operand numbers.', default='2-5')
    parser.add_argument("--length_range", type=str, help='The range of the proof lengths.', default=None)
    
    parser.add_argument("--train_count", type=int, help='The count of training samples.', default=100000)
    parser.add_argument("--valid_count", type=int, help='The count of validation samples.', default=500)
    
    return parser.parse_args()

from utils import parse_range
import os, json

def format_input_dir(args):
    # output_base / values / proof_num_op{num_operands}
    base_dir = os.path.join(args.output_base, args.values)
    if not args.input_dir:
        args.input_dir = f"proof_num_op{args.num_operands}"
    return os.path.join(base_dir, args.input_dir)


def load_data(args):
    input_dir = format_input_dir(args)
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

TOKEN_TO_TACTIC={
    token: name for name, token in TACTIC_TO_TOKEN.items()
}


import random
def parse_proof(proof):
    source = Expr.from_dict(proof['source'])
    target = Expr.from_dict(proof['target'])
    steps = [ProofStep.from_dict(step) for step in proof['steps']]

    data = []
    # get all one step tactics
    all_targets = [step.expr for step in steps]
    sources = [source] + all_targets[:-1]
    for i in range(len(all_targets)):
        source = sources[i]
        target = all_targets[i]
        step = steps[i]
        data.append({
            'source': str(source),
            'target': str(target),
            'tactic': TACTIC_TO_TOKEN[class2name(step.tactic)],
            'position': str(step.position)
        })

    return data


INPUT_MAX_LENGTH=15
OUTPUT_MAX_LENGTH=10

MASK_TOKEN="<mask>"

from transformers import PreTrainedTokenizerFast
def get_sample_ids(sample, tokenizer):
    source = sample['source']
    target = sample['target']
    tactic = sample['tactic']
    position = sample['position']

    # format the input as source <sep> tactic position </sep>
    SEP_TOKEN = tokenizer.bos_token
    SEP_END_TOKEN = tokenizer.eos_token

    input_text = f"{source} {SEP_TOKEN} {tactic} {position} {SEP_END_TOKEN}"
    # get input_ids and attention_mask
    input_ids = tokenizer(input_text, return_tensors="pt", padding="max_length", max_length=INPUT_MAX_LENGTH, truncation=True)
    attention_mask = input_ids['attention_mask'].squeeze()
    input_ids = input_ids['input_ids'].squeeze()

    # format the output as target
    output_ids = tokenizer(target, return_tensors="pt", padding="max_length", max_length=OUTPUT_MAX_LENGTH, truncation=True)
    output_attention_mask = output_ids['attention_mask'].squeeze()
    output_ids = output_ids['input_ids'].squeeze()

    return {
        "proof_state_before_with_tactic_ids": input_ids.tolist(),
        "proof_state_before_with_tactic_attention_mask": attention_mask.tolist(),
        "proof_state_after_ids": output_ids.tolist(),
        "proof_state_after_attention_mask": output_attention_mask.tolist()
    }


from tqdm import tqdm
def generate_data(args):
    # laod the data
    proofs = load_data(args)

    # get the tokenizer
    vocab_path = os.path.join(args.output_base, DEFAULT_VOCAB_PATH)
    tokenizer = get_tokenizer(vocab_path, INPUT_MAX_LENGTH, padding_side="right")
    # look for sep and /sep
    sep_id = tokenizer.eos_token_id
    sep_end_id = tokenizer.bos_token_id
    mask_id = tokenizer.convert_tokens_to_ids(MASK_TOKEN)
    print(f"SEP ID: {sep_id}, SEP END ID: {sep_end_id}, MASK ID: {mask_id}")

    # generate the data
    train_data = []
    valid_data = []
    # flatten 
    proofs = [proof for length in proofs for proof in proofs[length]]
    data = []

    for proof in tqdm(proofs, desc="Parsing proofs"):
        data.extend(parse_proof(proof))

    import random
    random.shuffle(data)
    print(f"Generated {len(data)} samples.")

    for sample in data:
        if len(valid_data) < args.valid_count:
            valid_data.append(get_sample_ids(sample, tokenizer))
        elif len(train_data) < args.train_count:
            train_data.append(get_sample_ids(sample, tokenizer))
        else:
            break

    if len(train_data) < args.train_count:
        print(f"Error: only generated {len(train_data)} training samples.")
        raise ValueError("Not enough training samples.")

    # print top 3 samples
    print("Top 3 samples:")
    for i in range(min(3, len(train_data))):
        print()
        # print the fiorls
        print(f"Sample {i+1}:")
        print(f"Source: {tokenizer.decode(train_data[i]['proof_state_before_with_tactic_ids'])}")
        print(f"Source IDs: {train_data[i]['proof_state_before_with_tactic_ids']}")
        print(f"Source Mask: {train_data[i]['proof_state_before_with_tactic_attention_mask']}")
        print(f"Target: {tokenizer.decode(train_data[i]['proof_state_after_ids'])}")
        print(f"Target IDs: {train_data[i]['proof_state_after_ids']}")
        print(f"Target Mask: {train_data[i]['proof_state_after_attention_mask']}")

    return train_data, valid_data


def format_save_paths(args, train_count, valid_count):
    base_dir = os.path.join(args.output_base, args.values)
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    # train or eval _ count _ length_range _ contain num if contain number is not None else "all" _ task
    train_path = os.path.join(base_dir, f"train_{train_count}_{args.length_range}_state.jsonl")
    valid_path = os.path.join(base_dir, f"eval_{valid_count}_{args.length_range}_state.jsonl")
    return train_path, valid_path


def save_data(train_data, valid_data, args):
    train_path, valid_path = format_save_paths(args, len(train_data), len(valid_data))

    with open(train_path, 'w') as f:
        for sample in train_data:
            f.write(json.dumps(sample) + '\n')
    
    with open(valid_path, 'w') as f:
        for sample in valid_data:
            f.write(json.dumps(sample) + '\n')

    print(f"Saved {len(train_data)} training samples to {train_path}")
    print(f"Saved {len(valid_data)} validation samples to {valid_path}")


def main():
    args = parse_args()
    train_data, valid_data = generate_data(args)
    save_data(train_data, valid_data, args)


if __name__ == "__main__":
    main()