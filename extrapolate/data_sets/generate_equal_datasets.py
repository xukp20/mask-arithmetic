"""
    Generate training and evaluation datasets from the raw data of equaltion.
"""

import argparse, os, sys, json, random

def parse_args():
    parser = argparse.ArgumentParser(description='Generate training datasets from the raw data of def or computation.')
    parser.add_argument('--data_base', type=str, help='The base path to load the data from.', default='data')

    # tokenizer settings
    parser.add_argument('--vocab_path', type=str, help='The path to the vocab file.', default='vocab.json')
    parser.add_argument('--max_length', type=int, help='The maximum length of the input sequence.', default=8)
    parser.add_argument('--padding_side', type=str, help='The side to pad the sequence.', default='right')
    parser.add_argument('--mask_token', type=str, help='The mask token to use.', default='<mask>')

    # data settings
    parser.add_argument('--input_path', type=str, help='The path to the input data.', default='equal_0-11_add&sub_2.json')
    parser.add_argument('--eval_ratio', type=float, help='The ratio of the data to use for evaluation.', default=0.05)
    parser.add_argument('--output_path', type=str, help='The path to save the output data.', default=None)
    parser.add_argument('--num_lhs_operands', type=list, help='The number of operands to use on each side of the equation.', default=[2])
    parser.add_argument('--num_rhs_operands', type=list, help='The number of operands to use on each side of the equation.', default=[2])
    parser.add_argument('--sample_num', type=int, help='The number of samples to generate for each expr, -1 for all.', default=-1)
    parser.add_argument('--target_num', type=int, help='The target number to mask.', default=None)
    parser.add_argument('--has_num', type=int, help='The number that must be in the equation.', default=None)
    return parser.parse_args()


def load_tokenizer(args):
    from tokenizer import get_tokenizer
    vocab_path = os.path.join(args.data_base, args.vocab_path)
    return get_tokenizer(vocab_path, args.max_length, args.padding_side)


def load_data(args):
    input_path = os.path.join(args.data_base, args.input_path)
    with open(input_path, 'r') as f:
        data = json.load(f)
    
    # data is dict of dict, key are "lhs_op_num" and "rhs_op_num"
    filter_data = []
    for lhs_op_num in args.num_lhs_operands:
        for rhs_op_num in args.num_rhs_operands:
            lhs_op_num = str(lhs_op_num)
            rhs_op_num = str(rhs_op_num)
            if lhs_op_num in data:
                lhs_data = data[lhs_op_num]
                if rhs_op_num in lhs_data:
                    rhs_data = lhs_data[rhs_op_num]
                    filter_data.extend(rhs_data)

    print(f"Loaded {len(filter_data)} samples.")

    return filter_data


# parse a expr sample to a mlm train sample
from generate_equal_data import format_expr, format_equation
IGNORE_INDEX=-100
def parse_single_def_sample(expr, tokenizer, mask_token, sample_num=-1, target_num=None, has_num=None):
    lhs_operands = expr['lhs_operands']
    rhs_operands = expr['rhs_operands']
    all_operands = lhs_operands + rhs_operands

    if has_num is not None and has_num not in all_operands:
        return []

    lhs_ops = expr['lhs_ops']
    rhs_ops = expr['rhs_ops']

    # choose sample_num operands to mask
    if not target_num:
        if sample_num > 0:
            mask_indices = random.sample(range(len(all_operands)), sample_num)
        else:
            mask_indices = range(len(all_operands))
    else:
        # find all the target_num in the operands
        mask_indices = [i for i, num in enumerate(all_operands) if num == target_num]
        sample_num = min(sample_num, len(mask_indices))
        if sample_num > 0:
            mask_indices = random.sample(mask_indices, sample_num)
    
    samples = []
    # create one sample for each of the mask_indices
    for mask_index in mask_indices:
        masked_operands = all_operands.copy()
        masked_operands[mask_index] = mask_token
        masked_lhs_operands = masked_operands[:len(lhs_operands)]
        masked_rhs_operands = masked_operands[len(lhs_operands):]
        equation = format_equation(masked_lhs_operands, lhs_ops, masked_rhs_operands, rhs_ops)
        # tokenizer
        tokenized = tokenizer(equation, return_tensors='pt', padding='max_length', truncation=True, max_length=tokenizer.model_max_length)
        input_ids = tokenized['input_ids'].squeeze().tolist()
        attention_mask = tokenized['attention_mask'].squeeze().tolist()

        # find the mask token id and build the labels
        labels = [IGNORE_INDEX] * len(input_ids)
        masked_token = str(all_operands[mask_index])
        # look for the mask token id
        mask_token_id = input_ids.index(tokenizer.convert_tokens_to_ids(mask_token))
        labels[mask_token_id] = tokenizer.convert_tokens_to_ids(masked_token)

        samples.append({
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        })

    return samples


from tqdm import tqdm
def parse_equation_data(data, tokenizer, mask_token, sample_num=-1, target_num=None, has_num=None):
    samples = []
    for expr in tqdm(data):
        samples.extend(parse_single_def_sample(expr, tokenizer, mask_token, sample_num, target_num, has_num))
    return samples

def save_data(samples, args):
    if args.output_path is None:
        output_name = args.input_path.strip('.json') + "_" + "&".join([str(num) for num in args.num_lhs_operands]) + "_" + "&".join([str(num) for num in args.num_rhs_operands]) + "_num" + str(args.sample_num) + "_tar" + (str(args.target_num) if args.target_num else "any") + "_has" + (str(args.has_num) if args.has_num else "any") + ".jsonl"
    else:
        output_name = args.output_path.strip('.json') + "_" + "&".join([str(num) for num in args.num_lhs_operands]) + "_" + "&".join([str(num) for num in args.num_rhs_operands]) + "_num" + str(args.sample_num) + "_tar" + (str(args.target_num) if args.target_num else "any") + "_has" + (str(args.has_num) if args.has_num else "any") + ".jsonl"

    train_output_path = os.path.join(args.data_base, f"train_{output_name}")
    eval_output_path = os.path.join(args.data_base, f"eval_{output_name}")

    import random
    random.shuffle(samples)

    eval_num = int(len(samples) * args.eval_ratio)
    eval_data = samples[:eval_num]
    train_data = samples[eval_num:]

    with open(train_output_path, 'w') as f:
        for sample in train_data:
            f.write(json.dumps(sample) + '\n')
    print(f"{len(train_data)} samples saved to {train_output_path}")

    with open(eval_output_path, 'w') as f:
        for sample in eval_data:
            f.write(json.dumps(sample) + '\n')
        
    print(f"{len(eval_data)} samples saved to {eval_output_path}")


def main():
    args = parse_args()
    tokenizer = load_tokenizer(args)
    data = load_data(args)
    samples = parse_equation_data(data, tokenizer, args.mask_token, args.sample_num, args.target_num, args.has_num)
    save_data(samples, args)


if __name__ == "__main__":
    main()