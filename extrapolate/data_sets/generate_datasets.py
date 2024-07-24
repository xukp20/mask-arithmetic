"""
    Generate training datasets from the raw data of def or computation.
"""

import argparse, os, sys, json, random

def parse_args():
    parser = argparse.ArgumentParser(description='Generate training datasets from the raw data of def or computation.')
    parser.add_argument('--data_base', type=str, help='The base path to load the data from.', default='data')
    parser.add_argument('--task', type=str, help='The task to generate the dataset for.', default='def', choices=['def', 'compute'])

    # tokenizer settings
    parser.add_argument('--vocab_path', type=str, help='The path to the vocab file.', default='vocab.json')
    parser.add_argument('--max_length', type=int, help='The maximum length of the input sequence.', default=8)
    parser.add_argument('--padding_side', type=str, help='The side to pad the sequence.', default='right')
    parser.add_argument('--mask_token', type=str, help='The mask token to use.', default='<mask>')

    # data settings
    parser.add_argument('--input_path', type=str, help='The path to the input data.', default='def_0-12_0-11_add&sub_2.json')
    parser.add_argument('--eval_ratio', type=float, help='The ratio of the data to use for evaluation.', default=0.05)
    parser.add_argument('--output_path', type=str, help='The path to save the output data.', default=None)
    parser.add_argument('--num_operands', type=list, help='The number of operands to use in the definition.', default=[2])
    # parser.add_argument('--apply_reverse', help='Whether to reverse the lhs and rhs for more data', action='store_true')
    return parser.parse_args()


def load_tokenizer(args):
    from tokenizer import get_tokenizer
    vocab_path = os.path.join(args.data_base, args.vocab_path)
    return get_tokenizer(vocab_path, args.max_length, args.padding_side)


def load_data(args):
    def load_def_data(args):
        input_path = os.path.join(args.data_base, args.input_path)
        with open(input_path, 'r') as f:
            data = json.load(f)
        number_dict = {int(k): v for k, v in data.items()}
        # take out the subsets of the given number of operands
        data = []
        for num, op_num_subsets in number_dict.items():
            for op_num, exprs in op_num_subsets.items():
                if int(op_num) in args.num_operands:
                    # should add mask_num to expr
                    for expr in exprs:
                        expr['mask_num'] = num
                        data.append(expr)
        return data
    
    def load_compute_data(args):
        raise NotImplementedError

    if args.task == 'def':
        return load_def_data(args)
    elif args.task == 'compute':
        return load_compute_data(args)


# parse a expr sample to a mlm train sample
IGNORE_INDEX=-100
def parse_single_def_sample(expr, tokenizer, mask_token):
    mask_num = expr['mask_num']
    # choose one of the mask_num in lhs or operands to be the mask
    all_numbers = [expr['lhs']] + expr['rhs_operands']
    same_numbers_ids = [i for i, num in enumerate(all_numbers) if num == mask_num]
    if len(same_numbers_ids) == 0:
        print(f"Warning: mask_num {mask_num} not found in the expr {expr}")
        return None
    elif len(same_numbers_ids) == 1:
        mask_num_id = same_numbers_ids[0]
    else:
        mask_num_id = random.choice(same_numbers_ids)

    def format_input_sequence(expr, mask_num_id, mask_token):
        pattern = "{lhs} = {rhs}"
        ops = expr['rhs_ops']
        operands = expr['rhs_operands']
        if mask_num_id == 0:
            lhs = mask_token
        else:
            lhs = str(expr['lhs'])
        
        rhs = []
        for i, op in enumerate(ops):
            if mask_num_id == i + 1:
                rhs.append(mask_token)
            else:
                rhs.append(str(operands[i]))
            rhs.append(op)
        if mask_num_id == len(ops) + 1:
            rhs.append(mask_token)
        else:
            rhs.append(str(operands[-1]))
        return pattern.format(lhs=lhs, rhs=" ".join(rhs))
    
    input_sequence = format_input_sequence(expr, mask_num_id, mask_token)

    # tokenize the input sequence
    tokenized = tokenizer(input_sequence, return_tensors='pt', padding='max_length', truncation=True, max_length=tokenizer.model_max_length)
    input_ids = tokenized['input_ids'].squeeze().tolist()
    attention_mask = tokenized['attention_mask'].squeeze().tolist()

    # find the mask token id and build the labels
    labels = [IGNORE_INDEX] * len(input_ids)
    mask_token_id = input_ids.index(tokenizer.convert_tokens_to_ids(mask_token))
    masked_token = str(mask_num)
    labels[mask_token_id] = tokenizer.convert_tokens_to_ids(masked_token)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }


from tqdm import tqdm
def parse_def_data(data, tokenizer, mask_token):
    parsed_data = []
    for expr in tqdm(data):
        parsed_sample = parse_single_def_sample(expr, tokenizer, mask_token)
        if parsed_sample is not None:
            parsed_data.append(parsed_sample)
    return parsed_data

def save_data(parsed_data, args):
    if args.output_path is None:
        output_name = args.input_path + "l"
    else:
        output_name = args.output_path + "l"

    train_output_path = os.path.join(args.data_base, f"train_{output_name}")
    eval_output_path = os.path.join(args.data_base, f"eval_{output_name}")

    import random
    random.shuffle(parsed_data)

    eval_size = int(len(parsed_data) * args.eval_ratio)
    train_data = parsed_data[eval_size:]
    eval_data = parsed_data[:eval_size]

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
    if args.task == 'def':
        parsed_data = parse_def_data(data, tokenizer, args.mask_token)
    save_data(parsed_data, args)
    # save the tokenizer
    tokenizer.save_pretrained(os.path.join(args.data_base, 'tokenizer'))


if __name__ == '__main__':
    main()


