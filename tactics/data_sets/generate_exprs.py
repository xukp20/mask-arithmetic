"""
    Generate a raw database for exprs of given number of operands and values.
"""

from expr import Expr
from tqdm import tqdm
import random

def generate_exprs(num_operands, values, num_range, num_exprs):
    """
    Generate a raw database for exprs of given number of operands and values.

    Args:
        num_operands (list of int): number of operands in the exprs
        values (list of int): the value of the operands
        num_range (list of int): the range of the operands
        num_exprs (int): number of exprs to generate for each value

    Returns:
        list of Expr: the generated exprs
    """
    exprs = {
        num_operand: {
            value: [] for value in values
        }
        for num_operand in num_operands
    }

    # generate for each operands
    for num_operand in num_operands:
        # first, take all the exprs possible
        def gather_dict(target, dicts):
            for key in target:
                value = target[key]
                for d in dicts:
                    if key in d:
                        value.extend(d[key])
                target[key] = value
            return target

        def get_exprs(num_operand, operands=[], signs=[]):
            exprs = {value: [] for value in values}
            if len(operands) < num_operand:
                dicts = []
                for operand in num_range:
                    dicts.append(get_exprs(num_operand, operands + [operand], signs))
                exprs = gather_dict(exprs, dicts)
            elif len(signs) < num_operand:
                dicts = []
                for sign in [True, False]:
                    dicts.append(get_exprs(num_operand, operands, signs + [sign]))
                exprs = gather_dict(exprs, dicts)
            else:
                expr = Expr(signs, operands)
                value = expr.evaluate()
                if value in values:
                    exprs[value].append(expr.to_dict())
            return exprs
        
        exprs[num_operand] = get_exprs(num_operand)
        # print the count for each value
        print(f"Number of exprs for {num_operand} operands:")
        for value in values:
            print(f"{value}: {len(exprs[num_operand][value])}")
        print()

        # if more than num_exprs, randomly sample
        if num_exprs is not None:
            for value in values:
                if len(exprs[num_operand][value]) > num_exprs:
                    exprs[num_operand][value] = random.sample(exprs[num_operand][value], num_exprs)
    return exprs

import os, argparse, json

from utils import parse_range


def parse_args():
    parser = argparse.ArgumentParser(description='Generate exprs for mask-arithmetic.')
    parser.add_argument('--num_operands', type=str, help='The number of operands to generate the exprs for.', default='2-5')
    parser.add_argument('--values', type=str, help='The values to generate the exprs for.', default='0-10')
    parser.add_argument('--num_range', type=str, help='The range of the operands.', default=None)
    parser.add_argument('--num_exprs', type=int, help='Number of exprs to generate for each value.', default=5000)
    parser.add_argument('--output_base', type=str, help='The base path to save the output to.', default='data')
    return parser.parse_args()


def format_save_path(args):
    if not os.path.exists(os.path.join(args.output_base, args.values)):
        os.makedirs(os.path.join(args.output_base, args.values))

    return os.path.join(args.output_base, args.values, f"exprs_num_op{args.num_operands}_num_exprs{args.num_exprs}.json")


def main():
    args = parse_args()
    num_operands = parse_range(args.num_operands)
    values = parse_range(args.values)
    if not args.num_range:
        num_range = values
    else:
        num_range = parse_range(args.num_range)

    exprs = generate_exprs(num_operands, values, num_range, args.num_exprs)
    save_path = format_save_path(args)
    with open(save_path, 'w') as f:
        json.dump(exprs, f, indent=2)
    print(f"Exprs saved to {save_path}")

if __name__ == '__main__':
    main()