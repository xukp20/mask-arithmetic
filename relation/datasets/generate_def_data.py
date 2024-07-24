"""
    Generate a single definition for all the numbers in the given set.
"""

import argparse, os, sys

def parse_args():
    parser = argparse.ArgumentParser(description='Generate a single definition for all the numbers in the given set.')
    parser.add_argument('--set', type=str, help='The set of numbers to generate the definition for.', default='0-100')
    parser.add_argument('--base_set', type=str, help='The set of numbers to form the definition from.', default=None)
    parser.add_argument('--output_base', type=str, help='The base path to save the output to.', default='data')

    # def settings
    parser.add_argument('--operator', type=str, help='The operator to use in the definition.', default='add', choices=['add', 'sub', 'mul', 'div'])
    parser.add_argument('--num_operands', type=int, help='The number of operands to use in the definition.', default=2)
    parser.add_argument('--order', help='The definition of any number should only use numbers before it.', action='store_true')
    return parser.parse_args()


from expr import Constant, BinaryOp, Expr
import random, json
class DefinitionGenerator:
    def __init__(self, args):
        self.args = args

    def generate_definition(self):
        # Generate the definition
        def_set = self.generate_definition_string()
        # Save the definition
        self.save_definition(def_set=def_set)

    def save_definition(self, def_set):
        # Save the definition
        output_path = os.path.join(self.args.output_base, f"def_{self.args.set}_{self.args.operator}_{self.args.num_operands}.json")
        with open(output_path, "w") as f:
            json.dump(def_set, f, indent=4)
        print(f"Definition saved to {output_path}")

    def generate_definition_string(self):
        # Generate the definition
        operator = self.args.operator
        op_name2sym = {"add": "+", "sub": "-", "mul": "*", "div": "/"}
        operator = op_name2sym[operator]

        num_operands = self.args.num_operands
        
        # only support num_operands = 2
        if num_operands != 2:
            raise ValueError(f"Unsupported number of operands {num_operands}")
        
        # Generate the definition
        number_set = self.parse_number_set(self.args.set)
        base_set = self.parse_number_set(self.args.base_set) if self.args.base_set else number_set
        # sort
        number_set = sorted(number_set)
        base_set = sorted(base_set)

        def_set = {}
        for target in number_set:
            # Generate the definition
            definition = self.generate_definition_for_number(target, base_set, operator)
            print(f"Definition for {target}: {definition}")
            def_set[target] = definition.to_dict() if definition else None

        return def_set
    
    def generate_definition_for_number(self, target, base_set, operator):
        # random choose operands until match the target
        MAX_TRY=5e7
        times = 0
        while times < MAX_TRY:
            if self.args.order:
                # only use numbers before the target
                base_set = [x for x in base_set if x < target]

            op1 = random.choice(base_set)
            op2 = random.choice(base_set)
            expr = BinaryOp(operator, Constant(op1), Constant(op2))
            if expr.evaluate() == target:
                return expr
            times += 1
        
        print(f"Failed to generate definition for {target} after {MAX_TRY} tries.")
        return None

    def parse_number_set(self, set_str):
        # a-b,c-d
        ranges = set_str.split(",")
        numbers = set()
        for r in ranges:
            if "-" in r:
                start, end = map(int, r.split("-"))
                numbers.update(range(start, end))
            else:
                numbers.add(int(r))
        return numbers


if __name__ == "__main__":
    args = parse_args()
    generator = DefinitionGenerator(args)
    generator.generate_definition()