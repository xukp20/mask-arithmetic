"""
    Generate definition data for all the numbers in the given set.
"""

import argparse, os, sys

def parse_args():
    parser = argparse.ArgumentParser(description='Generate definition data for all the numbers in the given set.')
    parser.add_argument('--set', type=str, help='The set of numbers to generate the definition for.', default='0-11')
    parser.add_argument('--base_set', type=str, help='The set of numbers to form the definition from.', default=None)
    parser.add_argument('--output_base', type=str, help='The base path to save the output to.', default='data')

    # def settings
    parser.add_argument('--ops', type=list, help='The operators to use in the definition.', default=['add', 'sub'])
    parser.add_argument('--num_operands', type=list, help='The number of operands to use in the definition.', default=[2])
    return parser.parse_args()


from expr import Constant, BinaryOp, Expr
import random, json

def format_save_path(args):
    return os.path.join(args.output_base, f"def_{args.set}_{args.base_set if args.base_set else args.set}_{'&'.join(args.ops)}_{'&'.join([str(num) for num in args.num_operands])}.json")

class DefinitionGenerator:
    def __init__(self, args):
        self.args = args
        self._load_number_set()

    def _load_number_set(self):
        self.number_set = self.parse_number_set(self.args.set)
        self.base_set = self.parse_number_set(self.args.base_set) if self.args.base_set else self.number_set

        # look for the extrapolate number that is not in the base set
        self.extrapolate_set = self.number_set - self.base_set

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

    def generate_definition(self):
        # generate the expressions for the numbers in the base_set first
        exprs = self.generate_base_set_definition()

        # generate the expressions for the numbers in the extrapolate_set
        exprs.update(self.generate_extrapolate_set_definition())

        # save
        self.save_definition(exprs)

    def save_definition(self, exprs):
        # Save the definition
        output_path = format_save_path(self.args)
        if not os.path.exists(self.args.output_base):
            os.makedirs(self.args.output_base)

        with open(output_path, "w") as f:
            json.dump(exprs, f, indent=4)
        print(f"Definition saved to {output_path}")

    def generate_base_set_definition(self):
        # go through all the possible combinations of operators and number of operands
        all_exprs = {}
        for num_operands in self.args.num_operands:
            all_exprs[num_operands] = self.generate_base_set_exprs_for_num_operands(num_operands)

        # for each number in the base set, find the expression that contains it
        def check_contains(expr, target):
            if target == expr["lhs"]:
                return True
            for i, operand in enumerate(expr["rhs_operands"]):
                if target == operand:
                    return True
            return False

        base_set_exprs = {}
        for num in self.base_set:
            base_set_exprs[num] = {}
            for num_operands in self.args.num_operands:
                base_set_exprs[num][num_operands] = []
                for expr in all_exprs[num_operands]:
                    if check_contains(expr, num):
                        base_set_exprs[num][num_operands].append(expr)

        return base_set_exprs

    def generate_base_set_exprs_for_num_operands(self, num_operands):
        def op_name2sym(op_name):
            op_name2sym = {"add": "+", "sub": "-", "mul": "*", "div": "/"}
            return op_name2sym[op_name]

        def gather_exprs(lhs=None, rhs_operands=[], rhs_ops=[]):
            if lhs is None:
                all_exprs = []
                for operand in self.base_set:
                    all_exprs.extend(gather_exprs(operand, rhs_operands=rhs_operands, rhs_ops=rhs_ops))
            elif len(rhs_operands) < num_operands:
                # choose operands first
                all_exprs = []
                for operand in self.base_set:
                    all_exprs.extend(gather_exprs(lhs, rhs_operands=rhs_operands+[operand], rhs_ops=rhs_ops))
            elif len(rhs_ops) < num_operands - 1:
                # choose operators
                all_exprs = []
                for op in self.args.ops:
                    all_exprs.extend(gather_exprs(lhs, rhs_operands=rhs_operands, rhs_ops=rhs_ops+[op]))
            else:
                # generate the expression and check if it is valid
                all_exprs = []
                rhs_expr = ""
                # NOTE: only for add and sub, don't need to consider the priority
                for i, op in enumerate(rhs_ops):
                    rhs_expr += f"{rhs_operands[i]} {op_name2sym(op)} "
                rhs_expr += str(rhs_operands[-1])
                rhs = eval(rhs_expr)
                if rhs == lhs:
                    all_exprs.append({
                        "lhs": lhs,
                        "rhs_expr": rhs_expr,
                        "rhs_operands": rhs_operands,
                        "rhs_ops": [op_name2sym(op) for op in rhs_ops]
                    })
            return all_exprs

        print(f"Generating base set expressions for {num_operands} operands")
        all_exprs = gather_exprs()
        return all_exprs

    def generate_extrapolate_set_definition(self):
        # for each number in the extrapolate set, find the expression that contains it once, while the other operand is in the base set
        interpolate_set_exprs = {}
        for number in self.extrapolate_set:
            # look for the exprs
            interpolate_set_exprs[number] = {}
            for num_operands in self.args.num_operands:
                interpolate_set_exprs[number][num_operands] = self.generate_extrapolate_set_exprs_for_num_operands(number, num_operands)

        return interpolate_set_exprs

    def generate_extrapolate_set_exprs_for_num_operands(self, number, num_operands):
        # modified from generate_base_set_exprs_for_num_operands
        # needs to choose one place to put the number
        def op_name2sym(op_name):
            op_name2sym = {"add": "+", "sub": "-", "mul": "*", "div": "/"}
            return op_name2sym[op_name]
        
        def eval_expr(lhs, rhs_operands, rhs_ops):
            rhs_expr = ""
            for i, op in enumerate(rhs_ops):
                rhs_expr += f"{rhs_operands[i]} {op_name2sym(op)} "
            rhs_expr += str(rhs_operands[-1])
            return eval(rhs_expr) == lhs, rhs_expr

        def gather_exprs(lhs=None, rhs_operands=[], rhs_ops=[]):
            def remove_duplicates(exprs):
                seen = set()
                new_exprs = []
                for expr in exprs:
                    key = f"{expr['lhs']}_{expr['rhs_expr']}"
                    if key not in seen:
                        seen.add(key)
                        new_exprs.append(expr)
                return new_exprs

            if lhs is None:
                all_exprs = []
                for operand in self.base_set:
                    all_exprs.extend(gather_exprs(operand, rhs_operands=rhs_operands, rhs_ops=rhs_ops))
            elif len(rhs_operands) < num_operands:
                # choose operands first
                all_exprs = []
                for operand in self.base_set:
                    all_exprs.extend(gather_exprs(lhs, rhs_operands=rhs_operands+[operand], rhs_ops=rhs_ops))
            elif len(rhs_ops) < num_operands - 1:
                # choose operators
                all_exprs = []
                for op in self.args.ops:
                    all_exprs.extend(gather_exprs(lhs, rhs_operands=rhs_operands, rhs_ops=rhs_ops+[op]))
            else:
                # generate the expression and check if it is valid
                all_exprs = []
                # NOTE: only for add and sub, don't need to consider the priority
                
                # now replace each of the operands with the number and check if it is valid
                # 1. lhs
                lhs_valid, rhs_expr = eval_expr(number, rhs_operands, rhs_ops)
                if lhs_valid:
                    all_exprs.append({
                        "lhs": number,
                        "rhs_expr": rhs_expr,
                        "rhs_operands": rhs_operands,
                        "rhs_ops": [op_name2sym(op) for op in rhs_ops]
                    })
                # 2. each of the rhs_operands
                for i, operand in enumerate(rhs_operands):
                    new_rhs_operands = rhs_operands.copy()
                    new_rhs_operands[i] = number
                    rhs_valid, rhs_expr = eval_expr(lhs, new_rhs_operands, rhs_ops)
                    if rhs_valid:
                        all_exprs.append({
                            "lhs": lhs,
                            "rhs_expr": rhs_expr,
                            "rhs_operands": new_rhs_operands,
                            "rhs_ops": [op_name2sym(op) for op in rhs_ops]
                        })
            
            all_exprs = remove_duplicates(all_exprs)
            return all_exprs

        all_exprs = gather_exprs()

        return all_exprs


if __name__ == "__main__":
    args = parse_args()
    generator = DefinitionGenerator(args)
    generator.generate_definition()