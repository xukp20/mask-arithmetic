"""
    Generate equal data like a +- b = c +- d. 
    Use all the numbers to generate all possible equations in the base set.

    Data structure:
    {
        "lhs_op_num": {
            "rhs_op_num": [
                {
                    "lhs_operands": [op1, op2, ...],
                    "rhs_operands": [op1, op2, ...],
                    "lhs_expr": "op1 + op2",
                    "rhs_expr": "op3 - op4",
                    ...
                },
                ...
            ],
            ...
        }
"""

import argparse, os, sys

def parse_args():
    parser = argparse.ArgumentParser(description='Generate equal data like a +- b = c +- d.')
    parser.add_argument('--set', type=str, help='The set of numbers to generate data for.', default='0-11')
    parser.add_argument('--output_base', type=str, help='The base path to save the output to.', default='data')

    # def settings
    parser.add_argument('--ops', type=list, help='The operators to use in the definition.', default=['add', 'sub'])
    parser.add_argument('--num_operands', type=list, help='The number of operands to use on each side of the equation.', default=[2])
    return parser.parse_args()


from expr import Constant, BinaryOp, Expr
import random, json

def format_save_path(args):
    return os.path.join(args.output_base, f"equal_{args.set}_{'&'.join(args.ops)}_{'&'.join([str(num) for num in args.num_operands])}.json")

def format_expr(operands, ops):
    expr = str(operands[0])
    for i, op in enumerate(ops):
        expr += f" {op} {operands[i+1]}"
    return expr

def format_equation(lhs_operands, lhs_ops, rhs_operands, rhs_ops):
    lhs_expr = format_expr(lhs_operands, lhs_ops)
    rhs_expr = format_expr(rhs_operands, rhs_ops)
    return f"{lhs_expr} = {rhs_expr}"

class EquationGenerator:
    def __init__(self, args):
        self.args = args
        self._load_number_set()

    def _load_number_set(self):
        self.number_set = self.parse_number_set(self.args.set)

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

    def generate_equations(self):
        # generate the expressions for the numbers in the base_set first
        exprs = self.generate_base_set_equations()

        # save
        self.save_equations(exprs)

    def save_equations(self, exprs):
        # Save the definition
        output_path = format_save_path(self.args)
        with open(output_path, "w") as f:
            json.dump(exprs, f, indent=4)
        print(f"Equations saved to {output_path}")

    def generate_base_set_equations(self):
        # Generate the equations for each combination of operand number counts
        lhs_op_nums = self.args.num_operands
        rhs_op_nums = self.args.num_operands
        exprs = {}
        for lhs_op_num in lhs_op_nums:
            exprs[lhs_op_num] = {}
            for rhs_op_num in rhs_op_nums:
                exprs[lhs_op_num][rhs_op_num] = self.generate_equations_for_op_nums(lhs_op_num, rhs_op_num)

        return exprs

    def generate_equations_for_op_nums(self, lhs_op_num, rhs_op_num):
        op_name2sym = {"add": "+", "sub": "-"}

        def gather_exprs(lhs_opreands=[], lhs_ops=[], rhs_operands=[], rhs_ops=[]):
            # recusively generate the equations
            exprs = []
            if len(lhs_opreands) < lhs_op_num:
                for opreand in self.number_set:
                    exprs.extend(gather_exprs(lhs_opreands + [opreand], lhs_ops, rhs_operands, rhs_ops))
            elif len(lhs_ops) < lhs_op_num - 1:
                for op in self.args.ops:
                    exprs.extend(gather_exprs(lhs_opreands, lhs_ops + [op], rhs_operands, rhs_ops))
            elif len(rhs_operands) < rhs_op_num:
                for opreand in self.number_set:
                    exprs.extend(gather_exprs(lhs_opreands, lhs_ops, rhs_operands + [opreand], rhs_ops))
            elif len(rhs_ops) < rhs_op_num - 1:
                for op in self.args.ops:
                    exprs.extend(gather_exprs(lhs_opreands, lhs_ops, rhs_operands, rhs_ops + [op]))
            else:
                # generate the expression
                lhs_ops = [op_name2sym[op] for op in lhs_ops]
                rhs_ops = [op_name2sym[op] for op in rhs_ops]
                lhs_expr = format_expr(lhs_opreands, lhs_ops)
                rhs_expr = format_expr(rhs_operands, rhs_ops)
                lhs_value = eval(lhs_expr)
                rhs_value = eval(rhs_expr)
                if lhs_value == rhs_value:
                    exprs.append({
                        "lhs_operands": lhs_opreands,
                        "lhs_ops": lhs_ops,
                        "rhs_operands": rhs_operands,
                        "rhs_ops": rhs_ops,
                        "lhs_expr": lhs_expr,
                        "rhs_expr": rhs_expr,
                        "equation": format_equation(lhs_opreands, lhs_ops, rhs_operands, rhs_ops)
                    })
            
            return exprs

        exprs = gather_exprs()
        print(f"Generated {len(exprs)} equations for {lhs_op_num} and {rhs_op_num} operands")
        return exprs
        
def main():
    args = parse_args()
    generator = EquationGenerator(args)
    generator.generate_equations()


if __name__ == "__main__":
    main()
