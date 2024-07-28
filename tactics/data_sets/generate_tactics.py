"""
    Generate the tactics from an expr to another expr.
"""

from expr import Expr
from tactic import Tactic, ALL_TACTICS

import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Generate tactics for mask-arithmetic.')
    parser.add_argument('--input_file', type=str, help='The path to the input file.', default='exprs_num_op2-5_values0-10_num_range0-10_num_exprs1000.json')
    parser.add_argument('--output_base', type=str, help='The base path to save the output to.', default='data')

    # settings: the range of operand numbers, total count
    parser.add_argument('--num_operands', type=str, help='The range of operand numbers.', default='2-5')
    parser.add_argument('--count', type=int, help='The count of exprs to generate for each value.', default=100000)
    parser.add_argument('--upper', type=int, help='The upper bound of the numbers.', default=10)
    parser.add_argument('--lower', type=int, help='The lower bound of the numbers.', default=0)
    parser.add_argument('--seeds', type=int, help='The seeds for random sampling.', default=42)
    return parser.parse_args()


class ExprFileParser:
    """
        Structure:
        {
            "num_operands": {
                "value": [
                    {
                        "signs": [],
                        "nums": []
                    }
                ]
            }
        }
    """

    def __init__(self, filename):
        self.filename = filename
        self.exprs = self.load_exprs()

    def load_exprs(self):
        with open(self.filename, 'r') as f:
            return json.load(f)

    def get_values(self):
        value_strs = list(list(self.exprs.values())[0].keys())
        return list(set([int(value) for value in value_strs]))

    def get_expr(self, num_operands, value):
        # if num_operands is a list, flatten the exprs 
        exprs = []
        if isinstance(num_operands, list):
            for num_op in num_operands:
                exprs.extend(self.exprs[str(num_op)][str(value)])
        else:
            exprs = self.exprs[str(num_operands)][str(value)]
        return [Expr.from_dict(expr) for expr in exprs]

    def get_expr_size(self, num_operands, value=-1):
        if isinstance(num_operands, list):
            size = 0
            for num_op in num_operands:
                size += len(self.exprs[str(num_op)][str(value)])
        else:
            size = len(self.exprs[str(num_operands)][str(value)])
        
        return size


import os, json, random
from tqdm import tqdm
from utils import parse_range
from proof import Proof, ProofStep
from collections import deque
import copy

class TacticGenerator:
    def __init__(self, args):
        self.old_args = args
        args.num_operands = list(parse_range(args.num_operands))
        self.args = args
        input_path = os.path.join(args.output_base, args.input_file)
        self.expr_parser = ExprFileParser(input_path)

    def generate_tactics(self):
        # take all the exprs for each value
        values = self.expr_parser.get_values()
        all_proofs = []
        for value in values:
            exprs = self.expr_parser.get_expr(self.args.num_operands, value)
            all_proofs.extend(self.generate_tactics_for_value(exprs, value))
        # group by the length of the proofs
        proof_dict = {}
        for proof in all_proofs:
            length = len(proof)
            if length not in proof_dict:
                proof_dict[length] = []
            proof_dict[length].append(proof)
        # save
        for length, proofs in proof_dict.items():
            size = len(proofs)
            print(f"Saving {size} proofs of length {length}")
            output_path = self.format_save_path(length, size)
            with open(output_path, 'w') as f:
                for proof in proofs:
                    f.write(json.dumps(proof.to_dict()) + '\n')

    def generate_tactics_for_value(self, exprs, value):
        # take a pair from the exprs to the source and target 
        # then apply bfs for the best path
        exprs_count = len(exprs)
        max_count = exprs_count * (exprs_count - 1)
        sample_count = self.args.count

        proofs = []
        if sample_count > max_count:
            print(f"Warning: sample count {sample_count} is larger than the max count {max_count}, set to max count.")
            sample_count = max_count
            # go through all the pair
            tbar = tqdm(total=max_count, desc='Generating proofs')
            for i in range(exprs_count):
                for j in range(exprs_count):
                    if i == j:
                        continue
                    source = exprs[i]
                    target = exprs[j]
                    proof = self.generate_proof(source, target)
                    if proof is not None:
                        proofs.append(proof)
                    tbar.update(1)
            tbar.close()
        else:
            bitmap = [
                [False for _ in range(exprs_count)]
                for _ in range(exprs_count)
            ]
            for i in range(exprs_count):
                bitmap[i][i] = True

            def sample_pair(exprs_count, bitmap):
                i = random.randint(0, exprs_count - 1)
                j = random.randint(0, exprs_count - 1)
                while bitmap[i][j]:
                    i = random.randint(0, exprs_count - 1)
                    j = random.randint(0, exprs_count - 1)
                bitmap[i][j] = True
                return i, j
            
            proofs = []
            tbar = tqdm(total=sample_count, desc='Generating proofs')
            while len(proofs) < sample_count:
                i, j = sample_pair(exprs_count, bitmap)
                source = exprs[i]
                target = exprs[j]
                proof = self.generate_proof(source, target)
                if proof is not None:
                    proofs.append(proof)
                    tbar.update(1)
            tbar.close()

        # print count
        tqdm.write(f"Generated {len(proofs)} proofs for value {value}")
        return proofs
    
    def format_save_path(self, length, size):
        return os.path.join(self.args.output_base, f"tactics_{length}_num_op{self.old_args.num_operands}_{size}.jsonl")

    def generate_proof(self, source, target):
        # apply bfs to find the best path
        # the path is a list of proof steps
        
        def bfs(source: Expr, target: Expr, lower: int = 0, upper: int = 10, max_steps: int = 20):
            queue = deque([(source, [])])
            visited = set()
            visited.add(str(source))

            while queue:
                current_expr, current_proof = queue.popleft()
                print(f"Current proof: {Proof(source, target, current_proof)}")

                if current_expr == target:
                    return Proof(source, target, current_proof)

                if len(current_proof) >= max_steps:
                    continue

                for tactic_name, tactic_class in ALL_TACTICS.items():
                    for position in range(len(current_expr.nums)):
                        new_expr = tactic_class.apply(current_expr, position, lower, upper)
                        print(f"current_expr: {current_expr}")
                        print(f"{tactic_name}: {position} -> {new_expr}")
                        if new_expr is not None and str(new_expr) not in visited:
                            visited.add(str(new_expr))
                            new_proof = copy.deepcopy(current_proof)
                            print(ProofStep(tactic_name, position, new_expr))
                            new_proof.append(ProofStep(tactic_name, position, new_expr))
                            queue.append((new_expr, new_proof))

            return None  # No solution found within max_steps

        from transform_alg import get_path
        
        # bfs(source, target, self.args.lower, self.args.upper)
        steps = get_path(source, target, self.args.lower, self.args.upper)
        proof = Proof(source, target, steps)
        if not proof.check_valid():
            print(f"Warning: Invalid proof found: {proof}")
            exit()
            return None

        return proof

def main():
    args = parse_args()
    random.seed(args.seeds)
    generator = TacticGenerator(args)
    generator.generate_tactics()


if __name__ == '__main__':
    main()