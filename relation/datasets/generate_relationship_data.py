"""
    Generate the relationship data from the definition data.
    1. Start from a root, build the tree of that number by replacing the numbers in the subtree with their definitions.
    2. For each tree, random sample paths from the root to a leaf.
    3. Choose relation order + 1 nodes from the path and generate the relation data.
"""

import os, argparse, sys, random, json, math

def parse_args():
    parser = argparse.ArgumentParser(description='Generate the relationship data from the definition data.')
    parser.add_argument('--def_set', type=str, help='The path to the definition data.', default='def_0-100_add_2.json')
    parser.add_argument('--output_base', type=str, help='The base path to save the output to.', default='data')
    parser.add_argument('--relation_order', type=int, help='The order of the relation.', default=2)
    parser.add_argument('--num_paths', type=int, help='The number of paths to sample per tree.', default=5)
    parser.add_argument('--num_relations', type=int, help='The number of relations to sample per path.', default=5)
    parser.add_argument('--seed', type=int, help='The random seed to use.', default=42)
    return parser.parse_args()


# data structure
from expr import Constant, BinaryOp, Expr

class RelationGenerator:
    def __init__(self, args):
        self.args = args
        self.load_definition()
    
    def load_definition(self):
        # Load the definition
        with open(self.args.def_set, "r") as f:
            self.def_set = json.load(f)
        # map keys to integers
        self.def_set = {int(k): v for k, v in self.def_set.items()}
    
    def generate_relation(self):
        # Generate the full tree
        trees = self.generate_trees()
        # Save trees
        self.save_trees(trees)
        
        # Generate the relation data
        relation_data = self.generate_relation_data(trees)
        # Save the relation data
        self.save_relation(relation_data)

    def save_trees(self, trees):
        # Save the trees
        # get base name from the input file
        input_file_name = os.path.basename(self.args.def_set).split('.')[0]
        tree_file_name = input_file_name.replace('def', 'tree') + ".json"
        output_path = os.path.join(self.args.output_base, tree_file_name)
        with open(output_path, "w") as f:
            json.dump(trees, f, indent=4)
        print(f"Trees saved to {output_path}")

    def save_relation(self, relation_data):
        # Save the relation data
        # get base name from the input file
        input_file_name = os.path.basename(self.args.def_set).split('.')[0]
        relation_file_name = input_file_name.replace('def', 'relation') + f"_{self.args.relation_order}_{self.args.num_paths}_{self.args.num_relations}.json"
        output_path = os.path.join(self.args.output_base, relation_file_name)
        with open(output_path, "w") as f:
            json.dump(relation_data, f, indent=4)
        print(f"Relation data saved to {output_path}")

    def generate_trees(self):
        # For each number, generate the tree
        trees = {}
        for target, def_expr in self.def_set.items():
            tree = self.generate_tree(def_expr)
            trees[target] = tree
        return trees

    def generate_tree(self, def_expr):
        # Generate the tree
        if def_expr is None:
            return None
        
        # Split a expr: if binop, split the left and right; if constant, look for the definition in the def_set
        def split_expr(expr):
            if isinstance(expr, Constant):
                constant_expr = self.def_set.get(expr.value, None)
                # recursion ends when the constant is not in the def_set
                if not constant_expr:
                    return expr
                else:
                    # split the constant expression also
                    return split_expr(constant_expr)

            elif isinstance(expr, BinaryOp):
                left = split_expr(expr.left)
                right = split_expr(expr.right)
                return BinaryOp(expr.op, left, right)

        tree = split_expr(def_expr)
        return tree

    def generate_relation_data(self, trees):
        relation_data = {}
        for target, tree in trees.items():
            # Generate the relation data
            relation_data[target] = self.generate_relation_data_for_tree(target, tree)

        return relation_data

    def generate_relation_data_for_tree(self, target, tree):
        # Generate the relation data for a tree
        if tree is None:
            return None
        
        # Generate the relation data
        paths = self.generate_paths(tree, self.args.num_paths, self.args.relation_order)
        relation_data = []
        for path in paths:
            relations = self.generate_relations(path, self.args.num_relations, self.args.relation_order)
            relation_data.append(relations)
        
        return relation_data

    def generate_paths(self, tree, num_paths, relation_order):
        # Choose random paths from the tree
        def check_same_path(path1, path2):
            if len(path1) != len(path2):
                return False
            for i in range(len(path1)):
                if path1[i] != path2[i]:
                    return False
            return True

        def check_exists(path, paths):
            for p in paths:
                if check_same_path(p, path):
                    return True
            return False
        
        def compute_max_paths(tree):
            if tree is None:
                return 0
            if isinstance(tree, Constant):
                return 1
            if isinstance(tree, BinaryOp):
                return compute_max_paths(tree.left) + compute_max_paths(tree.right)

        paths = []
        max_paths = compute_max_paths(tree)
        if max_paths < num_paths:
            print(f"Warning: only {max_paths} paths in the tree for target {tree.evaluate()}")
            num_paths = max_paths

        MAX_TRY=5000
        times = 0
        while len(paths) < num_paths:
            path = self.choose_path(tree)
            if len(path) >= relation_order + 1 and not check_exists(path, paths):
                paths.append(path)
                times = 0
            times += 1
        
        return paths

    def choose_path(self, tree):
        # Choose a random path from the tree
        path = []
        node = tree
        while node is not None:
            path.append(node.evaluate())
            if isinstance(node, BinaryOp):
                if random.choice([True, False]):
                    node = node.left
                else:
                    node = node.right
            else:
                break
        
        return path

    def generate_relations(self, path, num_relations, relation_order):
        # Generate the relations
        def check_same_relation(rel1, rel2):
            if len(rel1) != len(rel2):
                return False
            for i in range(len(rel1)):
                if rel1[i] != rel2[i]:
                    return False
            return True
        
        def check_exists(relation, relations):
            for r in relations:
                if check_same_relation(r, relation):
                    return True
            return False
        
        def compute_max_relations(path):
            # the number of ways to choose relation order nodes from the path except the first
            path_len = len(path) - 1
            comb = math.comb(path_len, relation_order)
            return comb

        relations = []
        max_relations = compute_max_relations(path)
        if max_relations < num_relations:
            print(f"Warning: only {max_relations} relations in the path {path}")

        while len(relations) < num_relations:
            relation = self.choose_relation(path)
            relations.append(relation)
        
        return relations
