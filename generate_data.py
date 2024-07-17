"""
    Generate a sample for compound arithmetic operation.
    [+, -, *, /] in the range of 0-MAX_NUMBER, mod MOD for each answer.
"""

from tokenizer import MAX_NUMBER
MOD=101
assert MAX_NUMBER < MOD, f"MAX_NUMBER should be less than {MOD}"

NUMBER_SET=list(range(MAX_NUMBER + 1))

class Bracket:
    def __init__(self, expr):
        self.expr = expr

    def __str__(self):
        return f"( {self.expr} )"

    def to_dict(self):
        return {"type": "brac", "expr": self.expr.to_dict()}

    def evaluate(self):
        return self.expr.evaluate()

    def random_split(self, op_ratios, upper_op=None, is_left=False):
        self.expr = self.expr.random_split(op_ratios, "brac", is_left)
        return self

    def __eq__(self, other):
        return isinstance(other, Bracket) and self.expr == other.expr


# abstract class for expression
class Expr:
    def __init__(self):
        pass

    def __str__(self):
        return ""

    def to_dict(self):
        return {"type": "expr"}

    def evaluate(self):
        return 0

    def random_split(self, op_ratios):
        raise NotImplementedError


class Constant(Expr):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return str(self.value)

    def to_dict(self):
        return {"type": "const", "value": self.value}

    def evaluate(self):
        return self.value

    def random_split(self, op_ratios, upper_op=None, is_left=False):
        # split the constant into an operation
        op = random.choices(["+", "-", "*", "/"], weights=op_ratios)[0]
        # sample op until the result is the constant
        while True:
            left = random.choice(NUMBER_SET)
            right = random.choice(NUMBER_SET)
            if op == "+":
                tar_value = (left + right) % MOD
            elif op == "-":
                tar_value = (left - right + MOD) % MOD
            elif op == "*":
                tar_value = (left * right) % MOD
            elif op == "/":
                if right == 0:
                    continue

                tar_value = (left // right) % MOD
            if tar_value == self.value:
                break

        expr = BinaryOp(op, Constant(left), Constant(right))

        # check if needs to add bracket
        is_right = not is_left
        if upper_op == "brac":
            return expr

        if op in ["+", "-"]:
            if upper_op in ["*", "/"]:
                return Bracket(expr)
            elif upper_op in ["+", "-"] and is_right:
                return Bracket(expr)
            else:
                return expr
        elif op in ["*", "/"]:
            if upper_op in ["*", "/"] and is_right:
                return Bracket(expr)
            else:
                return expr

    def __eq__(self, other):
        return isinstance(other, Constant) and self.value == other.value


import random
class BinaryOp(Expr):
    def __init__(self, op, left, right):
        self.op = op
        self.left = left
        self.right = right

    def __str__(self):
        return f"{self.left} {self.op} {self.right}"

    def to_dict(self):
        return {"type": "binop", "op": self.op, "left": self.left.to_dict(), "right": self.right.to_dict()}

    def evaluate(self):
        left = self.left.evaluate()
        right = self.right.evaluate()
        if self.op == "+":
            return (left + right) % MOD
        elif self.op == "-":
            return (left - right + MOD) % MOD
        elif self.op == "*":
            return (left * right) % MOD
        elif self.op == "/":
            return (left // right) % MOD
        else:
            raise ValueError(f"Unsupported operation {self.op}")

    def random_split(self, op_ratios, upper_op=None, is_left=False):
        if random.random() < 0.5:
            self.left = self.left.random_split(op_ratios, self.op, is_left=True)
        else:
            self.right = self.right.random_split(op_ratios, self.op, is_left=False)
        
        return self

    def __eq__(self, other):
        return isinstance(other, BinaryOp) and self.op == other.op and self.left == other.left and self.right == other.right


def generate_expr(depth, op_ratios, value=None):
    # init constant
    if value is None:
        root = Constant(random.choice(NUMBER_SET))
    else:
        root = Constant(value)

    ans = root.evaluate()
    for _ in range(depth):
        root = root.random_split(op_ratios)

    return root, ans


### Generate samples for MLM tasks
from tokenizer import DEFAULT_TOKENIZER

EXPRESSION_PATTERN="{} = {}"
from tqdm import tqdm
def generate_expressions(n_samples, max_depth, op_ratios):
    samples = []
    for _ in tqdm(range(n_samples)):
        expr, ans = generate_expr(max_depth, op_ratios)
        expr_str = str(expr)
        expr = EXPRESSION_PATTERN.format(ans, expr_str)
        # expr = EXPRESSION_PATTERN.format(expr_str, ans)
        samples.append(expr)

    return samples


# add definition and equality tasks
DEFINITION_PATTERN="{} : = {}"
def generate_definition_samples(sample_values, max_depth, op_ratios):
    samples = []
    for value in sample_values:
        expr, _ = generate_expr(max_depth, op_ratios, value)
        expr_str = str(expr)
        expr = DEFINITION_PATTERN.format(value, expr_str)
        samples.append(expr)

    return samples


EQUALITY_PATTERN="{} = {}"
def generate_equality_samples(num_samples, max_depth, op_ratios):
    samples = []
    for _ in range(num_samples):
        value = random.choice(NUMBER_SET)
        expr1, _ = generate_expr(max_depth, op_ratios, value)
        expr2, _ = generate_expr(max_depth, op_ratios, value)
        expr1_str = str(expr1)
        expr2_str = str(expr2)
        expr = EQUALITY_PATTERN.format(expr1_str, expr2_str)
        samples.append(expr)

    return samples

IGNORE_INDEX=-100
import random
def generate_mlm_samples(n_samples, max_depth, op_ratios, max_length, ans_mask_ratio=0.5):
    samples = generate_expressions(n_samples, max_depth, op_ratios)
    mlm_samples = []
    for sample in tqdm(samples):
        encoded = DEFAULT_TOKENIZER.encode(sample, max_length=max_length, padding=True)
        # choose one of the numbers to mask
        number_mask = DEFAULT_TOKENIZER.get_number_mask(encoded['input_ids'])
        masked_idx = [i for i, mask in enumerate(number_mask) if mask == 1]
        if random.random() < ans_mask_ratio:
            masked_idx = masked_idx[0]
        else:
            masked_idx = random.choice(masked_idx[1:])

        labels = [IGNORE_INDEX] * len(encoded['input_ids'])
        labels[masked_idx] = encoded['input_ids'][masked_idx]
        encoded['input_ids'][masked_idx] = DEFAULT_TOKENIZER.get_mask_token_id()
        mlm_samples.append({
            'input_ids': encoded['input_ids'],
            'attention_mask': encoded['attention_mask'],
            'labels': labels,
        })

    return mlm_samples

def generate_definition_mlm_samples(sample_values, max_depth, op_ratios, max_length, def_mask_ratio=0.5):
    samples = generate_definition_samples(sample_values, max_depth, op_ratios)
    mlm_samples = []
    for sample in tqdm(samples):
        encoded = DEFAULT_TOKENIZER.encode(sample, max_length=max_length, padding=True)
        # choose one of the numbers to mask
        number_mask = DEFAULT_TOKENIZER.get_number_mask(encoded['input_ids'])
        masked_idx = [i for i, mask in enumerate(number_mask) if mask == 1]
        if random.random() < def_mask_ratio:
            masked_idx = masked_idx[0]
        else:
            masked_idx = random.choice(masked_idx[1:])

        labels = [IGNORE_INDEX] * len(encoded['input_ids'])
        labels[masked_idx] = encoded['input_ids'][masked_idx]
        encoded['input_ids'][masked_idx] = DEFAULT_TOKENIZER.get_mask_token_id()
        mlm_samples.append({
            'input_ids': encoded['input_ids'],
            'attention_mask': encoded['attention_mask'],
            'labels': labels,
        })

    return mlm_samples


def generate_equality_mlm_samples(num_samples, max_depth, op_ratios, max_length):
    samples = generate_equality_samples(num_samples, max_depth, op_ratios)
    mlm_samples = []
    for sample in tqdm(samples):
        encoded = DEFAULT_TOKENIZER.encode(sample, max_length=max_length, padding=True)
        # choose one of the numbers to mask
        number_mask = DEFAULT_TOKENIZER.get_number_mask(encoded['input_ids'])
        masked_idx = [i for i, mask in enumerate(number_mask) if mask == 1]
        masked_idx = random.choice(masked_idx)

        labels = [IGNORE_INDEX] * len(encoded['input_ids'])
        labels[masked_idx] = encoded['input_ids'][masked_idx]
        encoded['input_ids'][masked_idx] = DEFAULT_TOKENIZER.get_mask_token_id()
        mlm_samples.append({
            'input_ids': encoded['input_ids'],
            'attention_mask': encoded['attention_mask'],
            'labels': labels,
        })

    return mlm_samples


# # Generate samples
# n_samples = 10
# max_depth = 3
# op_ratios = [0.25, 0.25, 0.25, 0.25]
# max_length = 32
# mlm_samples = generate_mlm_samples(n_samples, max_depth, op_ratios, max_length)

# # print 1
# sample = mlm_samples[0]
# print("Sample 1:")
# print("Input:", DEFAULT_TOKENIZER.decode(sample['input_ids']))
# print("Labels:", DEFAULT_TOKENIZER.decode(sample['labels']))
# print("Attention Mask:", sample['attention_mask'])

def print_sample(sample):
    print("Input:", DEFAULT_TOKENIZER.decode(sample['input_ids']))
    print("Labels:", DEFAULT_TOKENIZER.decode(sample['labels']))
    print("Attention Mask:", sample['attention_mask'])


# save the dataset
DATA_BASE = "./data"
import os, json
def generate_datasets(train_count, eval_count, max_depth, op_ratios, max_length, ans_mask_ratio):
    # evaluation dataset:
    # 1. only ans mask
    eval_samples = generate_mlm_samples(eval_count, max_depth, op_ratios, max_length, 1.0)
    eval_path = os.path.join(DATA_BASE, "eval_ans.jsonl")

    with open(eval_path, 'w') as f:
        for sample in eval_samples:
            f.write(json.dumps(sample) + "\n")
    print_sample(eval_samples[0])
    
    # 2. only number mask
    eval_samples = generate_mlm_samples(eval_count, max_depth, op_ratios, max_length, 0.0)
    eval_path = os.path.join(DATA_BASE, "eval_num.jsonl")

    with open(eval_path, 'w') as f:
        for sample in eval_samples:
            f.write(json.dumps(sample) + "\n")
    print_sample(eval_samples[0])

    # 3. both ans and number mask
    eval_samples = generate_mlm_samples(eval_count, max_depth, op_ratios, max_length, 0.5)
    eval_path = os.path.join(DATA_BASE, "eval_mix.jsonl")

    with open(eval_path, 'w') as f:
        for sample in eval_samples:
            f.write(json.dumps(sample) + "\n")
    print_sample(eval_samples[0])

    # training dataset
    train_samples = generate_mlm_samples(train_count, max_depth, op_ratios, max_length, ans_mask_ratio)
    train_path = os.path.join(DATA_BASE, "train_mix.jsonl")
    with open(train_path, 'w') as f:
        for sample in train_samples:
            f.write(json.dumps(sample) + "\n")
    print_sample(train_samples[0])

    train_samples = generate_mlm_samples(train_count, max_depth, op_ratios, max_length, 0.0)
    train_path = os.path.join(DATA_BASE, "train_num.jsonl")
    with open(train_path, 'w') as f:
        for sample in train_samples:
            f.write(json.dumps(sample) + "\n")
    print_sample(train_samples[0])

    train_samples = generate_mlm_samples(train_count, max_depth, op_ratios, max_length, 1.0)
    train_path = os.path.join(DATA_BASE, "train_ans.jsonl")
    with open(train_path, 'w') as f:
        for sample in train_samples:
            f.write(json.dumps(sample) + "\n")
    print_sample(train_samples[0])


def generate_def_datasets(train_sample_values, eval_count, max_depth, op_ratios, max_length, mask_num_per_value=1):
    # evaluation dataset:
    # 1. randomly choose eval_count samples, with def mask ratio 1.0
    # sample values from the NUMBER_SET, allow duplicates
    eval_sample_values = random.choices(NUMBER_SET, k=eval_count)
    eval_samples = generate_definition_mlm_samples(eval_sample_values, max_depth, op_ratios, max_length, 1.0)

    eval_path = os.path.join(DATA_BASE, "eval_def_def.jsonl")
    with open(eval_path, 'w') as f:
        for sample in eval_samples:
            f.write(json.dumps(sample) + "\n")
    eval_def_samples = eval_samples
    print("Eval def size:", len(eval_def_samples))
    print_sample(eval_samples[0])

    # 2. randomly choose eval_count samples, with def mask ratio 0.0
    eval_count = eval_count * mask_num_per_value    # match training dataset
    eval_sample_values = random.choices(NUMBER_SET, k=eval_count)
    eval_samples = generate_definition_mlm_samples(eval_sample_values, max_depth, op_ratios, max_length, 0.0)

    eval_path = os.path.join(DATA_BASE, "eval_def_num.jsonl")
    with open(eval_path, 'w') as f:
        for sample in eval_samples:
            f.write(json.dumps(sample) + "\n")
    eval_num_samples = eval_samples
    print("Eval num size:", len(eval_num_samples))
    print_sample(eval_samples[0])

    # 3. concat
    eval_samples = eval_def_samples + eval_num_samples

    eval_path = os.path.join(DATA_BASE, "eval_def_mix.jsonl")
    with open(eval_path, 'w') as f:
        for sample in eval_samples:
            f.write(json.dumps(sample) + "\n")
    print("Eval mix size:", len(eval_samples))

    # training dataset
    # for each value, can only sample one expr for the definition
    train_expr = generate_definition_samples(train_sample_values, max_depth, op_ratios)
    
    # 1. sample 1 for each expr with the ans masked
    train_ans_mask_samples = []
    for expr in train_expr:
        encoded = DEFAULT_TOKENIZER.encode(expr, max_length=max_length, padding=True)
        number_mask = DEFAULT_TOKENIZER.get_number_mask(encoded['input_ids'])
        masked_idx = [i for i, mask in enumerate(number_mask) if mask == 1]
        masked_idx = masked_idx[0]
        labels = [IGNORE_INDEX] * len(encoded['input_ids'])
        labels[masked_idx] = encoded['input_ids'][masked_idx]
        encoded['input_ids'][masked_idx] = DEFAULT_TOKENIZER.get_mask_token_id()
        train_ans_mask_samples.append({
            'input_ids': encoded['input_ids'],
            'attention_mask': encoded['attention_mask'],
            'labels': labels,
        })
    
    train_path = os.path.join(DATA_BASE, "train_def_def.jsonl")
    with open(train_path, 'w') as f:
        for sample in train_ans_mask_samples:
            f.write(json.dumps(sample) + "\n")
    print("Train def size:", len(train_ans_mask_samples))
    print_sample(train_ans_mask_samples[0])

    # 2. sample mask_num_per_value for each expr with the number masked
    train_num_mask_samples = []
    for expr in train_expr:
        encoded = DEFAULT_TOKENIZER.encode(expr, max_length=max_length, padding=True)
        number_mask = DEFAULT_TOKENIZER.get_number_mask(encoded['input_ids'])
        masked_idx = [i for i, mask in enumerate(number_mask) if mask == 1]
        masked_idx = random.choices(masked_idx[1:], k=mask_num_per_value)
        for idx in masked_idx:
            labels = [IGNORE_INDEX] * len(encoded['input_ids'])
            labels[idx] = encoded['input_ids'][idx]
            encoded['input_ids'][idx] = DEFAULT_TOKENIZER.get_mask_token_id()
            train_num_mask_samples.append({
                'input_ids': encoded['input_ids'],
                'attention_mask': encoded['attention_mask'],
                'labels': labels,
            })
    
    train_path = os.path.join(DATA_BASE, "train_def_num.jsonl")
    with open(train_path, 'w') as f:
        for sample in train_num_mask_samples:
            f.write(json.dumps(sample) + "\n")
    print("Train num size:", len(train_num_mask_samples))
    print_sample(train_num_mask_samples[0])

    # 3. concat
    train_samples = train_ans_mask_samples + train_num_mask_samples
    train_path = os.path.join(DATA_BASE, "train_def_mix.jsonl")

    with open(train_path, 'w') as f:
        for sample in train_samples:
            f.write(json.dumps(sample) + "\n")
    print("Train mix size:", len(train_samples))
    
    return train_samples, eval_samples


def generate_equality_datasets(train_count, eval_count, max_depth, op_ratios, max_length):
    # evaluation dataset:
    # there is no ans or expr
    eval_samples = generate_equality_mlm_samples(eval_count, max_depth, op_ratios, max_length)
    eval_path = os.path.join(DATA_BASE, "eval_eq.jsonl")

    with open(eval_path, 'w') as f:
        for sample in eval_samples:
            f.write(json.dumps(sample) + "\n")

    print_sample(eval_samples[0])

    # training dataset
    train_samples = generate_equality_mlm_samples(train_count, max_depth, op_ratios, max_length)
    train_path = os.path.join(DATA_BASE, "train_eq.jsonl")
    with open(train_path, 'w') as f:
        for sample in train_samples:
            f.write(json.dumps(sample) + "\n")
    print_sample(train_samples[0])

    return train_samples, eval_samples

# set seed and generate
# random.seed(42)
# generate_datasets(100000, 100, 2, [0.3, 0.3, 0.3, 0.1], 16, 0.5)
# generate_datasets(100000, 100, 2, [1.0, 0.0, 0.0, 0.0], 10, 0.5)

if __name__ == "__main__":
    random.seed(42)
    k = 1
    ratios = [1, 0, 0, 0]
    length = 16
    mask_num_per_value = 1

    # generate def data
    train_sample_values = NUMBER_SET
    random.shuffle(NUMBER_SET)
    print(len(NUMBER_SET))
    def_train, def_eval = generate_def_datasets(train_sample_values, 100, k, ratios, length, mask_num_per_value)
    # generate eq data
    eq_train, eq_eval = generate_equality_datasets(100000, 100, k, ratios, length)

    # create a relatively balanced dataset of def and equality
    # by duplicating the def by |eq| / |def| \div 5, so it gives a 1:5 ratio
    ratio = 5
    def_train = def_train * (len(eq_train) // len(def_train) // ratio)
    # def_eval = def_eval * (len(eq_eval) // len(def_eval) // ratio)
    print(len(def_train), len(eq_train))
    # print(len(def_eval), len(eq_eval))
    train = def_train + eq_train
    # eval_ = def_eval + eq_eval
    random.shuffle(train)
    # random.shuffle(eval_)

    # save
    train_path = os.path.join(DATA_BASE, "train_def_eq.jsonl")
    with open(train_path, 'w') as f:
        for sample in train:
            f.write(json.dumps(sample) + "\n")
    
    # eval_path = os.path.join(DATA_BASE, "eval_def_eq.jsonl")
    # with open(eval_path, 'w') as f:
    #     for sample in eval_:
    #         f.write(json.dumps(sample) + "\n")
