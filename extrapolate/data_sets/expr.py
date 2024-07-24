# definition the simplified version of expressions
class Expr:
    def __init__(self):
        pass

    def __str__(self):
        return ""

    def to_dict(self):
        return {"type": "expr"}

    def evaluate(self):
        return 0

    @staticmethod
    def from_dict():
        raise NotImplementedError

    # def random_split(self, op_ratios):
    #     raise NotImplementedError


class Constant(Expr):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return str(self.value)

    def to_dict(self):
        return {"type": "const", "value": self.value}

    def evaluate(self):
        return self.value

    @staticmethod
    def from_dict(d):
        return Constant(d["value"])


class BinaryOp(Expr):
    def __init__(self, op, left, right):
        self.op = op
        self.left = left
        self.right = right

    def __str__(self):
        return f"{self.left} {self.op} {self.right}"

    def to_dict(self):
        return {"type": "binop", "op": self.op, "left": self.left.to_dict(), "right": self.right.to_dict()}

    @staticmethod
    def from_dict(d):
        op = d["op"]
        left = Expr.from_dict(d["left"])
        right = Expr.from_dict(d["right"])
        return BinaryOp(op, left, right)
        
    def evaluate(self, mod=1000000007):
        left = self.left.evaluate()
        right = self.right.evaluate()
        if self.op == "+":
            return (left + right) % mod
        elif self.op == "-":
            if left < right:
                print("Warning: negative value in subtraction")

            return (left - right + mod) % mod
        elif self.op == "*":
            return (left * right) % mod
        elif self.op == "/":
            return (left // right) % mod
        else:
            raise ValueError(f"Unsupported operation {self.op}")
