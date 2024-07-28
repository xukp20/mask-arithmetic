from expr import Expr
from tactic import ALL_TACTICS, class2name

class ProofStep:
    def __init__(self, tactic: str, position: int, expr: Expr):
        if isinstance(tactic, str):
            tactic = ALL_TACTICS[tactic]
        self.tactic = tactic
        self.position = position
        self.expr = expr
    
    def to_dict(self):
        return {
            'tactic': str(self.tactic),
            'position': self.position,
            'expr': self.expr.to_dict()
        }

    def from_dict(d):
        tactic = d['tactic']
        position = d['position']
        expr = Expr.from_dict(d['expr'])
        return ProofStep(tactic, position, expr)

    def __str__(self):
        return f"{class2name(self.tactic)} at {self.position}: {self.expr}"

class ProofStepGroup:
    """
        A list of proof steps that applied to the same position sequentially.
    """
    def __init__(self, tactic: str, position: int, steps: list):
        pass

    

class Proof:
    def __init__(self, source, target, steps):
        self.source = source
        self.target = target
        self.steps = steps

    def to_dict(self):
        return {
            'source': self.source.to_dict(),
            'target': self.target.to_dict(),
            'steps': [step.to_dict() for step in self.steps]
        }

    def from_dict(d):
        source = Expr.from_dict(d['source'])
        target = Expr.from_dict(d['target'])
        steps = [ProofStep.from_dict(step) for step in d['steps']]
        return Proof(source, target, steps)
    
    def __str__(self):
        title = f"{self.source} -> {self.target} ({len(self.steps)} steps)\n"
        steps = '\n'.join([f"{i+1}. {class2name(step.tactic)} at {step.position}: {step.expr}" for i, step in enumerate(self.steps)])
        return title + steps

    def __len__(self):
        return len(self.steps)

    def check_valid(self, lower=0, upper=10):
        # check if the proof is valid
        current_expr = self.source
        for i, step in enumerate(self.steps):
            if step.position >= len(current_expr):
                return False
            current_expr = step.tactic.apply(current_expr, step.position, lower, upper)
            if current_expr is None:
                print(f"Step {i} is invalid: {step.tactic} at {step.position} gives None")
                return False
            if current_expr != step.expr:
                print(current_expr.to_dict())
                print(step.expr.to_dict())
                print(f"Step {i} is invalid: gives {current_expr} instead of {step.expr} from step")
                return False
        return current_expr == self.target

