"""
    Define the tactics applied to the exprs.
"""
from expr import Expr

def class2name(cls):
    return cls.__name__.split('.')[-1].strip().rstrip("'>")

class Tactic:
    def __init__(self):
        pass

    @staticmethod
    def apply(expr, position, lower=0, upper=10):
        """
        Apply the tactic to the expr.

        Args:
            expr (Expr): the expr to apply the tactic
            position (int): the position to apply the tactic
            lower (int): the lower bound of the numbers
            upper (int): the upper bound of the numbers

        Returns:
            Expr: the new expr after applying the tactic
        """
        raise NotImplementedError

    def __str__(self):
        # just give the base class name
        return self.__class__.__name__.split('.')[-1].strip().rstrip("'>")

    def __repr__(self):
        return str(self)


# For a + b
class AddIncrease(Tactic):
    """
    a + b = <a+1> + <b-1>
    or 
    - a - b = - <a+1> - <b-1>
    """

    @staticmethod
    def apply(expr, position, lower=0, upper=10):
        try:
            # take the two 
            a_sign = expr.signs[position]
            b_sign = expr.signs[position + 1]
            if not ((a_sign and b_sign) or (not a_sign and not b_sign)):
                raise ValueError("The tactic can only be applied to a + b or - a - b, where a, b are positive.")
            a = expr.nums[position]
            b = expr.nums[position + 1]
            a_plus = a + 1
            b_minus = b - 1
            if a_plus > upper or b_minus < lower:
                raise ValueError(f"max: {upper}, min: {lower}, a: {a}, b: {b}, a_plus: {a_plus}, b_minus: {b_minus}")
            new_expr = Expr(expr.signs[:position] + [a_sign, b_sign] + expr.signs[position + 2:], expr.nums[:position] + [a_plus, b_minus] + expr.nums[position + 2:])
        except:
            return None
        return new_expr


class AddDecrease(Tactic):
    """
    a + b = <a-1> + <b+1>
    or
    - a - b = - <a-1> - <b+1>
    """
    @staticmethod
    def apply(expr, position, lower=0, upper=10):
        try:
            # take the two 
            a_sign = expr.signs[position]
            b_sign = expr.signs[position + 1]
            if not ((a_sign and b_sign) or (not a_sign and not b_sign)):
                raise ValueError("The tactic can only be applied to a + b or - a - b, where a, b are positive.")
            a = expr.nums[position]
            b = expr.nums[position + 1]
            a_minus = a - 1
            b_plus = b + 1
            if a_minus < lower or b_plus > upper:
                raise ValueError(f"max: {upper}, min: {lower}, a: {a}, b: {b}, a_minus: {a_minus}, b_plus: {b_plus}")
            new_expr = Expr(expr.signs[:position] + [a_sign, b_sign] + expr.signs[position + 2:], expr.nums[:position] + [a_minus, b_plus] + expr.nums[position + 2:])
        except:
            return None
        return new_expr


# a - b
class SubIncrease(Tactic):
    """
    a - b = <a+1> - <b+1>
    or 
    - a + b = - <a+1> + <b+1>
    """
    @staticmethod
    def apply(expr, position, lower=0, upper=10):
        try:
            # take the two 
            a_sign = expr.signs[position]
            b_sign = expr.signs[position + 1]
            if a_sign == b_sign:
                raise ValueError("a, b must have different signs.")
            a = expr.nums[position]
            b = expr.nums[position + 1]
            a_plus = a + 1
            b_plus = b + 1
            if a_plus > upper or b_plus > upper:
                raise ValueError(f"max: {upper}, min: {lower}, a: {a}, b: {b}, a_plus: {a_plus}, b_plus: {b_plus}")
            new_expr = Expr(expr.signs[:position] + [a_sign, b_sign] + expr.signs[position + 2:], expr.nums[:position] + [a_plus, b_plus] + expr.nums[position + 2:])
        except:
            return None
        return new_expr


class SubDecrease(Tactic):
    """
    a - b = <a-1> - <b-1>
    or 
    - a + b = - <a-1> + <b-1>
    """
    @staticmethod
    def apply(expr, position, lower=0, upper=10):
        try:
            # take the two 
            a_sign = expr.signs[position]
            b_sign = expr.signs[position + 1]
            if a_sign == b_sign:
                raise ValueError("a, b must have different signs.")
            a = expr.nums[position]
            b = expr.nums[position + 1]
            a_minus = a - 1
            b_minus = b - 1
            if a_minus < lower or b_minus < lower:
                raise ValueError(f"max: {upper}, min: {lower}, a: {a}, b: {b}, a_minus: {a_minus}, b_minus: {b_minus}")
            new_expr = Expr(expr.signs[:position] + [a_sign, b_sign] + expr.signs[position + 2:], expr.nums[:position] + [a_minus, b_minus] + expr.nums[position + 2:])
        except Exception as e:
            return None
        return new_expr


# split and merge
class SplitAdd(Tactic):
    """
    +-a = +-a + 0
    """
    @staticmethod
    def apply(expr, position, lower=0, upper=10):
        try:
            sign_a = expr.signs[position]
            a = expr.nums[position]
            new_expr = Expr(expr.signs[:position] + [sign_a, True] + expr.signs[position + 1:], expr.nums[:position] + [a, 0] + expr.nums[position + 1:])
        except:
            return None
        return new_expr


class MergeAdd(Tactic):
    """
    +-a + 0 = +-a
    """
    @staticmethod
    def apply(expr, position, lower=0, upper=10):
        try:
            sign_a = expr.signs[position]
            a = expr.nums[position]
            sign_zero = expr.signs[position + 1]
            if expr.nums[position + 1] != 0:
                raise ValueError("The tactic can only be applied to +-a + 0.")
            if sign_zero != True:
                raise ValueError("The tactic can only be applied to +-a + 0.")

            new_expr = Expr(expr.signs[:position] + [sign_a] + expr.signs[position + 2:], expr.nums[:position] + [a] + expr.nums[position + 2:])
        except:
            return None
        return new_expr


class SplitSub(Tactic):
    """
    +-a = +-a - 0
    """
    @staticmethod
    def apply(expr, position, lower=0, upper=10):
        try:
            sign_a = expr.signs[position]
            a = expr.nums[position]
            new_expr = Expr(expr.signs[:position] + [sign_a, False] + expr.signs[position + 1:], expr.nums[:position] + [a, 0] + expr.nums[position + 1:])
        except:
            return None
        return new_expr


class MergeSub(Tactic):
    """
    +-a - 0 = +-a
    """
    @staticmethod
    def apply(expr, position, lower=0, upper=10):
        try:
            sign_a = expr.signs[position]
            a = expr.nums[position]
            sign_zero = expr.signs[position + 1]
            if expr.nums[position + 1] != 0:
                raise ValueError("The tactic can only be applied to +-a - 0.")
            if sign_zero != False:
                raise ValueError("The tactic can only be applied to +-a - 0.")

            new_expr = Expr(expr.signs[:position] + [sign_a] + expr.signs[position + 2:], expr.nums[:position] + [a] + expr.nums[position + 2:])
        except:
            return None
        return new_expr


# exchange
# <a> + <b> = <b> + <a>
# - <a> - <b> = - <b> - <a>
# <a> - <b> = <b> - <a>
# - <a> + <b> = - <b> + <a>
class Exchange(Tactic):
    @staticmethod
    def apply(expr, position, lower=0, upper=10):
        try:
            sign_a = expr.signs[position]
            a = expr.nums[position]
            sign_b = expr.signs[position + 1]
            b = expr.nums[position + 1]
            new_expr = Expr(expr.signs[:position] + [sign_b, sign_a] + expr.signs[position + 2:], expr.nums[:position] + [b, a] + expr.nums[position + 2:])
        except:
            return None
        return new_expr


# 0727: for now, treat AddIncrease and SubIncrease as the same
# same for AddDecrease and SubDecrease
class Increase(Tactic):
    @staticmethod
    def apply(expr, position, lower=0, upper=10):
        try:
            # check signs
            sign_a = expr.signs[position]
            sign_b = expr.signs[position + 1]
            if sign_a == sign_b:
                return AddIncrease.apply(expr, position, lower, upper)
            else:
                return SubIncrease.apply(expr, position, lower, upper)
        except:
            return None
        return new_expr


class Decrease(Tactic):
    @staticmethod
    def apply(expr, position, lower=0, upper=10):
        try:
            # check signs
            sign_a = expr.signs[position]
            sign_b = expr.signs[position + 1]
            if sign_a == sign_b:
                new_expr = AddDecrease.apply(expr, position, lower, upper)
            else:
                new_expr = SubDecrease.apply(expr, position, lower, upper)
        except:
            return None
        return new_expr


# MergeAdd and MergeSub
class Merge(Tactic):
    @staticmethod
    def apply(expr, position, lower=0, upper=10):
        try:
            # check sign of the second number
            sign_b = expr.signs[position + 1]
            if sign_b:
                new_expr = MergeAdd.apply(expr, position, lower, upper)
            else:
                new_expr = MergeSub.apply(expr, position, lower, upper)
        except:
            return None
        return new_expr


# only six tactics
ALL_TACTICS={
    "Increase": Increase,
    "Decrease": Decrease,
    "SplitAdd": SplitAdd,
    "SplitSub": SplitSub,
    "Exchange": Exchange,
    "Merge": Merge
}

# tool
def get_reverse_tactic(tactic, position, expr):
    # look for the reverse tactic
    if isinstance(tactic, str):
        tactic = ALL_TACTICS[tactic]
    
    if tactic == Increase:
        return Decrease, position
    elif tactic == Decrease:
        return Increase, position
    elif tactic == SplitAdd:
        return Merge, position
    elif tactic == SplitSub:
        return Merge, position
    elif tactic == Merge:
        # check the sign of the second number
        sign_b = expr.signs[position + 1]
        if sign_b:
            return SplitAdd, position
        else:
            return SplitSub, position
    elif tactic == Exchange:
        return Exchange, position

def get_reversed_steps(steps, source_expr):
    from proof import ProofStep
    # go through the steps again
    reversed_steps = []
    for step in steps:
        tactic = step.tactic
        position = step.position
        new_tactic, new_position = get_reverse_tactic(tactic, position, source_expr)
        reversed_steps.append(ProofStep(new_tactic, new_position, source_expr))
        source_expr = tactic.apply(source_expr, position)
    
    # reverse the steps
    reversed_steps = reversed_steps[::-1]
    return reversed_steps