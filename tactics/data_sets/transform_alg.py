from expr import Expr
from tactic import *
from proof import *


def eliminate_opposite_terms(expr: Expr, lower: int, upper: int):
    steps = []

    # remove all the zeros first
    while 0 in expr.nums[1:]:
        for i in range(1, len(expr.nums)):
            if expr.nums[i] == 0:
                new_expr = ALL_TACTICS["Merge"].apply(expr, i - 1, lower, upper)
                if new_expr:
                    steps.append(ProofStep("Merge", i - 1, new_expr))
                    expr = new_expr
                    break

    # handle the first one
    if len(expr.nums) > 1 and expr.nums[0] == 0:
        # exchange the first two terms
        new_expr = ALL_TACTICS["Exchange"].apply(expr, 0, lower, upper)
        if new_expr:
            steps.append(ProofStep("Exchange", 0, new_expr))
            expr = new_expr
        # merge
        new_expr = ALL_TACTICS["Merge"].apply(expr, 0, lower, upper)
        if new_expr:
            steps.append(ProofStep("Merge", 0, new_expr))
            expr = new_expr
    
    value = expr.evaluate()
    target_sign = True if value >= 0 else False

    while True:
        # print(f"Current expr: {expr}")
        # Check if there are any opposite terms left
        if all(sign == target_sign for sign in expr.signs):
            break

        # if only one -0 left, convert to +0, by -0 + 0, 0 - 0, 0
        if len(expr.nums) == 1 and expr.nums[0] == 0 and expr.signs[0] == False:
            new_expr = ALL_TACTICS["SplitAdd"].apply(expr, 0, lower, upper)
            steps.append(ProofStep("SplitAdd", 0, new_expr))
            expr = new_expr

            new_expr = ALL_TACTICS["Exchange"].apply(expr, 0, lower, upper)
            steps.append(ProofStep("Exchange", 0, new_expr))
            expr = new_expr

            new_expr = ALL_TACTICS["Merge"].apply(expr, 0, lower, upper)
            steps.append(ProofStep("Merge", 0, new_expr))
            expr = new_expr

        # Find the first term with the target sign
        target_index = next((i for i, sign in enumerate(expr.signs) if sign == target_sign), None)
        # print(f"Target index: {target_index}")

        # Find the first opposite term
        opposite_index = next((i for i, sign in enumerate(expr.signs) if sign != target_sign), None)
        # print(f"Opposite index: {opposite_index}")

        if opposite_index is None:
            break
            
        if opposite_index < target_index:
            # Exchange the terms
            while opposite_index < target_index - 1:
                new_expr = ALL_TACTICS["Exchange"].apply(expr, opposite_index, lower, upper)
                if new_expr:
                    steps.append(ProofStep("Exchange", opposite_index, new_expr))
                    expr = new_expr
                    opposite_index += 1
            # print(f"Applied Exchange at {opposite_index}: {expr}")
        else:
            # Exchange until the opposite term is adjacent to the target term
            while opposite_index > target_index + 1:
                new_expr = ALL_TACTICS["Exchange"].apply(expr, opposite_index - 1, lower, upper)
                if new_expr:
                    steps.append(ProofStep("Exchange", opposite_index - 1, new_expr))
                    expr = new_expr
                    opposite_index -= 1

        # Apply Decrease or Increase until one term becomes 0
        assert abs(target_index - opposite_index) == 1
        left_index = min(target_index, opposite_index)
        # print(f"Left index: {left_index}")
        while expr.nums[left_index] != 0 and expr.nums[left_index + 1] != 0:
            tactic = "Decrease"
            new_expr = ALL_TACTICS[tactic].apply(expr, left_index, lower, upper)
            if new_expr:
                steps.append(ProofStep(tactic, left_index, new_expr))
                expr = new_expr
                # print(f"Applied {tactic} at {left_index}: {expr}")
            else:
                break  # Cannot apply tactic anymore

        # Ensure the positive 0 is on the right side
        if expr.nums[left_index] == 0:
            new_expr = ALL_TACTICS["Exchange"].apply(expr, left_index, lower, upper)
            if new_expr:
                steps.append(ProofStep("Exchange", left_index, new_expr))
                expr = new_expr

        # Merge the 0
        new_expr = ALL_TACTICS["Merge"].apply(expr, left_index, lower, upper)
        if new_expr:
            steps.append(ProofStep("Merge", left_index, new_expr))
            expr = new_expr

        # print(f"Applied Merge at {left_index}: {expr}")
    return steps, expr


def sort_terms(expr: Expr, lower: int, upper: int, start_pos: int = 0):
    steps = []
    n = len(expr.nums)
    
    # Bubble sort implementation using Exchange tactic
    for i in range(n - start_pos):
        for j in range(start_pos, n - i - 1):
            if abs(expr.nums[j]) > abs(expr.nums[j + 1]):
                new_expr = ALL_TACTICS["Exchange"].apply(expr, j, lower, upper)
                if new_expr:
                    steps.append(ProofStep("Exchange", j, new_expr))
                    expr = new_expr

    return steps, expr


def match_term_count(source: Expr, target: Expr, lower: int, upper: int):
    steps = []
    expr = source
    target_term_count = len(target.nums)
    overall_sign = True if expr.evaluate() >= 0 else False

    # Initial sort
    sort_steps, expr = sort_terms(expr, lower, upper)
    steps.extend(sort_steps)

    while len(expr.nums) != target_term_count:
        if len(expr.nums) > target_term_count:
            # Decrease the first term until it's zero or the second is max
            while expr.nums[0] > 0 and expr.nums[1] < upper:
                tactic = "Decrease"
                new_expr = ALL_TACTICS[tactic].apply(expr, 0, lower, upper)
                if new_expr:
                    steps.append(ProofStep(tactic, 0, new_expr))
                    expr = new_expr
                else:
                    break

            if expr.nums[0] == 0:
                # Swap with the second term if it exists
                if len(expr.nums) > 1:
                    new_expr = ALL_TACTICS["Exchange"].apply(expr, 0, lower, upper)
                    if new_expr:
                        steps.append(ProofStep("Exchange", 0, new_expr))
                        expr = new_expr

                # Merge the zero term
                new_expr = ALL_TACTICS["Merge"].apply(expr, 0, lower, upper)
                if new_expr:
                    steps.append(ProofStep("Merge", 0, new_expr))
                    expr = new_expr

            # Sort again
            sort_steps, expr = sort_terms(expr, lower, upper)
            steps.extend(sort_steps)

        else:
            # Add zeros to the front
            tactic = "SplitAdd" if overall_sign else "SplitSub"
            new_expr = ALL_TACTICS[tactic].apply(expr, 0, lower, upper)
            if new_expr:
                steps.append(ProofStep(tactic, 0, new_expr))
                expr = new_expr

            # Swap the zero to the front
            new_expr = ALL_TACTICS["Exchange"].apply(expr, 0, lower, upper)
            if new_expr:
                steps.append(ProofStep("Exchange", 0, new_expr))
                expr = new_expr

    return steps, expr


def match_terms(source: Expr, target: Expr, lower: int, upper: int):
    steps = []
    expr = source
    n = len(expr.nums)

    for i in range(n - 2):  # We'll handle the last two terms separately
        while expr.nums[i] != target.nums[i]:
            if expr.nums[i] < target.nums[i]:
                # if the next is not enough, swap the largest one to the next
                if expr.nums[i + 1] == 0:
                    # sort the rest
                    sort_steps, expr = sort_terms(expr, lower, upper, i + 1)
                    steps.extend(sort_steps)
                    # swap(n-2, n-1), swap... swap(i+1, i+2)
                    for j in range(n - 1, i + 1, -1):
                        new_expr = ALL_TACTICS["Exchange"].apply(expr, j - 1, lower, upper)
                        if new_expr:
                            steps.append(ProofStep("Exchange", j - 1, new_expr))
                            expr = new_expr

                new_expr = ALL_TACTICS["Increase"].apply(expr, i, lower, upper)
            else:
                new_expr = ALL_TACTICS["Decrease"].apply(expr, i, lower, upper)
            
            if new_expr:
                steps.append(ProofStep("Increase" if expr.nums[i] < target.nums[i] else "Decrease", i, new_expr))
                expr = new_expr
            else:
                break  # Cannot apply tactic anymore

        # Sort the remaining terms
        sort_steps, expr = sort_terms(expr, lower, upper, i + 1)
        steps.extend(sort_steps)

    # Handle the last two terms
    for i in range(n - 2, n):
        while expr.nums[i] != target.nums[i]:
            if expr.nums[i] < target.nums[i]:
                new_expr = ALL_TACTICS["Increase"].apply(expr, i, lower, upper)
            else:
                new_expr = ALL_TACTICS["Decrease"].apply(expr, i, lower, upper)
            
            if new_expr:
                steps.append(ProofStep("Increase" if expr.nums[i] < target.nums[i] else "Decrease", i, new_expr))
                expr = new_expr
            else:
                break  # Cannot apply tactic anymore

    return steps, expr


def remove_useless_steps(steps):
    find_useless = True
    while find_useless:
        find_useless = False
        new_steps = []
        for i in range(len(steps) - 1):
            for j in range(i + 1, len(steps)):
                if steps[i].expr == steps[j].expr:
                    find_useless = True
                    new_steps = steps[:i+1] + steps[j+1:]
                    break
            if find_useless:
                break
        if find_useless:
            steps = new_steps
    return steps

def get_path(source: Expr, target: Expr, lower: int, upper: int):
    steps = []
    # Apply the tactics to get the path
    eliminate_source_steps, source = eliminate_opposite_terms(source, lower, upper)
    steps.extend(eliminate_source_steps)

    eliminate_target_steps, new_target = eliminate_opposite_terms(target, lower, upper)
    # for i, step in enumerate(eliminate_target_steps):
    #     print(f"Step {i+1}: {step}")

    reversed_eliminate_target_steps = get_reversed_steps(eliminate_target_steps, target)
    # for i, step in enumerate(reversed_eliminate_target_steps):
    #     print(f"Reversed step {i+1}: {step}")

    target = new_target

    match_term_count_steps, source = match_term_count(source, target, lower, upper)
    steps.extend(match_term_count_steps)
    match_terms_steps, source = match_terms(source, target, lower, upper)
    steps.extend(match_terms_steps)

    steps.extend(reversed_eliminate_target_steps)

    steps = remove_useless_steps(steps)

    return steps

if __name__ == "__main__":
    # -4 - 2 + 3 - 1 + 2 + 10
    # expr = Expr([True, False, False, True, False, False, True, True], [0, 4, 2, 3, 1, 0, 2, 10])
    # steps, new_expr = eliminate_opposite_terms(expr, 0, 10)
    # for step in steps:
    #     print(step)
    # print(new_expr)

    # test the sorting
    # 4, 0, 0, 2, 10, 5
    # expr = Expr([True, True, True, True, True, True], [3, 4, 1, 2, 10, 5])
    # steps, new_expr = sort_terms(expr, 0, 10, start_pos=4)
    # for step in steps:
    #     print(step)
    # print(new_expr)

    # test the match term count
    # 1, 1, 3, 3, 7
    # 4, 1, 10
    # expr = Expr([True, True, True, True, True], [1, 1, 3, 3, 7])
    # target = Expr([True, True, True], [4, 1, 10])
    # steps, new_expr = match_term_count(expr, target, 0, 10)
    # for step in steps:
    #     print(step)
    # print(new_expr)

    # test the match terms
    # 1, 1, 3, 3, 7
    # 2, 2, 3, 4, 4
    # expr = Expr([True, True, True, True, True], [1, 1, 3, 3, 7])
    # target = Expr([True, True, True, True, True], [2, 2, 3, 4, 4])

    # steps, new_expr = match_terms(expr, target, 0, 10)
    # for step in steps:
    #     print(step)
    # print(new_expr)

    # test all
    # -4 - 2 + 3 - 1 + 2 + 10
    # 1 - 1 + 6 - 2 + 4
    # expr = Expr([False, False, True, False, True, True], [4, 2, 3, 1, 2, 10])
    # target = Expr([True, False, True, False, True], [1, 1, 6, 2, 4])

    # - 3 + 6 + 0 -> 8 + 1 + 1 - 7
    expr = Expr([False, True, True], [3, 6, 0])
    target = Expr([True, True, True, False], [8, 1, 1, 7])

    steps = get_path(expr, target, 0, 10)
    print("Tar:", target)
    print("Source:", expr)
    for i, step in enumerate(steps):
        print(f"{i+1}. {step}")
    proof = Proof(expr, target, steps)
    correct = proof.check_valid()
    print(correct)
    
