# expr structure
class Expr:
    def __init__(self, signs, nums):
        self.signs = signs
        self.nums = nums
    
    def __str__(self):
        sign_strs = ['+' if sign else '-' for sign in self.signs]
        if sign_strs[0] == '+':
            sign_strs[0] = ''

        return ' '.join([f'{sign_strs[i]} {self.nums[i]}' for i in range(len(self.nums))]).strip()
    
    def __repr__(self):
        return str(self)
    
    def evaluate(self):
        return sum([num if sign else -num for sign, num in zip(self.signs, self.nums)])
    
    def __eq__(self, other):
        if not len(self.signs) == len(other.signs):
            return False
        if not len(self.nums) == len(other.nums):
            return False
        for sign1, sign2 in zip(self.signs, other.signs):
            if sign1 != sign2:
                return False
        for num1, num2 in zip(self.nums, other.nums):
            if num1 != num2:
                return False
        return True
    
    def __len__(self):
        return len(self.signs)

    def to_dict(self):
        return {
            'signs': self.signs,
            'nums': self.nums
        }
    
    def from_dict(d):
        return Expr(d['signs'], d['nums'])

