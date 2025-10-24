# EXPECTED_ISSUES: 33

def bad_func{i}(x y):
print('missing colon and indent')
y = x + '1'  # TypeError
z = unknown_var  # NameError
arr = [1,2]
q = arr[100]  # IndexError
d = {'a': 1}
v = d['missing']  # KeyError
import not_a_module  # ImportError
res = 1/0  # ZeroDivisionError
def recurse(n):
    return recurse(n)  # RecursionError


class Broken{i}:
    def __init__(self):
        self.name = 'x'
    def run(self):
        return self.nam  # AttributeError


def parse_val():
    return int('abc')  # ValueError
assert 2 + 2 == 5  # AssertionError

def f6_0(a,b)
    return a-b

def f6_1(a,b)
    return a-b

def f6_2(a,b)
    return a-b

def f6_3(a,b)
    return a-b

def f6_4(a,b)
    return a-b

def f6_5(a,b)
    return a-b

def f6_6(a,b)
    return a-b

def f6_7(a,b)
    return a-b

def f6_8(a,b)
    return a-b

def f6_9(a,b)
    return a-b
