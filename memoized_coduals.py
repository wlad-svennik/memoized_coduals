from functools import cache
from math import cos, sin
from typing import Callable

class Codual:
    def __init__(self, x: float, dx: Callable[[float],float]):
        self.x = x
        self.dx = cache(dx)

    # Needed in order to do (x, dx) = X where X is a Codual
    def __iter__(self):
        yield self.x
        yield self.dx

    # Syntax sugar for rev_add, which also converts V automatically to a Codual
    def __add__(U, V):
        if type(V) != Codual:
            V = Codual(V, lambda k: 0)
        return rev_add(U, V)

    # This is a Pythonism for when the left operand of "+" is not a Codual number, but we still want Codual addition
    def __radd__(U, V):
        return U + V

    # Syntax sugar for rev_mul, which also converts V automatically to a Codual
    def __mul__(U, V):
        if type(V) != Codual:
            V = Codual(V, lambda k: 0)
        return rev_mul(U, V)

    # This is a Pythonism for when the left operand of "*" is not a Codual number, but we still want Codual multiplication
    def __rmul__(U, V):
        return U * V

    def __truediv__(U, V):
        if type(V) != Codual:
            V = Codual(V, lambda k: 0)
        return rev_div(U, V)

    def __rtruediv__(U, V):
        return Codual(V, lambda k: 0) / U

    def __sub__(U, V):
        if type(V) != Codual:
            V = Codual(V, lambda k: 0)
        return rev_sub(U, V)

    def __rsub__(U, V):
        return Codual(V, lambda k: 0) - U

    def __neg__(U):
        return 0 - U

    def __gt__(U, V):
        (u, _) = U
        if type(V) == Codual:
            (v, _) = V
        else:
            v = V
        return u > v

    def __lt__(U, V):
        (u, _) = U
        if type(V) == Codual:
            (v, _) = V
        else:
            v = V
        return u < v

    def __ge__(U, V):
        (u, _) = U
        if type(V) == Codual:
            (v, _) = V
        else:
            v = V
        return u >= v

    def __le__(U, V):
        (u, _) = U
        if type(V) == Codual:
            (v, _) = V
        else:
            v = V
        return u <= v

    def __abs__(U):
        if U >= 0:
            return U
        else:
            return -U

    # Beware: This triggers derivative evaluation!
    def __str__(U):
        (u, du) = U
        return f"{u} + eps*{du(1)}"

def fwd_mul(U: Codual, V: Codual) -> Codual:
    (u, du) = U
    (v, dv) = V
    return Codual(u * v, lambda k: k * (u * dv(1) + du(1) * v))

def rev_mul(U: Codual, V: Codual) -> Codual:
    (u, du) = U
    (v, dv) = V
    return Codual(u * v, lambda k: dv(u * k) + du(k * v))


def fwd_add(U: Codual, V: Codual) -> Codual:
    (u, du) = U
    (v, dv) = V
    return Codual(u + v, lambda k: k * (dv(1) + du(1)))

def rev_add(U: Codual, V: Codual) -> Codual:
    (u, du) = U
    (v, dv) = V
    return Codual(u + v, lambda k: dv(k) + du(k))


def fwd_sin(X: Codual) -> Codual:
    (x, dx) = X
    return Codual(sin(x), lambda k: k * cos(x) * dx(1))

def rev_sin(X: Codual) -> Codual:
    (x, dx) = X
    return Codual(sin(x), lambda k: dx(cos(x) * k))


def rev_div(U: Codual, V: Codual) -> Codual:
    (u, du) = U
    (v, dv) = V
    return Codual(u / v, lambda k: (du(k * v) - dv(u * k)) / v**2)


def rev_sub(U: Codual, V: Codual) -> Codual:
    (u, du) = U
    (v, dv) = V
    return Codual(u - v, lambda k: du(k) - dv(k))


if __name__ == '__main__':
    print("Running test example:")
    print("Using Babylonian method to compute square root of 2")

    a = Codual(2, lambda k: k)
    X = 1
    while not (1e-5 > X*X - a > -1e-5):
        X = (X + a/X)/2

    (x, dx) = X
    print(f"Its value is: {x}")
    print(f"Its derivative is: {dx(1)}")