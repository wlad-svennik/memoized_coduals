# Memoised coduals

A small Python experiment in automatic differentiation using a value and a closure for its derivative. It explores the codual-number representation suggested by Sandro Magi (Naasking), with `functools.cache` applied to each derivative closure.

## Run the example

The Python implementation uses only the standard library and requires Python 3.9 or later.

```sh
python memoized_coduals.py
```

The example differentiates an iteration of the Babylonian method for computing the square root of 2. It prints the resulting value and derivative.

To differentiate a simple expression:

```python
from memoized_coduals import Codual

x = Codual(3.0, lambda k: k)
y = x * x + 2 * x

print(y.x)      # 15.0
print(y.dx(1))  # 8.0
```

## Representation

A codual stores a scalar value $u$ and a linear map $d_u$ that returns a weighted derivative. For example, the input variable is represented by `(x, lambda k: k)`, while a constant has the zero derivative map.

Multiplication is implemented as

$$
(u,d_u)(v,d_v) = \bigl(uv,\;k\mapsto d_v(uk)+d_u(vk)\bigr).
$$

The code also includes addition, subtraction, division, comparisons on the stored values, and sine. The `fwd_*` and `rev_*` functions show two ways of arranging the same derivative calculation. The overloaded arithmetic operators use the `rev_*` functions.

For scalar derivatives, a linear map is determined by its value at 1. This relates the closure representation to the usual dual-number pair $(u,d_u(1))$.

## What memoisation does

Caching avoids repeating a call to the same derivative closure with the same argument. It can help when a calculation revisits an identical derivative query.

It does **not** remove exponential behaviour in general. Different paths through a shared expression can pass different weights to the same closure, giving distinct cache entries. For example, a chain of expressions of the form `y = a * y + b * y`, using a fresh pair of distinct primes at each step, has a linear number of expression nodes but can produce exponentially many different incoming weights.

This repository is an experiment in the representation and evaluation of scalar derivatives. It does not establish the complexity guarantees of a conventional reverse-mode implementation with accumulated adjoints.
