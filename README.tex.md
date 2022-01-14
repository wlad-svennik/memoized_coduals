#   The dual numbers *can* do efficient autodiff!

The *codual numbers* are a simple method of doing *automatic differentiation* in *reverse mode*. They contrast with the *dual numbers* which provide an easy way of doing automatic differentiation in *forward mode*. The difference between the two modes is that sometimes one is faster than the other.

The folklore appears to be that **forward mode** autodiff is easy to implement because it can be done using the beautiful algebra of dual numbers, while the same is assumed to not be the case for **reverse mode**. This repository presents a counterargument that a variant of the dual numbers -- called the codual numbers -- can be used to represent an implementation of **reverse mode** autodiff that is just as elegant and terse as can be done for forward mode. This idea was first suggested by Sandro Magi (pseudonym: Naasking).

This implementation of the codual numbers **differs** from Sandro Magi's by using simple *memoisation* to eliminate the exponential worst-case behaviour he encountered. In Magi's original implementation, this idea seems obscured, largely because the code was more effectful and therefore the opportunity for memoisation was less apparent. The memoisation is achieved using only one additional line of code!

This implementation should be simpler and more transparent than Magi's, I hope. It also suggests that Magi's reasoning behind the term "codual numbers" is perhaps misleading.

## Definition of *dual number* and *codual number*

The codual numbers are the set
$$\mathbb R \times \mathbb R,$$
while the codual numbers are a subset of
$$\mathbb R \times \mathbb R ^ {\mathbb R}$$
where the second component is always a *linear map*.

A notation that's used to write a dual number is $a + b \varepsilon$, which stands for $(a,b)$. Formally, $\varepsilon^2 = 0$ while $\varepsilon \neq 0$.

The codual numbers may be represented using exactly the same notation as the dual numbers. They are no different than the dual numbers, except in how they're represented on a computer! Using lambda calculus notation (which I assume you are familiar with) any dual number $(a,b)$ can be turned into the codual number $(a, \lambda k. \,kb)$, and conversely every codual number $(a,B)$ can be turned into the dual number $(a,B(1))$. The difference is merely one of data structure; we need a *closure* to represent the codual numbers.

The definition of an operation on the codual numbers can be inferred from its definition on the dual numbers. We demonstrate this using multiplication. For dual numbers, we may define multiplication by:
$$(a,a') \times (b,b') = (ab, ab' + ba').$$
For the codual numbers, we may use the correspondence $(a,b') \mapsto (a, \lambda k. \,kb)$ to get:
$$(a,A) \times (b,B) = (ab, \lambda k. \,k\cdot(a\cdot B(1) + b\cdot A(1))),$$
where by "$\cdot$", we mean multiplication of real numbers. Using the fact that $A$ and $B$ are linear maps, we can rearrange this to:
$$(a,A) \times (b,B) = (ab, \lambda k. \,B(ak) + A(bk))).$$
This is precisely how we define multiplication of codual numbers in the code.

##  Relationship with other autodiff strategies

It appears that there are three ways of doing reverse-mode autodiff, which correspond directly to the three "stages" of solving a problem using dynamic programming. See the table below:


| Idea                                                      | Example   | Corresponding autodiff algorithm |
| --------------------------------------------------------- | --------- | -------------------------------- |
| Unmemoised recursion                                      | Exhibit A | Unmemoised coduals               |
| Memoised recursion, or <br/> top-down dynamic programming | Exhibit B | Memoised coduals                 |
| Bottom-up dynamic programming                             | Exhibit C | Tape-based autodiff              |

This suggests that the tape-based approach can be derived from the coduals.

Exhibit A:
```python
def fib(n):
    if n == 0 or n == 1:
        return n
    else:
        return fib(n-1) + fib(n-2)
```

Exhibit B:

```python
from functools import cache

@cache
def fib(n):
    if n == 0 or n == 1:
        return n
    else:
        return fib(n-1) + fib(n-2)
```

Exhibit C:

```python
def fib(n):
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a
```