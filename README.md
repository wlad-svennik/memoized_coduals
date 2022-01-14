# The dual numbers *can* do efficient autodiff!

The *codual numbers* are a simple method of doing *automatic differentiation* in *reverse mode*. They contrast with the *dual numbers* which provide an easy way of doing automatic differentiation in *forward mode*. The difference between the two modes is that sometimes one is faster than the other.

The folklore appears to be that **forward mode** autodiff is easy to implement because it can be done using the beautiful algebra of dual numbers, while the same is assumed to not be the case for **reverse mode**. This repository presents a counterargument that a variant of the dual numbers – called the codual numbers – can be used to represent an implementation of **reverse mode** autodiff that is just as elegant and terse as can be done for forward mode. This idea was first suggested by Sandro Magi (pseudonym: Naasking).

This implementation of the codual numbers **differs** from Sandro Magi’s by using simple *memoisation* to eliminate the exponential worst-case behaviour he encountered. In Magi’s original implementation, this idea seems obscured, largely because the code was more effectful and therefore the opportunity for memoisation was less apparent. The memoisation is achieved using only one additional line of code!

This implementation should be simpler and more transparent than Magi’s, I hope. It also suggests that Magi’s reasoning behind the term “codual numbers” is perhaps misleading.

## Definition of *dual number* and *codual number*

The codual numbers are the set

![\\mathbb R \\times \\mathbb R,](https://latex.codecogs.com/png.latex?%5Cmathbb%20R%20%5Ctimes%20%5Cmathbb%20R%2C "\mathbb R \times \mathbb R,")

while the codual numbers are a subset of

![\\mathbb R \\times \\mathbb R ^ {\\mathbb R}](https://latex.codecogs.com/png.latex?%5Cmathbb%20R%20%5Ctimes%20%5Cmathbb%20R%20%5E%20%7B%5Cmathbb%20R%7D "\mathbb R \times \mathbb R ^ {\mathbb R}")

where the second component is always a *linear map*.

A notation that’s used to write a dual number is ![a + b \\varepsilon](https://latex.codecogs.com/png.latex?a%20%2B%20b%20%5Cvarepsilon "a + b \varepsilon"), which stands for ![(a,b)](https://latex.codecogs.com/png.latex?%28a%2Cb%29 "(a,b)"). Formally, ![\\varepsilon^2 = 0](https://latex.codecogs.com/png.latex?%5Cvarepsilon%5E2%20%3D%200 "\varepsilon^2 = 0") while ![\\varepsilon \\neq 0](https://latex.codecogs.com/png.latex?%5Cvarepsilon%20%5Cneq%200 "\varepsilon \neq 0").

The codual numbers may be represented using exactly the same notation as the dual numbers. They are no different than the dual numbers, except in how they’re represented on a computer! Using lambda calculus notation (which I assume you are familiar with) any dual number ![(a,b)](https://latex.codecogs.com/png.latex?%28a%2Cb%29 "(a,b)") can be turned into the codual number ![(a, \\lambda k. \\,kb)](https://latex.codecogs.com/png.latex?%28a%2C%20%5Clambda%20k.%20%5C%2Ckb%29 "(a, \lambda k. \,kb)"), and conversely every codual number ![(a,B)](https://latex.codecogs.com/png.latex?%28a%2CB%29 "(a,B)") can be turned into the dual number ![(a,B(1))](https://latex.codecogs.com/png.latex?%28a%2CB%281%29%29 "(a,B(1))"). The difference is merely one of data structure; we need a *closure* to represent the codual numbers.

The definition of an operation on the codual numbers can be inferred from its definition on the dual numbers. We demonstrate this using multiplication. For dual numbers, we may define multiplication by:

![(a,a') \\times (b,b') = (ab, ab' + ba').](https://latex.codecogs.com/png.latex?%28a%2Ca%27%29%20%5Ctimes%20%28b%2Cb%27%29%20%3D%20%28ab%2C%20ab%27%20%2B%20ba%27%29. "(a,a') \times (b,b') = (ab, ab' + ba').")

For the codual numbers, we may use the correspondence ![(a,b') \\mapsto (a, \\lambda k. \\,kb)](https://latex.codecogs.com/png.latex?%28a%2Cb%27%29%20%5Cmapsto%20%28a%2C%20%5Clambda%20k.%20%5C%2Ckb%29 "(a,b') \mapsto (a, \lambda k. \,kb)") to get:

![(a,A) \\times (b,B) = (ab, \\lambda k. \\,k\\cdot(a\\cdot B(1) + b\\cdot A(1))),](https://latex.codecogs.com/png.latex?%28a%2CA%29%20%5Ctimes%20%28b%2CB%29%20%3D%20%28ab%2C%20%5Clambda%20k.%20%5C%2Ck%5Ccdot%28a%5Ccdot%20B%281%29%20%2B%20b%5Ccdot%20A%281%29%29%29%2C "(a,A) \times (b,B) = (ab, \lambda k. \,k\cdot(a\cdot B(1) + b\cdot A(1))),")

where by “![\\cdot](https://latex.codecogs.com/png.latex?%5Ccdot "\cdot")”, we mean multiplication of real numbers. Using the fact that ![A](https://latex.codecogs.com/png.latex?A "A") and ![B](https://latex.codecogs.com/png.latex?B "B") are linear maps, we can rearrange this to:

![(a,A) \\times (b,B) = (ab, \\lambda k. \\,B(ak) + A(bk))).](https://latex.codecogs.com/png.latex?%28a%2CA%29%20%5Ctimes%20%28b%2CB%29%20%3D%20%28ab%2C%20%5Clambda%20k.%20%5C%2CB%28ak%29%20%2B%20A%28bk%29%29%29. "(a,A) \times (b,B) = (ab, \lambda k. \,B(ak) + A(bk))).")

This is precisely how we define multiplication of codual numbers in the code.

## Relationship with other autodiff strategies

It appears that there are three ways of doing reverse-mode autodiff, which correspond directly to the three “stages” of solving a problem using dynamic programming. See the table below:

<table><colgroup><col style="width: 58%" /><col style="width: 9%" /><col style="width: 32%" /></colgroup><thead><tr class="header"><th>Idea</th><th>Example</th><th>Corresponding autodiff algorithm</th></tr></thead><tbody><tr class="odd"><td>Unmemoised recursion</td><td>Exhibit A</td><td>Unmemoised coduals</td></tr><tr class="even"><td>Memoised recursion, or <br/> top-down dynamic programming</td><td>Exhibit B</td><td>Memoised coduals</td></tr><tr class="odd"><td>Bottom-up dynamic programming</td><td>Exhibit C</td><td>Tape-based autodiff</td></tr></tbody></table>

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