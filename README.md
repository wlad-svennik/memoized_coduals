#   The dual numbers *can* do efficient autodiff!

The *codual numbers* are a simple method of doing *automatic differentiation* in *reverse mode*. They contrast with the *dual numbers* which provide an easy way of doing automatic differentiation in *forward mode*. The difference between the two modes is that sometimes one is faster than the other.

The folklore appears to be that **forward mode** autodiff is easy to implement because it can be done using the beautiful algebra of dual numbers, while the same is assumed to not be the case for **reverse mode**. This repository presents a counterargument that a variant of the dual numbers -- called the codual numbers -- can be used to represent an implementation of **reverse mode** autodiff that is just as elegant and terse as can be done for forward mode. This idea was first suggested by Sandro Magi (pseudonym: Naasking).

This implementation of the codual numbers **differs** from Sandro Magi's by using simple *memoisation* to eliminate the exponential worst-case behaviour he encountered. In Magi's original implementation, this idea seems obscured, largely because the code was more effectful and therefore the opportunity for memoisation was less apparent. The memoisation is achieved using only one additional line of code!

This implementation should be simpler and more transparent than Magi's, I hope. It also suggests that Magi's reasoning behind the term "codual numbers" is perhaps misleading.

## Definition of *dual number* and *codual number*

The codual numbers are the set
<p align="center"><img src="https://rawgit.com/ogogmad/memoized_coduals/main/svgs/463a148ecf9178c6a31f2d9434393c5c.svg?invert_in_darkmode" align=middle width=48.401773199999994pt height=14.52054615pt/></p>
while the codual numbers are a subset of
<p align="center"><img src="https://rawgit.com/ogogmad/memoized_coduals/main/svgs/f9fef1cbe592d95cf2ebb9bc6e02e809.svg?invert_in_darkmode" align=middle width=52.1459763pt height=16.084077899999997pt/></p>
where the second component is always a *linear map*.

A notation that's used to write a dual number is <img src="https://rawgit.com/ogogmad/memoized_coduals/main/svgs/2f45476a75f3c18d51e537e01fb59725.svg?invert_in_darkmode" align=middle width=43.50064454999998pt height=22.831056599999986pt/>, which stands for <img src="https://rawgit.com/ogogmad/memoized_coduals/main/svgs/0cd27d4708cd735f6ea469dc3debed0e.svg?invert_in_darkmode" align=middle width=35.83526759999999pt height=24.65753399999998pt/>. Formally, <img src="https://rawgit.com/ogogmad/memoized_coduals/main/svgs/ad6370c8c8de22b67ebb85cbc747ef57.svg?invert_in_darkmode" align=middle width=45.17680365pt height=26.76175259999998pt/> while <img src="https://rawgit.com/ogogmad/memoized_coduals/main/svgs/47b926f041a6644d642b809cab8b1d23.svg?invert_in_darkmode" align=middle width=37.80234314999999pt height=22.831056599999986pt/>.

The codual numbers may be represented using exactly the same notation as the dual numbers. They are no different than the dual numbers, except in how they're represented on a computer! Using lambda calculus notation (which I assume you are familiar with) any dual number <img src="https://rawgit.com/ogogmad/memoized_coduals/main/svgs/0cd27d4708cd735f6ea469dc3debed0e.svg?invert_in_darkmode" align=middle width=35.83526759999999pt height=24.65753399999998pt/> can be turned into the codual number <img src="https://rawgit.com/ogogmad/memoized_coduals/main/svgs/f5f35c8eedaf7d48a2b01fbc5eea4723.svg?invert_in_darkmode" align=middle width=70.88095739999999pt height=24.65753399999998pt/>, and conversely every codual number <img src="https://rawgit.com/ogogmad/memoized_coduals/main/svgs/74501cb990f9f95856ceb086ac0d2960.svg?invert_in_darkmode" align=middle width=42.07387469999999pt height=24.65753399999998pt/> can be turned into the dual number <img src="https://rawgit.com/ogogmad/memoized_coduals/main/svgs/7cdb2639df010aebdc19861684d32137.svg?invert_in_darkmode" align=middle width=63.078516599999986pt height=24.65753399999998pt/>. The difference is merely one of data structure; we need a *closure* to represent the codual numbers.

The definition of an operation on the codual numbers can be inferred from its definition on the dual numbers. We demonstrate this using multiplication. For dual numbers, we may define multiplication by:
<p align="center"><img src="https://rawgit.com/ogogmad/memoized_coduals/main/svgs/6f66d533ae7ee6b20a69722885167d8e.svg?invert_in_darkmode" align=middle width=224.10743849999997pt height=17.2895712pt/></p>
For the codual numbers, we may use the correspondence <img src="https://rawgit.com/ogogmad/memoized_coduals/main/svgs/ec22c56dab5633645b1b09e4ef2b78ce.svg?invert_in_darkmode" align=middle width=136.89869984999999pt height=24.7161288pt/> to get:
<p align="center"><img src="https://rawgit.com/ogogmad/memoized_coduals/main/svgs/53b6b2a4ccc066ca92b98cdb396e7808.svg?invert_in_darkmode" align=middle width=350.872797pt height=16.438356pt/></p>
where by "<img src="https://rawgit.com/ogogmad/memoized_coduals/main/svgs/211dca2f7e396e7b572b4982e8ab3d19.svg?invert_in_darkmode" align=middle width=4.5662248499999905pt height=14.611911599999981pt/>", we mean multiplication of real numbers. Using the fact that <img src="https://rawgit.com/ogogmad/memoized_coduals/main/svgs/53d147e7f3fe6e47ee05b88b166bd3f6.svg?invert_in_darkmode" align=middle width=12.32879834999999pt height=22.465723500000017pt/> and <img src="https://rawgit.com/ogogmad/memoized_coduals/main/svgs/61e84f854bc6258d4108d08d4c4a0852.svg?invert_in_darkmode" align=middle width=13.29340979999999pt height=22.465723500000017pt/> are linear maps, we can rearrange this to:
<p align="center"><img src="https://rawgit.com/ogogmad/memoized_coduals/main/svgs/63de038cb1e36972c659ac916023b797.svg?invert_in_darkmode" align=middle width=301.50107955pt height=16.438356pt/></p>
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