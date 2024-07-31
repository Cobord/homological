# Ring and Field Generalities

Traits for rings and fields

Don't have to implement anything extra beyond usual arithmetic traits, but may want to.
- Add<Output = Self>
- Sub<Output = Self>
- Neg<Output = Self>
- Mul<Output = Self>
- Eq
- From<usize>
- Sized

- Div<Output = Self>
- Commutative

Commutative being only a marker trait to say that we are relying on the multiplication being commutative.

# Linear Combinations

LazyLinear<N, T> covers formal linear combinations of T with coefficients in N.

One can add, negate, subtract, multiply by constants convert from T and (N,T).

If T implements a TermMultiplier for how it gets multiplied with a T2 to produce a linear combination of T's and
T2 does for how it gets multiplied with a T2 to produce a linear combination of T2's then one can multiply
LazyLinear<N,T> and LazyLinear<N,T2> where the former is the module for the later which is an algebra.

# Matrix Store

# Factorized Matrix

## Elementary Matrix

# F2 Vect

# Chain Complex

# DG Algebra