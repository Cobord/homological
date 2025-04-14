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

`LazyLinear<N, T>` covers formal linear combinations of T with coefficients in N.

One can add, negate, subtract, multiply by constants convert from T and (N,T).

If T implements a TermMultiplier for how it gets multiplied with a T2 to produce a linear combination of T's and
T2 does for how it gets multiplied with a T2 to produce a linear combination of T2's then one can multiply
LazyLinear<N,T> and LazyLinear<N,T2> where the former is the module for the later which is an algebra.

# Matrix Store

This abstracts how matrices are stored so they only need to have the capabilities of

- addition, multiplication and scalar multiplication
- querying dimensions
- querying if a product of two matrices is 0
- transposing
- querying if diagonal
- multiplying by a column vector (also abstracted in how it is stored with an associated type)
- give a basis for kernel or just it's dimension
- construct from factorized form

# Factorized Matrix

Explicitly given as a product of some elementary matrices then something in the middle followed by some other elementary matrices.
This gives the form in which row and column operations have been made manifest. The way the middle term is stored is abstracted as above (without requiring a method to compute kernel/homology information).

## Elementary Matrix

Standard row/column operations

# F2 Vect

uses bitvec to cram entries together. This is the main concern for why abstracted how matrices were stored. So we could use the same syntax for chain complexes over F2 as well as for rational numbers and get the compressed storage instead of storing a full byte for each entry of F2 matrices.

# Chain Complex

These can be homological or cohomological. All the vector spaces in each (co)homological degree are finite dimensional and their dimensions are given. They also must be bounded in degree. They are given in linked list type format. Shifting and computing specific (co)homology ranks methods are implemented in terms of the abstract matrix storage methods.

# DG Algebra

If you provide a type which is an algebra over the base field and which has a `From` to translate into algebra elements from ith basis vector of the homological degree d piece, then you can do calculations in that dg-algebra. You must also provide how these above basis elements multiply so all the calculations maintain `LazyLinear<F, (HomologicalIndexing, BasisIndexing)>` versions of the underlying element. Those are useful for operations like projecting to specific (co)homological degree.