use core::ops::{Add, Mul, MulAssign, Sub};
use std::rc::Rc;

use crate::chain_complex::{ChainFVect, HomSigns, HomologicalIndexing};
use crate::field_generals::Field;
use crate::linear_comb::LazyLinear;
use crate::matrix_store::{BasisIndexing, MatrixStore};

#[allow(dead_code)]
trait AlgebraStore<F: Field>:
    MulAssign<F>
    + Mul<Output = Self>
    + Add<Output = Self>
    + Sub<Output = Self>
    + From<(HomologicalIndexing, BasisIndexing)>
{
}

type BasisMultiplier<F> = fn(
    (HomologicalIndexing, BasisIndexing),
    (HomologicalIndexing, BasisIndexing),
) -> LazyLinear<F, (HomologicalIndexing, BasisIndexing)>;

#[allow(dead_code)]
struct DGAlgebra<R: HomSigns, F: Field, M: MatrixStore<F>, A: AlgebraStore<F>> {
    underlying_chain: Rc<ChainFVect<R, F, M>>,
    elt: A,
    elt_basis: LazyLinear<F, (HomologicalIndexing, BasisIndexing)>,
    two_summand_mul: BasisMultiplier<F>,
}

impl<R, F, M, A> DGAlgebra<R, F, M, A>
where
    R: HomSigns,
    F: Field,
    M: MatrixStore<F>,
    A: AlgebraStore<F>,
{
    #[allow(dead_code)]
    fn apply_differential(&mut self) {
        todo!()
    }
}

impl<R, F, M, A> MulAssign<F> for DGAlgebra<R, F, M, A>
where
    R: HomSigns,
    F: Field + Clone + 'static,
    M: MatrixStore<F>,
    A: AlgebraStore<F>,
{
    fn mul_assign(&mut self, rhs: F) {
        self.elt *= rhs.clone();
        self.elt_basis *= rhs;
    }
}

impl<R, F, M, A> Add for DGAlgebra<R, F, M, A>
where
    R: HomSigns,
    F: Field + 'static,
    M: MatrixStore<F>,
    A: AlgebraStore<F>,
{
    type Output = Self;
    fn add(self, rhs: Self) -> Self::Output {
        // self.underlying_chain points to same as rhs.underlying_chain
        assert_eq!(self.two_summand_mul, rhs.two_summand_mul);
        Self {
            underlying_chain: self.underlying_chain,
            elt: self.elt + rhs.elt,
            elt_basis: self.elt_basis + rhs.elt_basis,
            two_summand_mul: self.two_summand_mul,
        }
    }
}

impl<R, F, M, A> Sub for DGAlgebra<R, F, M, A>
where
    R: HomSigns,
    F: Field + 'static,
    M: MatrixStore<F>,
    A: AlgebraStore<F>,
{
    type Output = Self;
    fn sub(self, rhs: Self) -> Self::Output {
        // self.underlying_chain points to same as rhs.underlying_chain
        assert_eq!(self.two_summand_mul, rhs.two_summand_mul);
        Self {
            underlying_chain: self.underlying_chain,
            elt: self.elt - rhs.elt,
            elt_basis: self.elt_basis - rhs.elt_basis,
            two_summand_mul: self.two_summand_mul,
        }
    }
}

impl<R, F, M, A> Mul for DGAlgebra<R, F, M, A>
where
    R: HomSigns,
    F: Field,
    M: MatrixStore<F>,
    A: AlgebraStore<F>,
{
    type Output = Self;
    fn mul(self, _rhs: Self) -> Self::Output {
        todo!()
    }
}
