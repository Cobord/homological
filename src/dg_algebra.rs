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
    + num::Zero
    + Eq
{
}

type BasisMultiplier<F> = fn(
    (HomologicalIndexing, BasisIndexing),
    (HomologicalIndexing, BasisIndexing),
) -> LazyLinear<F, (HomologicalIndexing, BasisIndexing)>;

#[allow(dead_code)]
struct DGAlgebra<R: HomSigns, F: Field, M: MatrixStore<F>, A: AlgebraStore<F> + From<(HomologicalIndexing, M::ColumnVector)>> {
    underlying_chain: Rc<ChainFVect<R, F, M>>,
    elt: A,
    elt_basis: LazyLinear<F, (HomologicalIndexing, BasisIndexing)>,
    two_summand_mul: BasisMultiplier<F>,
    back_converter: fn(HomologicalIndexing, &M::ColumnVector) -> LazyLinear<F, (HomologicalIndexing, BasisIndexing)>,
}

impl<R, F, M, A> DGAlgebra<R, F, M, A>
where
    R: HomSigns,
    F: Field + 'static,
    M: MatrixStore<F>,
    A: AlgebraStore<F> + From<(HomologicalIndexing, M::ColumnVector)>,
{
    #[allow(dead_code)]
    fn apply_differential(&mut self) {
        if self.elt == A::zero() {
            return;
        }
        let mut my_summands : LazyLinear::<F, (HomologicalIndexing, BasisIndexing)> = (F::one(),(0,0)).into();
        core::mem::swap(&mut my_summands, &mut self.elt_basis);
        let mut all_vecs = vec![].into();
        let mut min_index = 0;
        let mut max_index = 0;
        for _term in my_summands.summands {
            todo!();
        }
        let starting_index = if R::differential_increases() {min_index} else {max_index};
        self.underlying_chain.apply_all_differentials(starting_index,&mut all_vecs);
        let direction = if R::differential_increases() {1} else {-1};
        let mut hom_index = starting_index;
        let mut new_elt = A::zero();
        let mut new_elt_basis = LazyLinear::<_,_>{ summands: Box::new(vec![].into_iter()) };
        for term in all_vecs.into_iter() {
            new_elt_basis = new_elt_basis + (self.back_converter)(hom_index,&term);
            new_elt = new_elt + (hom_index,term).into();
            hom_index += direction;
        }
        self.elt = new_elt;
        self.elt_basis = new_elt_basis;
    }
}

impl<R, F, M, A> MulAssign<F> for DGAlgebra<R, F, M, A>
where
    R: HomSigns,
    F: Field + Clone + 'static,
    M: MatrixStore<F>,
    A: AlgebraStore<F> + From<(HomologicalIndexing, M::ColumnVector)>,
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
    A: AlgebraStore<F>+ From<(HomologicalIndexing, M::ColumnVector)>,
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
            back_converter: todo!()
        }
    }
}

impl<R, F, M, A> Sub for DGAlgebra<R, F, M, A>
where
    R: HomSigns,
    F: Field + 'static,
    M: MatrixStore<F>,
    A: AlgebraStore<F> + From<(HomologicalIndexing, M::ColumnVector)>,
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
            back_converter: todo!()
        }
    }
}

impl<R, F, M, A> Mul for DGAlgebra<R, F, M, A>
where
    R: HomSigns,
    F: Field,
    M: MatrixStore<F>,
    A: AlgebraStore<F> + From<(HomologicalIndexing, M::ColumnVector)>,
{
    type Output = Self;
    fn mul(self, _rhs: Self) -> Self::Output {
        todo!()
    }
}
