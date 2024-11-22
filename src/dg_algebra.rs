use core::ops::{Add, Mul, MulAssign, Neg, Sub};
use std::collections::{HashMap, VecDeque};
use std::rc::Rc;

use crate::chain_complex::{ChainFVect, HomSigns, HomologicalIndexing};
use crate::field_generals::Field;
use crate::linear_comb::LazyLinear;
use crate::matrix_store::{AsBasisCombination, BasisIndexing, EffortfulMatrixStore, MatrixStore};

#[allow(dead_code)]
pub trait AlgebraStore<F: Field>:
    MulAssign<F>
    + Mul<F>
    + Mul<Output = Self>
    + Add<Output = Self>
    + Sub<Output = Self>
    + Neg<Output = Self>
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
pub struct DGAlgebra<
    R: HomSigns,
    F: Field + Clone,
    M: EffortfulMatrixStore<F>,
    A: AlgebraStore<F> + From<(HomologicalIndexing, M::ColumnVector)>,
> {
    underlying_chain: Rc<ChainFVect<R, F, M>>,
    elt: A,
    elt_basis: LazyLinear<F, (HomologicalIndexing, BasisIndexing)>,
    two_summand_mul: BasisMultiplier<F>,
}

impl<R, F, M, A> DGAlgebra<R, F, M, A>
where
    R: HomSigns,
    F: Field + Clone + 'static,
    M: EffortfulMatrixStore<F>,
    A: AlgebraStore<F> + From<(HomologicalIndexing, M::ColumnVector)>,
{
    pub fn new(
        underlying_chain: Rc<ChainFVect<R, F, M>>,
        two_summand_mul: BasisMultiplier<F>,
    ) -> Self {
        Self {
            underlying_chain,
            elt: A::zero(),
            elt_basis: LazyLinear::<_, _>::new(),
            two_summand_mul,
        }
    }

    /// `e_i` of the F^n which is the vector space at a particular homological degree
    #[allow(dead_code)]
    pub fn concentrated_element(
        underlying_chain: Rc<ChainFVect<R, F, M>>,
        two_summand_mul: BasisMultiplier<F>,
        which_degree: HomologicalIndexing,
        in_basis_of_that_space: BasisIndexing,
    ) -> Self {
        let dim_that_space = underlying_chain.dimensions_of_that(which_degree);
        let mut to_return = Self::new(underlying_chain, two_summand_mul);
        let e_i: M::ColumnVector = (
            dim_that_space,
            vec![(Into::<F>::into(1), in_basis_of_that_space)],
        )
            .into();
        to_return.elt = (which_degree, e_i).into();
        to_return.elt_basis = (1.into(), (which_degree, in_basis_of_that_space)).into();
        to_return
    }

    #[allow(dead_code)]
    pub fn apply_differential(&mut self) {
        if self.elt == A::zero() {
            self.elt_basis = LazyLinear::<_, _>::new();
            return;
        }
        let mut my_summands = LazyLinear::<F, (HomologicalIndexing, BasisIndexing)>::new();
        core::mem::swap(&mut my_summands, &mut self.elt_basis);
        let mut all_vecs: VecDeque<<M as MatrixStore<F>>::ColumnVector> = vec![].into();
        let mut min_index = 0;
        let mut max_index = 0;
        let mut all_vecs_index_bound = 0;
        let index_2_dimension: HashMap<HomologicalIndexing, BasisIndexing> = self
            .underlying_chain
            .dimensions_each_index()
            .into_iter()
            .collect();
        for (which_term, (coeff, (term_index, term_which))) in my_summands.summands.enumerate() {
            if which_term == 0 {
                min_index = term_index;
                max_index = term_index;
                all_vecs_index_bound = term_index;
                let my_dimension = index_2_dimension.get(&term_index).map_or(0, |z| *z);
                let to_push = M::ColumnVector::from((my_dimension, vec![(coeff, term_which)]));
                all_vecs.push_front(to_push);
            } else {
                while term_index < all_vecs_index_bound {
                    all_vecs_index_bound -= 1;
                    let this_space_dimension = index_2_dimension
                        .get(&all_vecs_index_bound)
                        .map_or(0, |z| *z);
                    all_vecs.push_front(M::ColumnVector::from((this_space_dimension, vec![])));
                }
                #[allow(clippy::cast_possible_wrap)]
                let mut index_of_last_dimension =
                    all_vecs_index_bound + ((all_vecs.len() - 1) as HomologicalIndexing);
                while term_index > index_of_last_dimension {
                    index_of_last_dimension += 1;
                    let this_space_dimension = index_2_dimension
                        .get(&index_of_last_dimension)
                        .map_or(0, |z| *z);
                    all_vecs.push_back(M::ColumnVector::from((this_space_dimension, vec![])));
                }
                let offset_in_all_vecs = term_index - all_vecs_index_bound;
                #[allow(clippy::cast_sign_loss, clippy::cast_possible_truncation)]
                let offset_in_all_vecs = offset_in_all_vecs as usize;
                all_vecs[offset_in_all_vecs] += (coeff, term_which);
                min_index = core::cmp::min(min_index, term_index);
                max_index = core::cmp::max(max_index, term_index);
            }
        }
        let starting_index = if R::differential_increases() {
            min_index
        } else {
            let all_vecs_len = all_vecs.len();
            let last_lower_end = (if all_vecs_len % 2 == 0 {
                all_vecs_len / 2
            } else {
                (all_vecs_len - 1) / 2
            }) - 1;
            for i in 0..last_lower_end {
                all_vecs.swap(i, all_vecs_len - i - 1);
            }
            max_index
        };
        self.underlying_chain
            .apply_all_differentials(starting_index, &mut all_vecs);
        let direction = if R::differential_increases() { 1 } else { -1 };
        let mut hom_index = starting_index;
        let mut new_elt = A::zero();
        let mut new_elt_basis = LazyLinear::<_, _> {
            summands: Box::new(vec![].into_iter()),
        };
        for term in all_vecs {
            new_elt_basis += term.make_entries().map(move |x| (hom_index, x));
            new_elt = new_elt + (hom_index, term).into();
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
    M: EffortfulMatrixStore<F>,
    A: AlgebraStore<F> + From<(HomologicalIndexing, M::ColumnVector)>,
{
    fn mul_assign(&mut self, rhs: F) {
        self.elt *= rhs.clone();
        self.elt_basis *= rhs;
    }
}

impl<R, F, M, A> Mul<F> for DGAlgebra<R, F, M, A>
where
    R: HomSigns,
    F: Field + Clone + 'static,
    M: EffortfulMatrixStore<F>,
    A: AlgebraStore<F> + From<(HomologicalIndexing, M::ColumnVector)>,
{
    type Output = Self;

    fn mul(self, rhs: F) -> Self::Output {
        let mut new_elt = self.elt;
        new_elt *= rhs.clone();
        let mut new_elt_basis = self.elt_basis;
        new_elt_basis *= rhs;
        Self {
            underlying_chain: self.underlying_chain,
            elt: new_elt,
            elt_basis: new_elt_basis,
            two_summand_mul: self.two_summand_mul,
        }
    }
}

impl<R, F, M, A> Add for DGAlgebra<R, F, M, A>
where
    R: HomSigns,
    F: Field + Clone + 'static,
    M: EffortfulMatrixStore<F>,
    A: AlgebraStore<F> + From<(HomologicalIndexing, M::ColumnVector)>,
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

impl<R, F, M, A> Neg for DGAlgebra<R, F, M, A>
where
    R: HomSigns,
    F: Field + Clone + 'static,
    M: EffortfulMatrixStore<F>,
    A: AlgebraStore<F> + From<(HomologicalIndexing, M::ColumnVector)>,
{
    type Output = Self;

    fn neg(self) -> Self::Output {
        Self {
            underlying_chain: self.underlying_chain,
            elt: -self.elt,
            elt_basis: -self.elt_basis,
            two_summand_mul: self.two_summand_mul,
        }
    }
}

impl<R, F, M, A> Sub for DGAlgebra<R, F, M, A>
where
    R: HomSigns,
    F: Field + Clone + 'static,
    M: EffortfulMatrixStore<F>,
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
        }
    }
}

impl<R, F, M, A> Mul for DGAlgebra<R, F, M, A>
where
    R: HomSigns,
    F: Field + Clone + 'static,
    M: EffortfulMatrixStore<F>,
    A: AlgebraStore<F> + From<(HomologicalIndexing, M::ColumnVector)>,
{
    type Output = Self;
    fn mul(self, rhs: Self) -> Self::Output {
        // self.underlying_chain points to same as rhs.underlying_chain
        let new_basis = {
            let rhs_materialized = rhs.elt_basis.summands.collect::<Vec<_>>();
            LazyLinear {
                summands: Box::new(self.elt_basis.summands.flat_map(move |self_summand| {
                    rhs_materialized
                        .clone()
                        .into_iter()
                        .flat_map(move |mut rhs_summand| {
                            rhs_summand.0.mul_assign_borrow(&self_summand.0);
                            let coeff = rhs_summand.0;
                            let pieces = (self.two_summand_mul)(self_summand.1, rhs_summand.1);
                            pieces.summands.map(move |(mut piece0, piece1)| {
                                piece0.mul_assign_borrow(&coeff);
                                (piece0, piece1)
                            })
                        })
                })),
            }
        };
        Self {
            underlying_chain: self.underlying_chain,
            elt: self.elt * rhs.elt,
            elt_basis: new_basis,
            two_summand_mul: self.two_summand_mul,
        }
    }
}
