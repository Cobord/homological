use core::ops::{Add, AddAssign, Div, Mul, MulAssign, Neg};

use super::{elementary_matrix::ElementaryMatrixProduct, linear_comb::LazyLinear};
use crate::base_ring::field_generals::Ring;

pub(crate) type BasisIndexing = usize;

pub trait LeftMultipliesBy<T>: Sized {
    /// multiply `self` to become `left_factor * self`
    /// `Self` is the type of column vectors
    /// and `T` is the type of the matrices
    fn left_multiply(&mut self, left_factor: &T);

    /// there may be better ways to multiply by lower/upper triangular matrices
    /// but it defaults to ignoring this information and using `left_multiply`
    fn left_multiply_by_triangular(&mut self, _lower_or_upper: bool, l_or_u_matrix: &T) {
        self.left_multiply(l_or_u_matrix);
    }

    /// there are definitely better ways to multiply by diagonal matrices
    /// this does not get such a default ignoring that it is diagonal
    fn left_multiply_by_diagonal(&mut self, d_matrix: &T);

    /// replace `self` with zero vector
    fn zero_out(&mut self, keep_length: bool);

    /// inject along the map `V -> V \bigoplus F^{how_much}`
    fn zero_pad(&mut self, how_much: BasisIndexing);
}

pub trait AsBasisCombination<N>
where
    N: Add<Output = N> + Neg<Output = N> + Mul<Output = N> + Clone,
{
    /// turn a column vector into explicit linear combination of basis vectors
    fn make_entries(&self) -> LazyLinear<N, BasisIndexing>;
}

#[allow(clippy::module_name_repetitions)]
pub trait MatrixStore<F: Ring + Clone>:
    Add<Output = Self>
    + Mul<Output = Self>
    + Mul<F>
    + MulAssign<F>
    + Sized
    + From<ElementaryMatrixProduct<F>>
{
    type ColumnVector: LeftMultipliesBy<Self>
        + From<(BasisIndexing, Vec<(F, BasisIndexing)>)>
        + AsBasisCombination<F>
        + AddAssign<(F, BasisIndexing)>
        + AddAssign<Self::ColumnVector>
        + MulAssign<F>
        + Clone;
    fn zero_matrix(rows: BasisIndexing, cols: BasisIndexing) -> Self;
    fn identity(dimension: BasisIndexing) -> Self;
    fn num_rows(&self) -> BasisIndexing;
    fn num_cols(&self) -> BasisIndexing;
    fn dimensions(&self) -> (BasisIndexing, BasisIndexing) {
        (self.num_rows(), self.num_cols())
    }
    fn is_zero_matrix(&self) -> bool;

    /// is the product of `self` and `other` the zero matrix
    fn composed_eq_zero(&self, other: &Self) -> bool;

    #[must_use]
    fn transpose(self) -> Self;

    #[must_use]
    fn diagonal_only(&self) -> Self;
}

#[allow(clippy::module_name_repetitions)]
pub trait EffortfulMatrixStore<F: Ring + Clone>:
    Add<Output = Self>
    + Mul<Output = Self>
    + Mul<F>
    + MulAssign<F>
    + Sized
    + From<ElementaryMatrixProduct<F>>
    + MatrixStore<F>
{
    /// rank of this matrix
    fn rank(&self) -> BasisIndexing;

    /// dimension of the kernel
    fn kernel(&self) -> BasisIndexing;

    /// a basis for the kernel
    fn kernel_basis(&self) -> Vec<Self::ColumnVector>;

    #[allow(clippy::result_unit_err)]
    /// the rank of (co)homology with outgoing differential `self` and incoming
    /// differential `previous_d`
    /// also return a basis for the kernel
    /// there will be some linear dependencies of that kernel
    /// when regarded as the corresponding equivalence classes in (co)homology
    /// # Errors
    /// if `d^2 \neq 0`
    fn homology_info(
        &self,
        previous_d: &Self,
        only_dimension: bool,
        check_composition: bool,
    ) -> Result<(BasisIndexing, Vec<Self::ColumnVector>), ()> {
        if check_composition && !self.composed_eq_zero(previous_d) {
            return Err(());
        }
        let my_kernel_size = self.kernel();
        let previous_image_size = previous_d.rank();
        if my_kernel_size < previous_image_size {
            return Err(());
        }
        if only_dimension {
            Ok((my_kernel_size - previous_image_size, vec![]))
        } else {
            Ok((my_kernel_size - previous_image_size, self.kernel_basis()))
        }
    }
}

#[allow(clippy::module_name_repetitions)]
pub trait FieldMatrixStore<F: Ring + Clone>:
    Add<Output = Self>
    + Mul<Output = Self>
    + Mul<F>
    + MulAssign<F>
    + Sized
    + From<ElementaryMatrixProduct<F>>
    + MatrixStore<F>
where
    F: Div<Output = F>,
{
    /// this matrix can be written as a product of 3 factors
    /// L, D and U
    fn ldu_decompose(self) -> (Self, Self, Self);

    /// invert the diagonal entries
    /// # Errors
    /// one of the diagonal entries was 0 so could not invert it
    #[allow(clippy::result_unit_err)]
    fn diagonal_invert(&mut self) -> Result<(), ()>;

    /// solve a linear system with (weighted) Jacobi iteration
    fn jacobi_iterate(
        self,
        mut initial_x: Self::ColumnVector,
        mut b_vec: Self::ColumnVector,
        weighting: F,
        stopping_criterion: (
            u8,
            Option<impl Fn(&Self::ColumnVector, &Self::ColumnVector) -> bool>,
        ),
    ) -> Self::ColumnVector {
        let (num_iterations, close_to_answer) = stopping_criterion;
        if num_iterations == 0 {
            return initial_x;
        }
        let num_cols = self.num_cols();
        let mut inverse_diagonal = self.diagonal_only();
        inverse_diagonal
            .diagonal_invert()
            .expect("diagonal entries should all be invertible");
        b_vec.left_multiply_by_diagonal(&inverse_diagonal);
        b_vec *= weighting.clone();
        let mut iteration_matrix = Self::identity(num_cols);
        let mut w_d_inverse_a = inverse_diagonal * self;
        w_d_inverse_a *= -weighting;
        iteration_matrix = iteration_matrix + w_d_inverse_a;
        let mut previous_iteration_x = initial_x.clone();
        for _iteration_num in 0..num_iterations {
            initial_x.left_multiply(&iteration_matrix);
            initial_x += b_vec.clone();
            if let Some(closeness_function) = &close_to_answer {
                if closeness_function(&previous_iteration_x, &initial_x) {
                    break;
                }
                previous_iteration_x = initial_x.clone();
            }
        }
        initial_x
    }
}
