use core::ops::{Add, AddAssign, Div, Mul, MulAssign, Neg};

use crate::elementary_matrix::ElementaryMatrixProduct;
use crate::field_generals::Ring;
use crate::linear_comb::LazyLinear;

pub(crate) type BasisIndexing = usize;

pub trait LeftMultipliesBy<T>: Sized {
    fn left_multiply(&mut self, left_factor: &T);

    #[allow(dead_code)]
    /// there may be better ways to multiply by lower/upper triangular matrices
    fn left_multiply_by_triangular(&mut self, _lower_or_upper: bool, l_or_u_matrix: &T) {
        self.left_multiply(l_or_u_matrix);
    }

    #[allow(dead_code)]
    fn left_multiply_by_diagonal(&mut self, d_matrix: &T);

    fn zero_out(&mut self, keep_length: bool);
    fn zero_pad(&mut self, how_much: BasisIndexing);
}

pub trait AsBasisCombination<N>
where
    N: Add<Output = N> + Neg<Output = N> + Mul<Output = N> + Clone,
{
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
    fn composed_eq_zero(&self, other: &Self) -> bool;
    fn transpose(self) -> Self;

    #[allow(dead_code)]
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
    fn rank(&self) -> BasisIndexing;
    fn kernel(&self) -> BasisIndexing;
    fn kernel_basis(&self) -> Vec<Self::ColumnVector>;
    fn homology_info(
        &self,
        previous_d: &Self,
        only_dimension: bool,
        check_composition: bool,
    ) -> Result<(BasisIndexing, Vec<Self::ColumnVector>), ()> {
        // the rank of homology with outgoing differential self and incoming
        // differential previous_d
        // also a basis for the kernel which will have some
        // linear dependencies when regarded as the equivalence classes in cohomology
        // but we haven't chosen a basis for the quotient, only the kernel
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
#[allow(dead_code)]
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
    fn ldu_decompose(self) -> (Self, Self, Self);

    fn diagonal_invert(&mut self) -> Result<(), ()>;

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
