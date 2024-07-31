use core::ops::{Add, AddAssign, Mul, MulAssign, Neg};

use crate::elementary_matrix::ElementaryMatrixProduct;
use crate::field_generals::Ring;
use crate::linear_comb::LazyLinear;

pub(crate) type BasisIndexing = usize;

pub trait LeftMultipliesBy<T>: Sized {
    fn left_multiply(&mut self, left_factor: &T);
    fn zero_out(&mut self, keep_length: bool);
    fn zero_pad(&mut self, how_much: BasisIndexing);
}

pub trait ReadEntries<N>
where
    N: Add<Output = N> + Neg<Output = N> + Mul<Output = N> + Clone,
{
    fn make_entries(&self) -> LazyLinear<N, BasisIndexing>;
}

pub trait MatrixStore<F: Ring + Clone>:
    Add<Output = Self> + Mul<Output = Self> + MulAssign<F> + Sized + From<ElementaryMatrixProduct<F>>
{
    type ColumnVector: LeftMultipliesBy<Self>
        + From<(BasisIndexing, Vec<(F, BasisIndexing)>)>
        + ReadEntries<F>
        + AddAssign<(F, BasisIndexing)>;
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
