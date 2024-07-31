use core::ops::{Add, Mul, MulAssign};

use crate::elementary_matrix::{ElementaryMatrix, ElementaryMatrixProduct};
use crate::field_generals::Ring;
use crate::matrix_store::{BasisIndexing, LeftMultipliesBy, MatrixStore};

pub struct FactorizedMatrix<F: Ring + Clone + 'static, M: MatrixStore<F>> {
    left_invertible: ElementaryMatrixProduct<F>,
    middle: M,
    right_invertible: ElementaryMatrixProduct<F>,
}

impl<F: Ring + Clone, M: MatrixStore<F>> Add for FactorizedMatrix<F, M> {
    type Output = Self;

    fn add(mut self, mut rhs: Self) -> Self::Output {
        assert_eq!(self.middle.num_rows(), rhs.middle.num_rows());
        assert_eq!(self.middle.num_cols(), rhs.middle.num_cols());
        let final_num_rows = self.middle.num_rows();
        let final_num_cols = self.middle.num_cols();
        let left_invertible = self.left_invertible.extract_common_prefix(
            &mut rhs.left_invertible,
            ElementaryMatrixProduct::<F>::new(final_num_rows),
        );
        let right_invertible = self.right_invertible.extract_common_suffix(
            &mut rhs.right_invertible,
            ElementaryMatrixProduct::<F>::new(final_num_cols),
        );
        let left_self = Into::<M>::into(self.left_invertible);
        let right_self = Into::<M>::into(self.right_invertible);
        let done_self = left_self * (self.middle * right_self);
        let left_other = Into::<M>::into(rhs.left_invertible);
        let right_other = Into::<M>::into(rhs.right_invertible);
        let done_other = left_other * (rhs.middle * right_other);
        Self {
            left_invertible,
            middle: done_self + done_other,
            right_invertible,
        }
    }
}
impl<F: Ring + Clone, M: MatrixStore<F>> Mul for FactorizedMatrix<F, M> {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        let right_self = Into::<M>::into(self.right_invertible);
        let done_self = self.middle * right_self;
        let left_other = Into::<M>::into(rhs.left_invertible);
        let done_other = left_other * rhs.middle;
        Self {
            left_invertible: self.left_invertible,
            middle: done_self * done_other,
            right_invertible: rhs.right_invertible,
        }
    }
}
impl<F: Ring + Clone, M: MatrixStore<F>> From<ElementaryMatrixProduct<F>>
    for FactorizedMatrix<F, M>
{
    fn from(left_invertible: ElementaryMatrixProduct<F>) -> Self {
        let dimension = left_invertible.dimension;
        Self {
            left_invertible,
            middle: M::identity(dimension),
            right_invertible: ElementaryMatrixProduct::<F>::new(dimension),
        }
    }
}

impl<F: Ring + Clone, M: MatrixStore<F>> LeftMultipliesBy<FactorizedMatrix<F, M>>
    for M::ColumnVector
{
    fn left_multiply(&mut self, left_factor: &FactorizedMatrix<F, M>) {
        let right_invertible = Into::<M>::into(left_factor.right_invertible.clone());
        self.left_multiply(&right_invertible);
        self.left_multiply(&left_factor.middle);
        let left_invertible = Into::<M>::into(left_factor.left_invertible.clone());
        self.left_multiply(&left_invertible);
    }

    fn zero_out(&mut self, keep_length: bool) {
        self.zero_out(keep_length);
    }

    fn zero_pad(&mut self, how_much: BasisIndexing) {
        self.zero_pad(how_much)
    }
}

impl<F, M> MulAssign<F> for FactorizedMatrix<F, M>
where
    F: Ring + Clone,
    M: MatrixStore<F>,
{
    fn mul_assign(&mut self, rhs: F) {
        if rhs.clone().try_inverse().is_some() {
            let num_rows = self.middle.num_rows();
            let num_cols = self.middle.num_cols();
            if num_rows <= num_cols {
                for idx in 0..num_rows {
                    self.left_invertible
                        .steps
                        .push_back(ElementaryMatrix::ScaleRow(idx, rhs.clone()));
                }
            } else {
                for idx in 0..num_cols {
                    self.right_invertible
                        .steps
                        .push_back(ElementaryMatrix::ScaleRow(idx, rhs.clone()));
                }
            }
        } else {
            self.middle *= rhs;
        }
    }
}

impl<F: Ring + Clone, M: MatrixStore<F> + Clone> MatrixStore<F> for FactorizedMatrix<F, M> {
    type ColumnVector = M::ColumnVector;

    fn zero_matrix(rows: BasisIndexing, cols: BasisIndexing) -> Self {
        Self {
            left_invertible: ElementaryMatrixProduct::<F>::new(rows),
            middle: M::zero_matrix(rows, cols),
            right_invertible: ElementaryMatrixProduct::<F>::new(cols),
        }
    }

    fn identity(dimension: BasisIndexing) -> Self {
        Self {
            left_invertible: ElementaryMatrixProduct::<F>::new(dimension),
            middle: M::identity(dimension),
            right_invertible: ElementaryMatrixProduct::<F>::new(dimension),
        }
    }

    fn num_rows(&self) -> BasisIndexing {
        self.middle.num_rows()
    }

    fn num_cols(&self) -> BasisIndexing {
        self.middle.num_cols()
    }

    fn is_zero_matrix(&self) -> bool {
        self.middle.is_zero_matrix()
    }

    fn composed_eq_zero(&self, rhs: &Self) -> bool {
        let right_self = Into::<M>::into(self.right_invertible.clone());
        let left_other = Into::<M>::into(rhs.left_invertible.clone());
        let done_other = right_self * left_other * rhs.middle.clone();
        self.middle.composed_eq_zero(&done_other)
    }

    fn transpose(self) -> Self {
        Self {
            left_invertible: self.left_invertible.transpose(),
            middle: self.middle.transpose(),
            right_invertible: self.right_invertible.transpose(),
        }
    }

    fn rank(&self) -> BasisIndexing {
        self.middle.rank()
    }

    fn kernel(&self) -> BasisIndexing {
        self.middle.kernel()
    }

    fn kernel_basis(&self) -> Vec<Self::ColumnVector> {
        // XMY
        // v in ker M
        // Y^-1 v in ker XMY
        let middle_kernel = self.middle.kernel_basis();
        let y_matrix_inverse = Into::<M>::into(
            self.right_invertible
                .clone()
                .try_inverse()
                .expect("Y should be invertible/explicitly product of invertibles"),
        );
        middle_kernel
            .into_iter()
            .map(|mut z| {
                z.left_multiply(&y_matrix_inverse);
                z
            })
            .collect()
    }
}
