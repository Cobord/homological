use core::ops::{Add, Mul, MulAssign};
use std::collections::VecDeque;

use crate::elementary_matrix::{ElementaryMatrix, ElementaryMatrixProduct};
use crate::field_generals::Ring;
use crate::matrix_store::{BasisIndexing, EffortfulMatrixStore, LeftMultipliesBy, MatrixStore};

pub struct FactorizedMatrix<F: Ring + Clone + 'static, M: MatrixStore<F>> {
    pub(crate) left_invertible: ElementaryMatrixProduct<F>,
    pub(crate) middle: M,
    pub(crate) right_invertible: ElementaryMatrixProduct<F>,
}

impl<F: Ring + Clone + 'static, M: MatrixStore<F>> FactorizedMatrix<F, M> {
    fn only_middle(middle: M) -> Self {
        let (rows, cols) = middle.dimensions();
        Self {
            left_invertible: ElementaryMatrixProduct::<F>::new(rows),
            middle,
            right_invertible: ElementaryMatrixProduct::<F>::new(cols),
        }
    }
}

pub trait RowReductionHelpers<F: Ring + Clone> {
    fn swap_rows(&mut self, row_idx: BasisIndexing, row_jdx: BasisIndexing);

    fn add_assign_rows(&mut self, row_idx: BasisIndexing, row_jdx: BasisIndexing);

    fn add_assign_factor_rows(&mut self, row_idx: BasisIndexing, factor: F, row_jdx: BasisIndexing);

    fn scale_row(&mut self, row_idx: BasisIndexing, factor: F);

    fn read_entry(&self, row_idx: BasisIndexing, col_idx: BasisIndexing) -> F;

    fn set_entry(&mut self, row_idx: BasisIndexing, col_idx: BasisIndexing, new_value: F);

    /// override if there is a better way to get all the entries below a certain row with column number fixed
    fn find_highest_absolute_value(
        &self,
        current_pivot_row: BasisIndexing,
        num_rows: BasisIndexing,
        current_pivot_column: BasisIndexing,
    ) -> BasisIndexing
    where
        F: PartialOrd,
    {
        let mut row_for_this_pivot_column = current_pivot_row;
        let mut best_abs_seen = self.read_entry(row_for_this_pivot_column, current_pivot_column);
        let neg_best_abs_seen = -best_abs_seen.clone();
        if neg_best_abs_seen > best_abs_seen {
            best_abs_seen = -best_abs_seen;
        }
        for trying_i in row_for_this_pivot_column + 1..num_rows {
            let current_entry = self.read_entry(trying_i, current_pivot_column);
            if current_entry > best_abs_seen {
                best_abs_seen = current_entry;
                row_for_this_pivot_column = trying_i;
                continue;
            }
            let neg_current_entry = -current_entry.clone();
            if neg_current_entry > best_abs_seen {
                best_abs_seen = neg_current_entry;
                row_for_this_pivot_column = trying_i;
            }
        }
        row_for_this_pivot_column
    }
}

#[allow(dead_code)]
pub fn row_echelon_form<F, T>(mut m: T, ones_pivot: bool) -> (ElementaryMatrixProduct<F>, T)
where
    F: Ring + Clone + PartialOrd + core::ops::Div<Output = F>,
    T: MatrixStore<F> + RowReductionHelpers<F>,
{
    let mut current_pivot_row = 0;
    let mut current_pivot_column = 0;
    let (num_rows, num_cols) = (m.num_rows(), m.num_cols());
    let mut elementary_row_operations: VecDeque<ElementaryMatrix<F>> = VecDeque::new();
    while current_pivot_row < num_rows && current_pivot_column < num_cols {
        let row_for_this_pivot_column =
            m.find_highest_absolute_value(current_pivot_row, num_rows, current_pivot_column);
        let pivot_entry = m.read_entry(row_for_this_pivot_column, current_pivot_column);
        if pivot_entry == 0.into() {
            current_pivot_column += 1;
            continue;
        }
        if current_pivot_row != row_for_this_pivot_column {
            m.swap_rows(row_for_this_pivot_column, current_pivot_row);
            elementary_row_operations.push_front(ElementaryMatrix::SwapRows(
                row_for_this_pivot_column,
                current_pivot_row,
            ));
        }
        if ones_pivot {
            let inverse_pivot_entry = (Into::<F>::into(1)) / pivot_entry.clone();
            m.scale_row(current_pivot_row, inverse_pivot_entry.clone());
            m.set_entry(current_pivot_row, current_pivot_column, 1.into());
            elementary_row_operations.push_front(ElementaryMatrix::ScaleRow(
                current_pivot_row,
                inverse_pivot_entry,
            ));
        }
        for row_below in (current_pivot_row + 1)..num_rows {
            let aik = m.read_entry(row_below, current_pivot_column);
            let scale_factor = if ones_pivot {
                -aik
            } else {
                -aik / pivot_entry.clone()
            };
            elementary_row_operations.push_front(ElementaryMatrix::AddAssignMultipleRow(
                current_pivot_row,
                scale_factor.clone(),
                row_below,
            ));
            m.add_assign_factor_rows(current_pivot_row, scale_factor, row_below);
            for zerod_column in 0..current_pivot_column {
                m.set_entry(row_below, zerod_column, 0.into());
            }
        }
        current_pivot_row += 1;
        current_pivot_column += 1;
    }
    (
        ElementaryMatrixProduct {
            dimension: num_rows,
            steps: elementary_row_operations,
        }
        .try_inverse()
        .expect("Elimination operations are invertible"),
        m,
    )
}

pub trait Canonicalizable<F>: MatrixStore<F>
where
    F: Ring + Clone + 'static,
{
    fn canonicalize(self) -> FactorizedMatrix<F, Self>;
}

impl<F, M> FactorizedMatrix<F, M>
where
    F: Ring + Clone,
    M: MatrixStore<F> + Canonicalizable<F>,
{
    #[allow(dead_code)]
    fn recanonicalize(&mut self) {
        let mut new_middle = M::zero_matrix(self.middle.num_rows(), self.middle.num_cols());
        core::mem::swap(&mut new_middle, &mut self.middle);
        let new_middle_canonical = new_middle.canonicalize();
        self.left_invertible *= new_middle_canonical.left_invertible;
        self.middle = new_middle_canonical.middle;
        let mut new_right_invertible = new_middle_canonical.right_invertible;
        core::mem::swap(&mut new_right_invertible, &mut self.right_invertible);
        self.right_invertible *= new_right_invertible;
    }
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

    fn left_multiply_by_diagonal(&mut self, d_matrix: &FactorizedMatrix<F, M>) {
        if d_matrix.left_invertible.is_empty() && d_matrix.right_invertible.is_empty() {
            self.left_multiply_by_diagonal(&d_matrix.middle);
        } else {
            let right_invertible = Into::<M>::into(d_matrix.right_invertible.clone());
            self.left_multiply(&right_invertible);
            self.left_multiply(&d_matrix.middle);
            let left_invertible = Into::<M>::into(d_matrix.left_invertible.clone());
            self.left_multiply(&left_invertible);
        }
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

impl<F, M> Mul<F> for FactorizedMatrix<F, M>
where
    F: Ring + Clone,
    M: MatrixStore<F>,
{
    type Output = Self;

    fn mul(mut self, rhs: F) -> Self::Output {
        self *= rhs;
        self
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
        if self.is_zero_matrix() || rhs.is_zero_matrix() {
            return true;
        }
        let right_self = Into::<M>::into(self.right_invertible.clone());
        let left_other = Into::<M>::into(rhs.left_invertible.clone());
        let done_other = right_self * left_other * rhs.middle.clone();
        self.middle.composed_eq_zero(&done_other)
    }

    fn transpose(self) -> Self {
        Self {
            left_invertible: self.right_invertible.transpose(),
            middle: self.middle.transpose(),
            right_invertible: self.left_invertible.transpose(),
        }
    }

    fn diagonal_only(&self) -> Self {
        let right_self = Into::<M>::into(self.right_invertible.clone());
        let left_self = Into::<M>::into(self.left_invertible.clone());
        let done_other = left_self * self.middle.clone() * right_self;
        let to_return = done_other.diagonal_only();
        Self::only_middle(to_return)
    }
}

impl<F: Ring + Clone, M: EffortfulMatrixStore<F> + Clone> EffortfulMatrixStore<F>
    for FactorizedMatrix<F, M>
{
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

mod test {
    use crate::field_generals::Ring;

    impl Ring for f64 {
        fn try_inverse(self) -> Option<Self> {
            Some(1. / self)
        }
    }
    impl Ring for f32 {
        fn try_inverse(self) -> Option<Self> {
            Some(1. / self)
        }
    }

    #[allow(dead_code)]
    const DO_PRINT: bool = false;

    #[test]
    fn row_echelon_example_0() {
        use super::row_echelon_form;
        use crate::array_store::SquareMatrixStore;
        use crate::matrix_store::MatrixStore;
        let my_matrix = SquareMatrixStore::<3, f64> {
            each_entry: [[1., 0., 0.], [0., -3., 0.], [0., 0., 2.]],
        };
        let expected = my_matrix.clone();
        let (z, w) = row_echelon_form(my_matrix, true);
        let z_transformed: SquareMatrixStore<3, f64> = z.into();
        if DO_PRINT {
            println!("Initial Matrix: {:?}", expected.each_entry);
            println!("Row reduced version: {:?}", w.each_entry);
            println!(
                "The elimination operations that are a prefactor: {:?}",
                z_transformed.each_entry
            );
        }
        let recombined = z_transformed * w;
        if DO_PRINT {
            println!("Recombined : {:?}", recombined);
        }
        assert_eq!(recombined, expected);

        let my_matrix = SquareMatrixStore::<3, f64> {
            each_entry: [[1., 0., 0.], [0., -3., 0.], [0., 0., 2.]],
        };
        let expected = my_matrix.clone();
        let expected_prefactor = SquareMatrixStore::<3, f64>::identity(3);
        let (z, w) = row_echelon_form(my_matrix, false);
        if DO_PRINT {
            println!("Initial Matrix: {:?}", expected.each_entry);
            println!("Row reduced version: {:?}", w.each_entry);
        }
        assert_eq!(w, expected);
        let z_transformed: SquareMatrixStore<3, f64> = z.into();
        assert_eq!(z_transformed, expected_prefactor);
        if DO_PRINT {
            println!(
                "The elimination operations that are a prefactor: {:?}",
                z_transformed.each_entry
            );
        }
        let recombined = z_transformed * w;
        if DO_PRINT {
            println!("Recombined : {:?}", recombined);
        }
        assert_eq!(recombined, expected);

        let my_matrix = SquareMatrixStore::<2, f64> {
            each_entry: [[0., 1.], [1., 1.]],
        };
        let expected = my_matrix.clone();
        let (z, w) = row_echelon_form(my_matrix, false);
        if DO_PRINT {
            println!("Initial Matrix: {:?}", expected.each_entry);
            println!("Row reduced version: {:?}", w.each_entry);
        }
        let z_transformed: SquareMatrixStore<2, f64> = z.into();
        if DO_PRINT {
            println!(
                "The elimination operations that are a prefactor: {:?}",
                z_transformed.each_entry
            );
        }
        let expected_prefactor = SquareMatrixStore::<2, f64> {
            each_entry: [[0., 1.], [1., 0.]],
        };
        assert_eq!(z_transformed, expected_prefactor);
        let recombined = z_transformed * w;
        if DO_PRINT {
            println!("Recombined : {:?}", recombined);
        }
        assert_eq!(recombined, expected);

        let my_matrix = SquareMatrixStore::<2, f64> {
            each_entry: [[0., 2.], [1., 1.]],
        };
        let expected = my_matrix.clone();
        let (z, w) = row_echelon_form(my_matrix, true);
        if DO_PRINT {
            println!("Initial Matrix: {:?}", expected.each_entry);
            println!("Row reduced version: {:?}", w.each_entry);
        }
        let z_transformed: SquareMatrixStore<2, f64> = z.into();
        if DO_PRINT {
            println!(
                "The elimination operations that are a prefactor: {:?}",
                z_transformed.each_entry
            );
        }
        let recombined = z_transformed * w;
        if DO_PRINT {
            println!("Recombined : {:?}", recombined);
        }
        assert_eq!(recombined, expected);
    }

    #[test]
    fn row_echelon_example_1() {
        use super::row_echelon_form;
        use crate::array_store::SquareMatrixStore;
        let my_matrix = SquareMatrixStore::<3, f64> {
            each_entry: [[1., 0., 0.], [0., -3., 0.], [0., 1., 2.]],
        };
        let expected = my_matrix.clone();
        let (z, w) = row_echelon_form(my_matrix, true);
        if DO_PRINT {
            println!("Initial Matrix: {:?}", expected.each_entry);
            println!("Row reduced version: {:?}", w.each_entry);
        }
        let z_transformed: SquareMatrixStore<3, f64> = z.into();
        if DO_PRINT {
            println!(
                "The elimination operations that are a prefactor: {:?}",
                z_transformed.each_entry
            );
        }
        let recombined = z_transformed * w;
        if DO_PRINT {
            println!("Recombined : {:?}", recombined);
        }
        assert_eq!(recombined, expected);
    }

    #[test]
    fn row_echelon_example_2() {
        use super::row_echelon_form;
        use crate::array_store::SquareMatrixStore;
        let my_matrix = SquareMatrixStore::<3, f32> {
            each_entry: [[1., -4., -5.], [10., -3., -6.], [4., 1., 3.]],
        };
        let expected = my_matrix.clone();
        let (z, w) = row_echelon_form(my_matrix, true);
        if DO_PRINT {
            println!("Initial Matrix: {:?}", expected.each_entry);
            println!("Row reduced version: {:?}", w.each_entry);
        }
        let z_transformed: SquareMatrixStore<3, f32> = z.into();
        if DO_PRINT {
            println!(
                "The elimination operations that are a prefactor: {:?}",
                z_transformed.each_entry
            );
        }
        let recombined = z_transformed * w;
        if DO_PRINT {
            println!("Recombined : {:?}", recombined);
        }
        assert_eq!(recombined, expected);
    }

    #[test]
    fn random_row_echelon() {
        use super::row_echelon_form;
        use crate::array_store::SquareMatrixStore;
        use crate::factorized_matrix::RowReductionHelpers;
        let mut rng = rand::thread_rng();
        const MATRIX_SIZE: usize = 5;

        let my_matrix = SquareMatrixStore::<MATRIX_SIZE, f32> {
            each_entry: core::array::from_fn(|_| {
                core::array::from_fn(|_| {
                    let z: f32 = rand::Rng::gen(&mut rng);
                    z
                })
            }),
        };
        let expected = my_matrix.clone();
        let (z, w) = row_echelon_form(my_matrix, true);
        let z_transformed: SquareMatrixStore<MATRIX_SIZE, f32> = z.into();
        let recombined = z_transformed * w;
        for row_idx in 0..MATRIX_SIZE {
            for col_idx in 0..MATRIX_SIZE {
                let in_recombined = recombined.read_entry(row_idx, col_idx);
                let in_expected = expected.read_entry(row_idx, col_idx);
                assert!(num::abs(in_expected - in_recombined) < 1e-5);
            }
        }
    }
}
