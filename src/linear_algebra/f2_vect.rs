use core::ops::{Add, AddAssign, Div, Mul, MulAssign, Neg, Sub};

use bitvec::order::Lsb0;
use bitvec::prelude::BitVec;
use num::{One, Zero};

use super::{
    elementary_matrix::{ElementaryMatrix, ElementaryMatrixProduct},
    factorized_matrix::{row_echelon_form, Canonicalizable, FactorizedMatrix, RowReductionHelpers},
    linear_comb::LazyLinear,
    matrix_store::{
        AsBasisCombination, BasisIndexing, EffortfulMatrixStore, LeftMultipliesBy, MatrixStore,
    },
};
use crate::base_ring::field_generals::{Commutative, Field, IntegerType, Ring};

#[derive(PartialEq, Eq, Debug, Clone, PartialOrd)]
#[repr(transparent)]
pub struct F2(bool);

impl Add for F2 {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        #[allow(clippy::suspicious_arithmetic_impl)]
        F2(self.0 ^ rhs.0)
    }
}

impl AddAssign for F2 {
    fn add_assign(&mut self, rhs: Self) {
        #[allow(clippy::suspicious_op_assign_impl)]
        if rhs.0 {
            self.0 ^= true;
        }
    }
}

impl Neg for F2 {
    type Output = Self;

    fn neg(self) -> Self::Output {
        self
    }
}
impl Sub for F2 {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        self.add(rhs)
    }
}
impl Mul for F2 {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        #[allow(clippy::suspicious_arithmetic_impl)]
        F2(self.0 & rhs.0)
    }
}
impl Commutative for F2 {}
impl Div for F2 {
    type Output = Self;

    fn div(self, rhs: Self) -> Self::Output {
        if rhs.0 {
            self
        } else {
            panic!("Division by 0")
        }
    }
}

impl From<IntegerType> for F2 {
    fn from(value: IntegerType) -> Self {
        Self(value % 2 == 0)
    }
}

impl Zero for F2 {
    fn zero() -> Self {
        0.into()
    }

    fn is_zero(&self) -> bool {
        !self.0
    }
}

impl One for F2 {
    fn one() -> Self {
        1.into()
    }
}

impl Ring for F2 {
    fn characteristic(_primes: Box<dyn Iterator<Item = IntegerType>>) -> IntegerType {
        2
    }

    fn try_inverse(self) -> Option<Self> {
        Some(self)
    }

    fn mul_assign_borrow(&mut self, other: &Self) {
        if !other.0 {
            self.0 = false;
        }
    }
}

impl Field for F2 {}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct F2Matrix {
    rows: BasisIndexing,
    cols: BasisIndexing,
    // Using BitVec to represent rows of the matrix
    pub(crate) data: Vec<BitVec>,
}

impl AddAssign for F2Matrix {
    fn add_assign(&mut self, other: Self) {
        assert_eq!(self.rows, other.rows);
        assert_eq!(self.cols, other.cols);
        for i in 0..self.rows {
            self.data[i] ^= &other.data[i];
        }
    }
}

impl Add for F2Matrix {
    type Output = Self;

    fn add(mut self, rhs: Self) -> Self::Output {
        self += rhs;
        self
    }
}

impl MulAssign for F2Matrix {
    fn mul_assign(&mut self, rhs: Self) {
        let mut dummy = F2Matrix::new(self.rows, self.cols, None);
        core::mem::swap(self, &mut dummy);
        *self = dummy * rhs;
    }
}

impl Mul for F2Matrix {
    type Output = Self;

    fn mul(self, other: Self) -> Self::Output {
        assert_eq!(self.cols, other.rows);

        let mut result_data = Vec::with_capacity(self.rows);
        let final_cols = other.cols;

        let other_transpose = other.transpose();
        for i in 0..self.rows {
            let self_row_i = self.data[i].clone();
            let mut new_row = BitVec::repeat(false, final_cols);
            for j in 0..final_cols {
                let mut other_col_j = other_transpose.data[j].clone();
                other_col_j &= self_row_i.clone();
                let ij_entry = other_col_j.count_ones() % 2 == 1;
                new_row.set(j, ij_entry);
            }
            result_data.push(new_row);
        }

        F2Matrix::new(self.rows, final_cols, Some(result_data))
    }
}

impl From<ElementaryMatrixProduct<F2>> for F2Matrix {
    #[allow(clippy::similar_names)]
    fn from(value: ElementaryMatrixProduct<F2>) -> Self {
        let mut result = Self::identity(value.dimension);
        for step in value.steps.into_iter().rev() {
            match step {
                ElementaryMatrix::SwapRows(row_idx, row_jdx) => {
                    result.swap_rows(row_idx, row_jdx);
                }
                ElementaryMatrix::AddAssignRow(row_idx, row_jdx) => {
                    result.add_assign_rows(row_idx, row_jdx);
                }
                ElementaryMatrix::ScaleRow(row_idx, factor) => {
                    result.scale_row(row_idx, factor);
                }
                ElementaryMatrix::AddAssignMultipleRow(row_idx, factor, row_jdx) => {
                    result.add_assign_factor_rows(row_idx, factor, row_jdx);
                }
            }
        }
        result
    }
}

impl LeftMultipliesBy<F2Matrix> for F2ColumnVec {
    fn left_multiply(&mut self, left_factor: &F2Matrix) {
        let left_factor_rows = left_factor.num_rows();
        let left_factor_cols = left_factor.num_cols();
        assert!(left_factor_cols == self.0 .0);
        let left_transpose = left_factor.clone().transpose();
        let mut new_result =
            F2ColumnVec((left_factor_rows, BitVec::repeat(false, left_factor_rows)));
        for (idx, bit) in self.0 .1.as_bitslice().iter().enumerate() {
            if *bit {
                new_result += F2ColumnVec((left_factor_rows, left_transpose.data[idx].clone()));
            }
        }
        *self = new_result;
    }

    fn zero_out(&mut self, keep_length: bool) {
        let new_len = if keep_length { self.0 .0 } else { 0 };
        let new_vec = BitVec::repeat(false, new_len);
        *self = F2ColumnVec((new_len, new_vec));
    }

    fn zero_pad(&mut self, how_much: BasisIndexing) {
        self.0 .0 += how_much;
        self.0
             .1
            .extend_from_bitslice(&BitVec::<usize, Lsb0>::repeat(false, how_much));
    }

    fn left_multiply_by_triangular(&mut self, _lower_or_upper: bool, l_or_u_matrix: &F2Matrix) {
        // TODO actually use triangularity
        self.left_multiply(l_or_u_matrix);
    }

    fn left_multiply_by_diagonal(&mut self, d_matrix: &F2Matrix) {
        let (num_rows, num_cols) = d_matrix.dimensions();
        let min_dimension = core::cmp::min(num_rows, num_cols);
        for idx in 0..min_dimension {
            let a_idx_idx = d_matrix.read_entry(idx, idx);
            if !a_idx_idx.0 {
                self.0 .1.set(idx, false);
            }
        }
    }
}

impl MulAssign<F2> for F2Matrix {
    fn mul_assign(&mut self, rhs: F2) {
        if !rhs.0 {
            *self = Self::new(self.rows, self.cols, None);
        }
    }
}

impl Mul<F2> for F2Matrix {
    type Output = Self;
    fn mul(self, rhs: F2) -> Self::Output {
        if rhs.0 {
            self
        } else {
            Self::new(self.rows, self.cols, None)
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
#[repr(transparent)]
pub struct F2ColumnVec((BasisIndexing, BitVec));

impl From<(BasisIndexing, Vec<(F2, BasisIndexing)>)> for F2ColumnVec {
    fn from((overall_dimension, entries): (BasisIndexing, Vec<(F2, BasisIndexing)>)) -> Self {
        let mut entries_bitvec = BitVec::repeat(false, overall_dimension);
        for (r0, r1) in entries {
            if r0.0 {
                let my_index = r1 % overall_dimension;
                let old_value = entries_bitvec
                    .get(my_index)
                    .as_deref()
                    .copied()
                    .unwrap_or(false);
                entries_bitvec.set(my_index, !old_value);
            }
        }
        Self((overall_dimension, entries_bitvec))
    }
}

impl AsBasisCombination<F2> for F2ColumnVec
where
    F2: 'static,
{
    fn make_entries(&self) -> LazyLinear<F2, BasisIndexing> {
        let overall_dimension = self.0 .0;
        let self_bit_vec = self.0 .1.clone();
        let my_iterator = (0..overall_dimension).filter_map(move |idx| {
            if self_bit_vec
                .get(idx % overall_dimension)
                .as_deref()
                .copied()
                .unwrap_or(false)
            {
                Some((F2::ring_one(), idx))
            } else {
                None
            }
        });
        LazyLinear::<_, _> {
            summands: Box::new(my_iterator),
        }
    }
}

impl AddAssign<(F2, BasisIndexing)> for F2ColumnVec {
    fn add_assign(&mut self, rhs: (F2, BasisIndexing)) {
        if rhs.0 .0 {
            let my_index = rhs.1 % self.0 .0;
            let old_value = self.0 .1.get(my_index).as_deref().copied().unwrap_or(false);
            self.0 .1.set(my_index, !old_value);
        }
    }
}

impl AddAssign<F2ColumnVec> for F2ColumnVec {
    fn add_assign(&mut self, mut rhs: F2ColumnVec) {
        let mut self_len = self.0 .0;
        let mut rhs_len = rhs.0 .0;
        while self_len < rhs_len {
            self.0 .1.push(false);
            self_len += 1;
        }
        while rhs_len < self_len {
            rhs.0 .1.push(false);
            rhs_len += 1;
        }
        self.0 .1 ^= rhs.0 .1;
        self.0 .0 = self_len;
    }
}

impl MulAssign<F2> for F2ColumnVec {
    fn mul_assign(&mut self, rhs: F2) {
        if !rhs.0 {
            let new_len = self.0 .0;
            let new_vec = BitVec::repeat(false, new_len);
            *self = F2ColumnVec((new_len, new_vec));
        }
    }
}

impl MatrixStore<F2> for F2Matrix {
    type ColumnVector = F2ColumnVec;

    fn num_rows(&self) -> BasisIndexing {
        self.rows
    }

    fn num_cols(&self) -> BasisIndexing {
        self.cols
    }

    fn composed_eq_zero(&self, other: &Self) -> bool {
        assert_eq!(self.cols, other.rows);
        if self.is_zero_matrix() || other.is_zero_matrix() {
            return true;
        }

        let mut result_data = Vec::with_capacity(self.rows);
        for i in 0..self.rows {
            let mut row: BitVec<BasisIndexing, bitvec::order::Lsb0> =
                BitVec::repeat(false, other.cols);
            for j in 0..other.cols {
                let mut sum = false;
                for k in 0..self.cols {
                    sum ^= self.data[i][k] & other.data[k][j];
                }
                if sum {
                    return false;
                }
                row.set(j, sum);
            }
            result_data.push(row);
        }
        let product_matrix = F2Matrix::new(self.rows, other.cols, Some(result_data));
        product_matrix.is_zero_matrix()
    }

    fn zero_matrix(rows: BasisIndexing, cols: BasisIndexing) -> Self {
        Self::new(rows, cols, None)
    }

    fn transpose(self) -> Self {
        let mut result_data = vec![BitVec::repeat(false, self.rows); self.cols];
        for i in 0..self.rows {
            for (j, cur_result_data_row) in result_data.iter_mut().enumerate().take(self.cols) {
                cur_result_data_row.set(i, self.data[i][j]);
            }
        }
        F2Matrix::new(self.cols, self.rows, Some(result_data))
    }

    fn is_zero_matrix(&self) -> bool {
        for idx in 0..self.data.len() {
            for jdx in 0..self.data[idx].len() {
                if self.data[idx][jdx] {
                    return false;
                }
            }
        }
        true
    }

    fn identity(dimension: BasisIndexing) -> Self {
        let mut to_return = Self::new(dimension, dimension, None);
        for idx in 0..dimension {
            to_return.data[idx].set(idx, true);
        }
        to_return
    }

    fn diagonal_only(&self) -> Self {
        let (num_rows, num_cols) = self.dimensions();
        let min_dimension = core::cmp::min(num_rows, num_cols);
        let mut to_return = Self::zero_matrix(num_rows, num_cols);
        for idx in 0..min_dimension {
            let a_idx_idx = self.read_entry(idx, idx);
            to_return.set_entry(idx, idx, a_idx_idx);
        }
        to_return
    }
}

impl EffortfulMatrixStore<F2> for F2Matrix {
    fn rank(&self) -> BasisIndexing {
        todo!("rank of f2 matrix")
    }

    fn kernel(&self) -> BasisIndexing {
        todo!("kernel of f2 matrix")
    }

    fn kernel_basis(&self) -> Vec<Self::ColumnVector> {
        todo!("basis for kernel of f2 matrix")
    }
}

impl F2Matrix {
    /// Constructor to create a new matrix
    /// either constructs the 0 matrix if `None` is `data`
    /// or fills it with the specified entries (given row by row)
    /// # Panics
    /// if you specify `data` then `rows` and `columns` should match
    ///     - how many rows there actually are in `data`
    ///     - how many columns each of those rows in `data` actually have
    #[must_use]
    pub fn new(rows: BasisIndexing, cols: BasisIndexing, data: Option<Vec<BitVec>>) -> Self {
        if let Some(real_data) = data {
            assert_eq!(rows, real_data.len());
            assert!(real_data.iter().all(|row| row.len() == cols));
            F2Matrix {
                rows,
                cols,
                data: real_data,
            }
        } else {
            let mut data = Vec::with_capacity(rows);
            for _idx in 0..rows {
                let new_row = BitVec::repeat(false, cols);
                data.push(new_row);
            }
            F2Matrix { rows, cols, data }
        }
    }

    /// # Panics
    /// on dimension mismatch
    /// both `self` and `other` must have the same number of rows and columns
    #[must_use]
    pub fn add_borrows(&self, other: &Self) -> Self {
        assert_eq!(self.rows, other.rows);
        assert_eq!(self.cols, other.cols);

        let mut result_data = Vec::with_capacity(self.rows);
        for i in 0..self.rows {
            let mut row = self.data[i].clone();
            row ^= &other.data[i];
            result_data.push(row);
        }

        F2Matrix::new(self.rows, self.cols, Some(result_data))
    }

    /// # Panics
    /// on dimension mismatch of `self`s columns and `other`s rows
    #[must_use]
    pub fn multiply_borrows(&self, other: &Self) -> Self {
        assert_eq!(self.cols, other.rows);

        let mut result_data = Vec::with_capacity(self.rows);

        let other_transpose = other.clone().transpose();
        for i in 0..self.rows {
            let self_row_i = self.data[i].clone();
            let mut new_row = BitVec::repeat(false, other.cols);
            for j in 0..other.cols {
                let mut other_col_j = other_transpose.data[j].clone();
                other_col_j &= self_row_i.clone();
                let ij_entry = other_col_j.count_ones() % 2 == 1;
                new_row.set(j, ij_entry);
            }
            result_data.push(new_row);
        }

        F2Matrix::new(self.rows, other.cols, Some(result_data))
    }

    pub fn print(&self) {
        println!("{} by {}", self.rows, self.cols);
        for row in &self.data {
            println!("{row:?}");
        }
    }

    #[must_use]
    pub fn read_row_entries(&self, i: BasisIndexing, js: &[BasisIndexing]) -> Vec<F2> {
        let relevant_row = self.data[i].clone();
        js.iter()
            .map(|j| F2(relevant_row.get(*j).as_deref().copied().unwrap_or(false)))
            .collect()
    }
}

impl RowReductionHelpers<F2> for F2Matrix {
    #[allow(clippy::similar_names)]
    fn swap_rows(&mut self, row_idx: BasisIndexing, row_jdx: BasisIndexing) {
        self.data.swap(row_idx, row_jdx);
    }

    #[allow(clippy::similar_names)]
    fn add_assign_rows(&mut self, row_idx: BasisIndexing, row_jdx: BasisIndexing) {
        let row_i = self.data[row_idx].clone();
        self.data[row_jdx] ^= row_i;
    }

    fn scale_row(&mut self, row_idx: BasisIndexing, factor: F2) {
        if !factor.0 {
            self.data[row_idx] = BitVec::repeat(false, self.num_cols());
        }
    }

    fn read_entry(&self, row_idx: BasisIndexing, col_idx: BasisIndexing) -> F2 {
        F2(self.data[row_idx]
            .get(col_idx)
            .as_deref()
            .copied()
            .unwrap_or(false))
    }
    fn set_entry(&mut self, row_idx: BasisIndexing, col_idx: BasisIndexing, new_value: F2) {
        match new_value {
            F2(true) => {
                self.data[row_idx].set(col_idx, true);
            }
            F2(false) => {
                self.data[row_idx].set(col_idx, false);
            }
        }
    }

    #[allow(clippy::similar_names)]
    fn add_assign_factor_rows(
        &mut self,
        row_idx: BasisIndexing,
        factor: F2,
        row_jdx: BasisIndexing,
    ) {
        if factor.0 {
            self.add_assign_rows(row_idx, row_jdx);
        }
    }
}

impl Canonicalizable<F2> for F2Matrix {
    #[allow(unused_mut)]
    fn canonicalize(mut self) -> FactorizedMatrix<F2, Self> {
        let num_cols = self.num_cols();
        let (left_invertible, middle) = row_echelon_form(self, false);
        FactorizedMatrix::<F2, Self> {
            left_invertible,
            middle,
            right_invertible: ElementaryMatrixProduct::new(num_cols),
        }
    }
}

mod test {

    #[allow(dead_code)]
    const DO_PRINT: bool = false;

    #[allow(clippy::many_single_char_names, clippy::similar_names)]
    #[test]
    fn basic_test() {
        use super::{F2ColumnVec, F2Matrix};
        use crate::linear_algebra::matrix_store::{LeftMultipliesBy, MatrixStore};
        use bitvec::vec::BitVec;

        let mut one_zero = BitVec::new();
        one_zero.insert(0, true);
        one_zero.insert(1, false);
        let mut one_one = BitVec::new();
        one_one.insert(0, true);
        one_one.insert(1, true);
        let mut zero_one = BitVec::new();
        zero_one.insert(0, false);
        zero_one.insert(1, true);

        // Example usage
        let a = F2Matrix::new(2, 2, Some(vec![one_zero.clone(), one_one.clone()]));
        let a_transpose = F2Matrix::new(2, 2, Some(vec![one_one.clone(), zero_one.clone()]));
        let b = F2Matrix::new(2, 2, Some(vec![zero_one.clone(), one_zero]));
        let a_plus_b = F2Matrix::new(2, 2, Some(vec![one_one.clone(), zero_one.clone()]));
        let a_times_b = F2Matrix::new(2, 2, Some(vec![zero_one, one_one]));

        if DO_PRINT {
            println!("Matrix A:");
            a.print();
            println!("Matrix B:");
            b.print();
        }

        let c = a.add_borrows(&b);
        if DO_PRINT {
            println!("A + B:");
            c.print();
        }
        assert_eq!(c, a_plus_b);

        let d = a.multiply_borrows(&b);
        if DO_PRINT {
            println!("A * B:");
            d.print();
        }
        assert_eq!(d, a_times_b);

        let e = a.transpose();
        if DO_PRINT {
            println!("Transpose of A:");
            e.print();
        }
        assert_eq!(e, a_transpose);

        let mut one_zero = BitVec::new();
        one_zero.insert(0, true);
        one_zero.insert(1, false);
        let mut x_axis = F2ColumnVec((2, one_zero));
        let expected_after_e = x_axis.clone();
        x_axis.left_multiply(&e);
        assert_eq!(x_axis, expected_after_e);

        let mut one_one = BitVec::new();
        one_one.insert(0, true);
        one_one.insert(1, true);
        let mut xy_axis = F2ColumnVec((2, one_one));

        let mut zero_one = BitVec::new();
        zero_one.insert(0, false);
        zero_one.insert(1, true);
        let mut y_axis = F2ColumnVec((2, zero_one));
        let expected_after_e = xy_axis.clone();
        let fixed_y_axis = y_axis.clone();
        y_axis.left_multiply(&e);
        assert_eq!(y_axis, expected_after_e);

        xy_axis.left_multiply(&e);
        assert_eq!(xy_axis, fixed_y_axis);
    }

    #[test]
    fn elementaries() {
        use super::{F2Matrix, F2};
        use crate::linear_algebra::elementary_matrix::{ElementaryMatrix, ElementaryMatrixProduct};
        use crate::linear_algebra::matrix_store::MatrixStore;
        let a_under = ElementaryMatrix::<F2>::AddAssignRow(1, 0);
        let a: F2Matrix = Into::<ElementaryMatrixProduct<F2>>::into((2, a_under.clone())).into();
        let mut a_expected = F2Matrix::identity(2);
        a_expected.data[0].set(1, true);
        assert_eq!(a.clone(), a_expected);
        let b_under = ElementaryMatrix::<F2>::AddAssignRow(0, 1);
        let b: F2Matrix = Into::<ElementaryMatrixProduct<F2>>::into((2, b_under.clone())).into();
        let mut b_expected = F2Matrix::identity(2);
        b_expected.data[1].set(0, true);
        assert_eq!(b.clone(), b_expected);
        let ab: F2Matrix = ElementaryMatrixProduct {
            steps: vec![a_under, b_under].into(),
            dimension: 2,
        }
        .into();
        assert_eq!(a * b, ab);

        let c_under = ElementaryMatrix::<F2>::SwapRows(0, 1);
        let c: F2Matrix = Into::<ElementaryMatrixProduct<F2>>::into((2, c_under)).into();
        let mut c_expected = F2Matrix::new(2, 2, None);
        c_expected.data[0].set(1, true);
        c_expected.data[1].set(0, true);
        assert_eq!(c.clone(), c_expected);
        assert_eq!(c.clone() * c, F2Matrix::identity(2));

        let s12_under = ElementaryMatrix::<F2>::SwapRows(0, 1);
        let s12: F2Matrix =
            Into::<ElementaryMatrixProduct<F2>>::into((3, s12_under.clone())).into();
        let s23_under = ElementaryMatrix::<F2>::SwapRows(1, 2);
        let s23: F2Matrix =
            Into::<ElementaryMatrixProduct<F2>>::into((3, s23_under.clone())).into();
        let expected = ElementaryMatrixProduct {
            steps: vec![s12_under.clone(), s23_under.clone()].into(),
            dimension: 3,
        }
        .into();
        let backwards = ElementaryMatrixProduct {
            steps: vec![s23_under, s12_under].into(),
            dimension: 3,
        }
        .into();
        assert_eq!(s12 * s23, expected);
        assert!(expected != backwards);
    }
}
