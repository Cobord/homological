use core::ops::{Add, AddAssign, Div, Mul, MulAssign, Neg, Sub};

use bitvec::prelude::BitVec;

use crate::elementary_matrix::{ElementaryMatrix, ElementaryMatrixProduct};
use crate::field_generals::{Field, Ring};
use crate::linear_comb::Commutative;
use crate::matrix_store::{BasisIndexing, LeftMultipliesBy, MatrixStore};

#[derive(PartialEq, Eq, Debug, Clone)]
#[repr(transparent)]
pub struct F2(bool);

impl Add for F2 {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        #[allow(clippy::suspicious_arithmetic_impl)]
        F2(self.0 ^ rhs.0)
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
        match rhs.0 {
            true => self,
            false => panic!("Division by 0"),
        }
    }
}

impl From<usize> for F2 {
    fn from(value: usize) -> Self {
        Self(value % 2 == 0)
    }
}

impl Ring for F2 {
    fn characteristic(_primes: Box<dyn Iterator<Item = usize>>) -> usize {
        2
    }

    fn try_inverse(self) -> Option<Self> {
        Some(self)
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
    fn from(value: ElementaryMatrixProduct<F2>) -> Self {
        let mut result = Self::identity(value.dimension);
        for step in value.steps.into_iter().rev() {
            match step {
                ElementaryMatrix::SwapRows(row_idx, row_jdx) => {
                    result.data.swap(row_idx, row_jdx);
                }
                ElementaryMatrix::AddAssignRow(row_idx, row_jdx) => {
                    let row_i = result.data[row_idx].clone();
                    result.data[row_jdx] ^= row_i;
                }
                ElementaryMatrix::ScaleRow(row_idx, factor) => match factor.0 {
                    true => {}
                    false => result.data[row_idx] = BitVec::repeat(false, value.dimension),
                },
            }
        }
        result
    }
}

impl LeftMultipliesBy<F2Matrix> for F2ColumnVec {
    fn left_multiply(&mut self, _left_factor: &F2Matrix) {
        todo!()
    }
    
    fn zero_out(&mut self) {
        let new_vec = BitVec::repeat(false, 0);
        *self = F2ColumnVec((0,new_vec));
    }

    
}

impl MulAssign<F2> for F2Matrix {
    fn mul_assign(&mut self, rhs: F2) {
        match rhs.0 {
            true => {}
            false => *self = Self::new(self.rows, self.cols, None),
        }
    }
}

#[repr(transparent)]
pub struct F2ColumnVec((BasisIndexing, BitVec));

impl From<(BasisIndexing, Vec<(F2, BasisIndexing)>)> for F2ColumnVec {
    fn from(_value: (BasisIndexing, Vec<(F2, BasisIndexing)>)) -> Self {
        todo!()
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

    fn rank(&self) -> BasisIndexing {
        todo!()
    }

    fn kernel(&self) -> BasisIndexing {
        todo!()
    }

    fn kernel_basis(&self) -> Vec<Self::ColumnVector> {
        todo!()
    }
}

impl F2Matrix {
    // Constructor to create a new matrix
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

    #[allow(dead_code)]
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

    #[allow(dead_code)]
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

    #[allow(dead_code)]
    pub fn print(&self) {
        println!("{} by {}", self.rows, self.cols);
        for row in &self.data {
            println!("{:?}", row);
        }
    }
}

mod test {

    #[test]
    fn basic_test() {
        use super::F2Matrix;
        use crate::matrix_store::MatrixStore;
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

        let do_print = false;

        // Example usage
        let a = F2Matrix::new(2, 2, Some(vec![one_zero.clone(), one_one.clone()]));
        let a_transpose = F2Matrix::new(2, 2, Some(vec![one_one.clone(), zero_one.clone()]));
        let b = F2Matrix::new(2, 2, Some(vec![zero_one.clone(), one_zero]));
        let a_plus_b = F2Matrix::new(2, 2, Some(vec![one_one.clone(), zero_one.clone()]));
        let a_times_b = F2Matrix::new(2, 2, Some(vec![zero_one, one_one]));

        if do_print {
            println!("Matrix A:");
            a.print();
            println!("Matrix B:");
            b.print();
        }

        let c = a.add_borrows(&b);
        if do_print {
            println!("A + B:");
            c.print();
        }
        assert_eq!(c, a_plus_b);

        let d = a.multiply_borrows(&b);
        if do_print {
            println!("A * B:");
            d.print();
        }
        assert_eq!(d, a_times_b);

        let e = a.transpose();
        if do_print {
            println!("Transpose of A:");
            e.print();
        }
        assert_eq!(e, a_transpose);
    }

    #[test]
    fn elementaries() {
        use super::{F2Matrix, F2};
        use crate::elementary_matrix::{ElementaryMatrix, ElementaryMatrixProduct};
        use crate::matrix_store::MatrixStore;
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
