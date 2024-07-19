use std::ops::{Add, AddAssign, Div, Mul, MulAssign, Sub};

use bitvec::prelude::BitVec;

use crate::field_generals::{Field, MatrixStore};

#[derive(PartialEq, Eq, Debug)]
#[repr(transparent)]
pub struct F2(bool);

impl Add for F2 {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        #[allow(clippy::suspicious_arithmetic_impl)]
        F2(self.0 ^ rhs.0)
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

impl Field for F2 {
    fn characteristic(_primes: Box<dyn Iterator<Item = usize>>) -> usize {
        2
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct F2Matrix {
    rows: usize,
    cols: usize,
    data: Vec<BitVec>, // Using BitVec to represent rows of the matrix
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

    fn add(self, rhs: Self) -> Self::Output {
        F2Matrix::add_borrows(&self, &rhs)
    }
}

impl MulAssign for F2Matrix {
    fn mul_assign(&mut self, rhs: Self) {
        *self = self.multiply_borrows(&rhs);
    }
}

impl Mul for F2Matrix {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        self.multiply_borrows(&rhs)
    }
}

impl MatrixStore<F2> for F2Matrix {
    fn num_rows(&self) -> usize {
        self.rows
    }

    fn num_cols(&self) -> usize {
        self.cols
    }

    #[allow(dead_code)]
    fn dimensions(&self) -> (usize, usize) {
        (self.num_rows(), self.num_cols())
    }

    fn composed_eq_zero(&self, other: &Self) -> bool {
        assert_eq!(self.cols, other.rows);

        let mut result_data = Vec::with_capacity(self.rows);
        for i in 0..self.rows {
            let mut row: BitVec<usize, bitvec::order::Lsb0> = BitVec::repeat(false, other.cols);
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

    fn zero_matrix(rows: usize, cols: usize) -> Self {
        Self::new(rows, cols, None)
    }

    fn transpose(&self) -> Self {
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

    #[allow(dead_code)]
    fn identity(dimension: usize) -> Self {
        let mut to_return = Self::new(dimension, dimension, None);
        for idx in 0..dimension {
            to_return.data[idx].set(idx, true);
        }
        to_return
    }

    fn rank(&self) -> usize {
        todo!()
    }

    fn kernel(&self) -> usize {
        todo!()
    }

    fn kernel_basis(&self) -> Vec<Self> {
        todo!()
    }
}

#[allow(dead_code)]
impl F2Matrix {
    // Constructor to create a new matrix
    pub fn new(rows: usize, cols: usize, data: Option<Vec<BitVec>>) -> Self {
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

    pub fn multiply_borrows(&self, other: &Self) -> Self {
        assert_eq!(self.cols, other.rows);

        let mut result_data = Vec::with_capacity(self.rows);

        let other_transpose = other.transpose();
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

    // Utility to print matrix
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
        use crate::field_generals::MatrixStore;
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
}
