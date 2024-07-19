use std::ops::{Add, Div, Mul, Sub};

pub trait Field: Add + Sub + Mul + Div + Sized {}

pub trait MatrixStore<F: Field>: Add + Mul + Sized {
    fn zero_matrix(rows: usize, cols: usize) -> Self;
    fn identity(dimension: usize) -> Self;
    fn num_rows(&self) -> usize;
    fn num_cols(&self) -> usize;
    fn dimensions(&self) -> (usize, usize);
    fn is_zero_matrix(&self) -> bool;
    fn composed_eq_zero(&self, other: &Self) -> bool;
    fn transpose(&self) -> Self;
}
