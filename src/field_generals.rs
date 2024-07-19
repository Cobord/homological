use std::ops::{Add, Div, Mul, Sub};

pub trait Field: Add + Sub + Mul + Div + Eq + From<usize> + Sized {
    fn characteristic(primes: Box<dyn Iterator<Item=usize>>) -> usize {
        let zero_f = Self::from(0);
        for i in primes {
            if Self::from(i) == zero_f {
                return i;
            }
        }
        return 0;
    }
}

pub trait MatrixStore<F: Field>: Add + Mul + Sized {
    fn zero_matrix(rows: usize, cols: usize) -> Self;
    fn identity(dimension: usize) -> Self;
    fn num_rows(&self) -> usize;
    fn num_cols(&self) -> usize;
    fn dimensions(&self) -> (usize, usize);
    fn is_zero_matrix(&self) -> bool;
    fn composed_eq_zero(&self, other: &Self) -> bool;
    fn transpose(&self) -> Self;
    fn rank(&self) -> usize;
    fn kernel(&self) -> usize;
    fn kernel_basis(&self) -> Vec<Self>;
    fn homology_info(&self, previous_d: &Self) -> (usize, Vec<Self>) {
        // the rank of homology with outgoing differential self and incoming
        // differential previous_d
        // also a basis for the kernel which will have some
        // linear dependencies when regarded as the equivalence classes in cohomology
        // but we haven't chosen a basis for the quotient, only the kernel
        assert!(self.composed_eq_zero(previous_d));
        let my_kernel_size = self.kernel();
        let previous_image_size = previous_d.rank();
        assert!(my_kernel_size >= previous_image_size);
        (my_kernel_size - previous_image_size, self.kernel_basis())
    }
}
