use crate::{
    elementary_matrix::{ElementaryMatrix, ElementaryMatrixProduct},
    factorized_matrix::RowReductionHelpers,
    field_generals::Ring,
    linear_comb::LazyLinear,
    matrix_store::{
        AsBasisCombination, BasisIndexing, EffortfulMatrixStore, LeftMultipliesBy, MatrixStore,
    },
};
use core::ops::{Add, AddAssign, Mul, MulAssign};

#[derive(PartialEq, Clone)]
#[repr(transparent)]
pub struct SquareMatrixStore<const N: usize, F: Ring + Clone> {
    pub(crate) each_entry: [[F; N]; N],
}

impl<const N: usize, F: Ring + Clone + core::fmt::Debug> core::fmt::Debug
    for SquareMatrixStore<N, F>
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SquareMatrixStore")
            .field("each_entry", &self.each_entry)
            .finish()
    }
}

#[repr(transparent)]
pub struct ArrayVectorStore<const N: usize, F: Ring> {
    entries: [F; N],
}

impl<const N: usize, F: Ring + Clone> Clone for ArrayVectorStore<N, F> {
    fn clone(&self) -> Self {
        Self {
            entries: self.entries.clone(),
        }
    }
}

impl<const N: usize, F: Ring + Clone> AddAssign<(F, BasisIndexing)> for ArrayVectorStore<N, F> {
    fn add_assign(&mut self, rhs: (F, BasisIndexing)) {
        let old_entry = self.entries[rhs.1].clone();
        self.entries[rhs.1] = old_entry + rhs.0;
    }
}

impl<const N: usize, F: Ring + Clone> AddAssign<Self> for ArrayVectorStore<N, F> {
    fn add_assign(&mut self, rhs: Self) {
        self.entries.iter_mut().zip(rhs.entries).for_each(|(z, w)| {
            let old_value = z.clone();
            *z = old_value + w;
        })
    }
}

impl<const N: usize, F: Ring + Clone> MulAssign<F> for ArrayVectorStore<N, F> {
    fn mul_assign(&mut self, rhs: F) {
        self.entries.iter_mut().for_each(|w| {
            let old_value = w.clone();
            *w = old_value * rhs.clone();
        });
    }
}

impl<const N: usize, F: Ring + Clone> AsBasisCombination<F> for ArrayVectorStore<N, F>
where
    F: 'static,
{
    fn make_entries(&self) -> LazyLinear<F, BasisIndexing> {
        let entries = self.entries.clone();
        let my_iterator = entries.into_iter().enumerate().map(|(a, b)| (b, a));
        LazyLinear::<_, _> {
            summands: Box::new(my_iterator),
        }
    }
}

impl<const N: usize, F: Ring + Clone> From<(BasisIndexing, Vec<(F, BasisIndexing)>)>
    for ArrayVectorStore<N, F>
where
    F: 'static,
{
    fn from(value: (BasisIndexing, Vec<(F, BasisIndexing)>)) -> Self {
        assert_eq!(
            value.0, N,
            "Fixed size at compile time unlike other ColumnVector types"
        );
        let mut to_return = Self {
            entries: core::array::from_fn(|_| 0.into()),
        };
        for (value, which_idx) in value.1 {
            to_return.entries[which_idx % N] = value;
        }
        to_return
    }
}

impl<const N: usize, F: Ring + Clone> LeftMultipliesBy<SquareMatrixStore<N, F>>
    for ArrayVectorStore<N, F>
{
    fn left_multiply(&mut self, left_factor: &SquareMatrixStore<N, F>) {
        #[cfg(feature = "column-major")]
        {
            todo!()
        }

        #[cfg(not(feature = "column-major"))]
        {
            let zero_f: F = 0.into();
            let mut result_data = self.clone();
            self.entries = core::array::from_fn(|_| 0.into());

            #[allow(clippy::needless_range_loop)]
            for i in 0..N {
                let i_entry = left_factor.each_entry[i].iter().zip(&self.entries).fold(
                    0.into(),
                    |acc, (x, y)| {
                        if *x != zero_f && *y != zero_f {
                            acc + x.clone() * y.clone()
                        } else {
                            acc
                        }
                    },
                );

                result_data.entries[i] = i_entry;
            }

            *self = result_data;
        }
    }

    fn zero_out(&mut self, keep_length: bool) {
        if !keep_length {
            panic!("Cannot zero out length, fixed size");
        }
        self.entries = core::array::from_fn(|_| 0.into())
    }

    fn zero_pad(&mut self, how_much: BasisIndexing) {
        if how_much > 0 {
            panic!("Cannot pad, fixed size");
        }
    }

    fn left_multiply_by_diagonal(&mut self, _d_matrix: &SquareMatrixStore<N, F>) {
        todo!()
    }

    fn left_multiply_by_triangular(
        &mut self,
        _lower_or_upper: bool,
        _l_or_u_matrix: &SquareMatrixStore<N, F>,
    ) {
        todo!()
    }
}

impl<const N: usize, F: 'static + Ring + Clone> AddAssign<Self> for SquareMatrixStore<N, F> {
    fn add_assign(&mut self, mut other: Self) {
        for idx in 0..N {
            for jdx in 0..N {
                let old_entry = self.each_entry[idx][jdx].clone();
                let mut dummy_entry: F = 0.into();
                core::mem::swap(&mut dummy_entry, &mut other.each_entry[idx][jdx]);
                self.each_entry[idx][jdx] = old_entry + dummy_entry;
            }
        }
    }
}

impl<const N: usize, F: 'static + Ring + Clone> Add<Self> for SquareMatrixStore<N, F> {
    type Output = Self;

    fn add(mut self, rhs: Self) -> Self::Output {
        self += rhs;
        self
    }
}

impl<const N: usize, F: 'static + Ring + Clone> Mul<F> for SquareMatrixStore<N, F> {
    type Output = Self;
    fn mul(mut self, rhs: F) -> Self {
        self *= rhs;
        self
    }
}

impl<const N: usize, F: 'static + Ring + Clone> MulAssign<F> for SquareMatrixStore<N, F> {
    fn mul_assign(&mut self, rhs: F) {
        self.each_entry.iter_mut().for_each(|z| {
            z.iter_mut().for_each(|w| {
                let old_value = w.clone();
                *w = old_value * rhs.clone();
            })
        });
    }
}

impl<const N: usize, F: 'static + Ring + Clone> Mul<Self> for SquareMatrixStore<N, F> {
    type Output = Self;

    fn mul(self, other: Self) -> Self::Output {
        #[cfg(feature = "column-major")]
        {
            todo!()
        }

        #[cfg(not(feature = "column-major"))]
        {
            let zero_f: F = 0.into();
            let mut result_data = Self::zero_matrix(N, N).each_entry;

            let other_transpose = other.transpose();
            #[allow(clippy::needless_range_loop)]
            for i in 0..N {
                let self_row_i = self.each_entry[i].clone();
                let mut new_row: [F; N] = core::array::from_fn(|_| 0.into());
                for (j, other_col_j) in other_transpose.each_entry.iter().enumerate() {
                    let ij_entry =
                        self_row_i
                            .iter()
                            .zip(other_col_j)
                            .fold(0.into(), |acc, (x, y)| {
                                if *x != zero_f && *y != zero_f {
                                    acc + x.clone() * y.clone()
                                } else {
                                    acc
                                }
                            });
                    new_row[j] = ij_entry;
                }
                result_data[i] = new_row;
            }

            Self {
                each_entry: result_data,
            }
        }
    }
}

impl<const N: usize, F: 'static + Ring + Clone> RowReductionHelpers<F> for SquareMatrixStore<N, F> {
    fn swap_rows(&mut self, row_idx: BasisIndexing, row_jdx: BasisIndexing) {
        #[cfg(feature = "column-major")]
        {
            todo!()
        }

        #[cfg(not(feature = "column-major"))]
        {
            self.each_entry.swap(row_idx % N, row_jdx % N);
        }
    }

    fn add_assign_rows(&mut self, row_idx: BasisIndexing, row_jdx: BasisIndexing) {
        #[cfg(feature = "column-major")]
        {
            todo!()
        }

        #[cfg(not(feature = "column-major"))]
        {
            let row_idx = self.each_entry[row_idx % N].clone();
            self.each_entry[row_jdx % N]
                .iter_mut()
                .zip(row_idx)
                .for_each(|(z, w)| {
                    let old_value = z.clone();
                    *z = old_value + w;
                })
        }
    }

    fn add_assign_factor_rows(
        &mut self,
        row_idx: BasisIndexing,
        factor: F,
        row_jdx: BasisIndexing,
    ) {
        #[cfg(feature = "column-major")]
        {
            todo!()
        }

        #[cfg(not(feature = "column-major"))]
        {
            let row_idx = self.each_entry[row_idx % N].clone();
            self.each_entry[row_jdx % N]
                .iter_mut()
                .zip(row_idx)
                .for_each(|(z, w)| {
                    let old_value = z.clone();
                    *z = old_value + w * factor.clone();
                })
        }
    }

    fn scale_row(&mut self, row_idx: BasisIndexing, factor: F) {
        #[cfg(feature = "column-major")]
        {
            todo!()
        }

        #[cfg(not(feature = "column-major"))]
        {
            self.each_entry[row_idx % N].iter_mut().for_each(|z| {
                let old_value = z.clone();
                *z = old_value * factor.clone();
            })
        }
    }

    fn read_entry(&self, row_idx: BasisIndexing, col_idx: BasisIndexing) -> F {
        #[cfg(feature = "column-major")]
        {
            self.each_entry[col_idx][row_idx].clone()
        }

        #[cfg(not(feature = "column-major"))]
        {
            self.each_entry[row_idx][col_idx].clone()
        }
    }

    fn set_entry(&mut self, row_idx: BasisIndexing, col_idx: BasisIndexing, new_value: F) {
        #[cfg(feature = "column-major")]
        {
            self.each_entry[col_idx][row_idx] = new_value;
        }

        #[cfg(not(feature = "column-major"))]
        {
            self.each_entry[row_idx][col_idx] = new_value;
        }
    }
}

impl<const N: usize, F: 'static + Ring + Clone> From<ElementaryMatrixProduct<F>>
    for SquareMatrixStore<N, F>
{
    fn from(value: ElementaryMatrixProduct<F>) -> Self {
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

impl<const N: usize, F: 'static + Ring + Clone> MatrixStore<F> for SquareMatrixStore<N, F> {
    type ColumnVector = ArrayVectorStore<N, F>;

    fn zero_matrix(rows: BasisIndexing, cols: BasisIndexing) -> Self {
        assert_eq!(rows, N);
        assert_eq!(cols, N);
        Self {
            each_entry: core::array::from_fn(|_| core::array::from_fn(|_| 0.into())),
        }
    }

    fn identity(dimension: BasisIndexing) -> Self {
        let mut to_return = Self::zero_matrix(dimension, dimension);
        for i in 0..dimension {
            to_return.each_entry[i][i] = 1.into();
        }
        to_return
    }

    fn num_rows(&self) -> BasisIndexing {
        N
    }

    fn num_cols(&self) -> BasisIndexing {
        N
    }

    fn is_zero_matrix(&self) -> bool {
        let zero_f: F = 0.into();
        self.each_entry
            .iter()
            .all(|e| e.iter().all(|z| *z == zero_f))
    }

    fn composed_eq_zero(&self, other: &Self) -> bool {
        if self.is_zero_matrix() || other.is_zero_matrix() {
            return true;
        }
        let zero_f: F = 0.into();
        #[cfg(feature = "column-major")]
        {
            todo!()
        }

        #[cfg(not(feature = "column-major"))]
        {
            let mut result_data = Self::zero_matrix(N, N).each_entry;

            let other_transpose = other.clone().transpose();
            #[allow(clippy::needless_range_loop)]
            for i in 0..N {
                let self_row_i = self.each_entry[i].clone();
                let mut new_row: [F; N] = core::array::from_fn(|_| 0.into());
                for (j, other_col_j) in other_transpose.each_entry.iter().enumerate() {
                    let ij_entry =
                        self_row_i
                            .iter()
                            .zip(other_col_j)
                            .fold(0.into(), |acc, (x, y)| {
                                if *x != zero_f && *y != zero_f {
                                    acc + x.clone() * y.clone()
                                } else {
                                    acc
                                }
                            });
                    if ij_entry != zero_f {
                        return false;
                    }
                    new_row[j] = ij_entry;
                }
                result_data[i] = new_row;
            }

            let result = Self {
                each_entry: result_data,
            };
            result.is_zero_matrix()
        }
    }

    fn transpose(mut self) -> Self {
        for idx in 0..N {
            for jdx in idx + 1..N {
                let mut dummy: F = 0.into();
                core::mem::swap(&mut self.each_entry[idx][jdx], &mut dummy);
                core::mem::swap(&mut self.each_entry[jdx][idx], &mut dummy);
                core::mem::swap(&mut self.each_entry[idx][jdx], &mut dummy);
            }
        }
        self
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

impl<const N: usize, F: 'static + Ring + Clone> EffortfulMatrixStore<F>
    for SquareMatrixStore<N, F>
{
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

mod test {

    #[test]
    fn elementaries() {
        use super::SquareMatrixStore;
        use crate::elementary_matrix::{ElementaryMatrix, ElementaryMatrixProduct};
        use crate::factorized_matrix::RowReductionHelpers;
        use crate::matrix_store::MatrixStore;
        let a_under = ElementaryMatrix::<f32>::AddAssignMultipleRow(1, 5., 0);
        let a: SquareMatrixStore<2, f32> =
            Into::<ElementaryMatrixProduct<f32>>::into((2, a_under.clone())).into();
        let mut a_expected = SquareMatrixStore::<2, f32>::identity(2);
        a_expected.set_entry(0, 1, 5.);
        assert_eq!(a.clone(), a_expected);
        let b_under = ElementaryMatrix::<f32>::AddAssignMultipleRow(0, -3., 1);
        let b: SquareMatrixStore<2, f32> =
            Into::<ElementaryMatrixProduct<f32>>::into((2, b_under.clone())).into();
        let mut b_expected = SquareMatrixStore::<2, f32>::identity(2);
        b_expected.set_entry(1, 0, -3.);
        assert_eq!(b.clone(), b_expected);
        let ab: SquareMatrixStore<2, f32> = ElementaryMatrixProduct {
            steps: vec![a_under.clone(), b_under.clone()].into(),
            dimension: 2,
        }
        .into();
        assert_eq!(a.clone() * b.clone(), ab);
        let ba: SquareMatrixStore<2, f32> = ElementaryMatrixProduct {
            steps: vec![b_under, a_under].into(),
            dimension: 2,
        }
        .into();
        assert!(a * b != ba);

        let c_under = ElementaryMatrix::<f32>::SwapRows(0, 1);
        let c: SquareMatrixStore<2, f32> =
            Into::<ElementaryMatrixProduct<f32>>::into((2, c_under)).into();
        let mut c_expected = SquareMatrixStore::<2, f32>::zero_matrix(2, 2);
        c_expected.set_entry(0, 1, (1 as i8).into());
        c_expected.set_entry(1, 0, (1 as i8).into());
        assert_eq!(c.clone(), c_expected);
        assert_eq!(c.clone() * c, SquareMatrixStore::<2, f32>::identity(2));

        let s12_under = ElementaryMatrix::<f32>::SwapRows(0, 1);
        let s12: SquareMatrixStore<3, f32> =
            Into::<ElementaryMatrixProduct<f32>>::into((3, s12_under.clone())).into();
        let s23_under = ElementaryMatrix::<f32>::SwapRows(1, 2);
        let s23: SquareMatrixStore<3, f32> =
            Into::<ElementaryMatrixProduct<f32>>::into((3, s23_under.clone())).into();
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
