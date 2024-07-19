use std::marker::PhantomData;

use crate::field_generals::{Field, MatrixStore};

pub trait HomSigns {
    fn differential_increases() -> bool;
}

struct HomologicalIndex {}
impl HomSigns for HomologicalIndex {
    fn differential_increases() -> bool {
        false
    }
}

struct CohomologicalIndex {}
impl HomSigns for CohomologicalIndex {
    fn differential_increases() -> bool {
        true
    }
}

pub struct ChainFVect<R: HomSigns, F: Field, M: MatrixStore<F>> {
    homological_index: i64,
    dimension: usize,
    differential: M,
    homsign: PhantomData<R>,
    field_info: PhantomData<F>,
    rest: Option<Box<ChainFVect<R, F, M>>>,
}

impl<R, F, M> ChainFVect<R, F, M>
where
    R: HomSigns,
    F: Field,
    M: MatrixStore<F>,
{
    pub fn concentrated_in_0(dimension: usize) -> Self {
        Self {
            homological_index: 0,
            dimension,
            differential: M::zero_matrix(0, dimension),
            field_info: PhantomData,
            homsign: PhantomData,
            rest: None,
        }
    }

    #[allow(dead_code)]
    pub fn dimensions_each_index(&self) -> Vec<(i64, usize)> {
        debug_assert!(self.dimension == self.differential.dimensions().1);
        let mut return_val = if let Some(real_rest) = self.rest.as_ref() {
            debug_assert!(real_rest.dimension == self.differential.dimensions().0);
            real_rest.dimensions_each_index()
        } else {
            debug_assert!(0 == self.differential.dimensions().0);
            vec![]
        };
        return_val.push((self.homological_index, self.dimension));
        return_val
    }

    #[allow(dead_code)]
    pub fn shift_all(&mut self, how_much: i64) {
        self.homological_index += how_much;
        if let Some(real_rest) = self.rest.as_deref_mut() {
            real_rest.shift_all(how_much);
        }
    }

    pub fn prepend_zero_spaces(&mut self, how_many: usize) {
        if how_many == 0 {
            return;
        }
        self.prepend_zero_space();
        self.prepend_zero_spaces(how_many - 1);
    }

    fn prepend_zero_space(&mut self) {
        let new_index = if R::differential_increases() {
            self.homological_index - 1
        } else {
            self.homological_index + 1
        };
        let next_dimension = self.dimension;
        let mut fake_self = Self::concentrated_in_0(0);
        std::mem::swap(&mut fake_self, self);
        *self = Self {
            homological_index: new_index,
            dimension: 0,
            differential: M::zero_matrix(next_dimension, 0),
            field_info: PhantomData,
            homsign: PhantomData,
            rest: Some(Box::new(fake_self)),
        };
    }

    #[allow(dead_code)]
    pub fn align_together(&mut self, other: &mut Self) {
        let self_index = self.homological_index;
        let other_index = other.homological_index;
        if self_index == other_index {
            return;
        }
        let to_prepend = other_index.abs_diff(self_index) as usize;
        if R::differential_increases() {
            if other_index > self_index {
                other.prepend_zero_spaces(to_prepend);
            } else {
                self.prepend_zero_spaces(to_prepend);
            }
        } else {
            #[allow(clippy::collapsible_else_if)]
            if other_index > self_index {
                self.prepend_zero_spaces(to_prepend);
            } else {
                other.prepend_zero_spaces(to_prepend);
            }
        }
        let self_index = self.homological_index;
        let other_index = other.homological_index;
        assert_eq!(self_index, other_index);
    }

    #[allow(dead_code)]
    pub fn prepend_space(&mut self, new_dimension: usize, new_differential: M) -> Result<(), bool> {
        let new_index = if R::differential_increases() {
            self.homological_index - 1
        } else {
            self.homological_index + 1
        };
        let next_dimension = self.dimension;
        let should_be_next_dimension = new_differential.num_rows();
        if next_dimension != should_be_next_dimension {
            return Err(false);
        }
        let next_differential = &self.differential;
        if !M::composed_eq_zero(next_differential, &new_differential) {
            return Err(true);
        }
        let mut fake_self = Self::concentrated_in_0(0);
        std::mem::swap(&mut fake_self, self);
        *self = Self {
            homological_index: new_index,
            dimension: new_dimension,
            differential: new_differential,
            field_info: PhantomData,
            homsign: PhantomData,
            rest: Some(Box::new(fake_self)),
        };
        Ok(())
    }
}

impl<F, M> ChainFVect<CohomologicalIndex, F, M>
where
    F: Field,
    M: MatrixStore<F>,
{
    #[allow(dead_code)]
    pub fn negate_homological_indices(self) -> ChainFVect<HomologicalIndex, F, M> {
        ChainFVect::<HomologicalIndex, F, M> {
            homological_index: -self.homological_index,
            dimension: self.dimension,
            differential: self.differential,
            field_info: PhantomData,
            homsign: PhantomData,
            rest: self
                .rest
                .map(|real_rest| Box::new(real_rest.negate_homological_indices())),
        }
    }
}

impl<F, M> ChainFVect<HomologicalIndex, F, M>
where
    F: Field,
    M: MatrixStore<F>,
{
    #[allow(dead_code)]
    pub fn negate_homological_indices(self) -> ChainFVect<CohomologicalIndex, F, M> {
        ChainFVect::<CohomologicalIndex, F, M> {
            homological_index: -self.homological_index,
            dimension: self.dimension,
            differential: self.differential,
            homsign: PhantomData,
            field_info: PhantomData,
            rest: self
                .rest
                .map(|real_rest| Box::new(real_rest.negate_homological_indices())),
        }
    }
}

mod test {

    #[test]
    fn two_term_id_complex() {
        use super::{ChainFVect, CohomologicalIndex};
        use crate::f2_vect::{F2Matrix, F2};
        use crate::field_generals::MatrixStore;
        let dimension = 5;
        let identity_matrix = F2Matrix::identity(dimension);
        assert!(!F2Matrix::composed_eq_zero(
            &identity_matrix,
            &identity_matrix
        ));
        let mut x = ChainFVect::<CohomologicalIndex, F2, F2Matrix>::concentrated_in_0(dimension);
        let should_ok = x.prepend_space(dimension, F2Matrix::identity(dimension));
        assert!(should_ok.is_ok());
        let dim_by_idx = x.dimensions_each_index();
        assert_eq!(dim_by_idx, vec![(0, dimension), (-1, dimension)]);
        let should_err = x.prepend_space(dimension, F2Matrix::identity(dimension));
        assert_eq!(should_err, Err(true));
        let should_ok = x.prepend_space(dimension, F2Matrix::new(dimension, dimension, None));
        assert!(should_ok.is_ok());
        let dim_by_idx = x.dimensions_each_index();
        assert_eq!(
            dim_by_idx,
            vec![(0, dimension), (-1, dimension), (-2, dimension)]
        );
    }
}
