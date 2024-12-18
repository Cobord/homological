use core::marker::PhantomData;
use std::collections::VecDeque;

use crate::base_ring::field_generals::Ring;
use crate::linear_algebra::matrix_store::{BasisIndexing, EffortfulMatrixStore, LeftMultipliesBy};

mod private {
    pub trait Sealed {}
}

pub trait HomSigns: private::Sealed {
    fn differential_increases() -> bool;
}

pub struct HomologicalIndex {}
impl private::Sealed for HomologicalIndex {}
impl HomSigns for HomologicalIndex {
    fn differential_increases() -> bool {
        false
    }
}

pub struct CohomologicalIndex {}
impl private::Sealed for CohomologicalIndex {}
impl HomSigns for CohomologicalIndex {
    fn differential_increases() -> bool {
        true
    }
}

pub(crate) type HomologicalIndexing = i64;

pub struct ChainFVect<R: HomSigns, F: Ring + Clone, M: EffortfulMatrixStore<F>> {
    homological_index: HomologicalIndexing,
    dimension: BasisIndexing,
    differential: M,
    homsign: PhantomData<R>,
    ring_info: PhantomData<F>,
    rest: Option<Box<ChainFVect<R, F, M>>>,
}

impl<R, F, M> ChainFVect<R, F, M>
where
    R: HomSigns,
    F: Ring + Clone,
    M: EffortfulMatrixStore<F>,
{
    /// a single vector space of `dimension` concentrated in degree 0
    #[must_use]
    pub fn concentrated_in_0(dimension: BasisIndexing) -> Self {
        Self {
            homological_index: 0,
            dimension,
            differential: M::zero_matrix(0, dimension),
            ring_info: PhantomData,
            homsign: PhantomData,
            rest: None,
        }
    }

    /// what are the dimensions of the vector spaces at each `HomologicalIndexing`
    pub fn dimensions_each_index(&self) -> Vec<(HomologicalIndexing, BasisIndexing)> {
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

    /// what are the dimensions of the vector spaces at `which_index` specifically
    pub fn dimensions_of_that(&self, which_index: HomologicalIndexing) -> BasisIndexing {
        if self.homological_index == which_index {
            return self.dimension;
        }
        if R::differential_increases() && which_index < self.homological_index {
            return 0;
        }
        if !R::differential_increases() && which_index > self.homological_index {
            return 0;
        }
        if let Some(rest) = &self.rest {
            rest.dimensions_of_that(which_index)
        } else {
            0
        }
    }

    /// all the betti numbers
    pub fn bettis_each_index(&self) -> Vec<(HomologicalIndexing, BasisIndexing)> {
        self.bettis_each_index_helper(None)
    }

    fn bettis_each_index_helper(
        &self,
        incoming_differential: Option<&M>,
    ) -> Vec<(HomologicalIndexing, BasisIndexing)> {
        debug_assert!(self.dimension == self.differential.dimensions().1);
        let mut return_val = if let Some(real_rest) = self.rest.as_ref() {
            debug_assert!(real_rest.dimension == self.differential.dimensions().0);
            real_rest.bettis_each_index_helper(Some(&self.differential))
        } else {
            debug_assert!(0 == self.differential.dimensions().0);
            vec![]
        };
        let my_betti = if let Some(real_incoming) = incoming_differential {
            self.differential
                .homology_info(real_incoming, true, false)
                .expect("d^2 = 0 by construction")
                .0
        } else {
            self.differential.kernel()
        };
        return_val.push((self.homological_index, my_betti));
        return_val
    }

    /// shift the complex by `how_much`
    /// depending on the sign of this and the marker R of `HomSigns`
    /// this may be in the direction of the differential or not
    pub fn shift_all(&mut self, how_much: HomologicalIndexing) {
        self.homological_index += how_much;
        if let Some(real_rest) = self.rest.as_deref_mut() {
            real_rest.shift_all(how_much);
        }
    }

    /// put more zero vector spaces before the first nonzero differential
    /// (not trivially 0, it might be 0 but not obviously)
    /// puts them opposite the direction of the differential
    pub fn prepend_zero_spaces(&mut self, how_many: HomologicalIndexing) {
        if how_many <= 0 {
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
        core::mem::swap(&mut fake_self, self);
        *self = Self {
            homological_index: new_index,
            dimension: 0,
            differential: M::zero_matrix(next_dimension, 0),
            ring_info: PhantomData,
            homsign: PhantomData,
            rest: Some(Box::new(fake_self)),
        };
    }

    /// get these two to line up so they start with the same `HomologicalIndexing`
    pub fn align_together(&mut self, other: &mut Self) {
        let self_index = self.homological_index;
        let other_index = other.homological_index;
        if self_index == other_index {
            return;
        }
        #[allow(clippy::cast_possible_wrap)]
        let to_prepend = other_index.abs_diff(self_index) as HomologicalIndexing;
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
        debug_assert_eq!(self_index, other_index);
    }

    /// like `prepend_zero_space` put a vector space before the first differential
    /// (one that is not implicitly 0)
    /// but here we are putting a nontrivial vector space and a differential to the next space
    /// # Errors
    /// Err(false) if the dimension of the space being appended/prepended
    ///     did not match the dimensions of the `new_differential` matrix
    /// Err(true) if `d^2 \neq 0`
    pub fn prepend_space(
        &mut self,
        new_dimension: BasisIndexing,
        new_differential: M,
    ) -> Result<(), bool> {
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
        core::mem::swap(&mut fake_self, self);
        *self = Self {
            homological_index: new_index,
            dimension: new_dimension,
            differential: new_differential,
            ring_info: PhantomData,
            homsign: PhantomData,
            rest: Some(Box::new(fake_self)),
        };
        Ok(())
    }

    pub(crate) fn apply_all_differentials(
        &self,
        starting_index: HomologicalIndexing,
        vectors: &mut VecDeque<M::ColumnVector>,
    ) {
        if starting_index == self.homological_index {
            if let Some(v0) = vectors.front_mut() {
                v0.left_multiply(&self.differential);
                let put_back = vectors.pop_front().expect("Already know nonempty");
                let next_index =
                    self.homological_index + if R::differential_increases() { 1 } else { -1 };
                if let Some(real_rest) = &self.rest {
                    real_rest.apply_all_differentials(next_index, vectors);
                } else {
                    for v in vectors.iter_mut() {
                        v.zero_out(false);
                    }
                }
                vectors.push_front(put_back);
            }
            return;
        }
        let vectors_deeper_in = if R::differential_increases() {
            starting_index > self.homological_index
        } else {
            starting_index < self.homological_index
        };
        if vectors_deeper_in {
            if let Some(real_rest) = &self.rest {
                real_rest.apply_all_differentials(starting_index, vectors);
            } else {
                for v in vectors {
                    v.zero_out(false);
                }
            }
            return;
        }
        if !vectors_deeper_in {
            let next_is_start = if R::differential_increases() {
                self.homological_index - starting_index == 1
            } else {
                starting_index - self.homological_index == 1
            };
            if let Some(v0) = vectors.front_mut() {
                v0.zero_out(false);
                if next_is_start {
                    v0.zero_pad(self.dimension);
                }
                let put_back = vectors.pop_front().expect("Already know nonempty");
                let next_index = starting_index + if R::differential_increases() { 1 } else { -1 };
                self.apply_all_differentials(next_index, vectors);
                vectors.push_front(put_back);
            }
        }
    }
}

impl<F, M> ChainFVect<CohomologicalIndex, F, M>
where
    F: Ring + Clone,
    M: EffortfulMatrixStore<F>,
{
    pub fn negate_homological_indices(self) -> ChainFVect<HomologicalIndex, F, M> {
        ChainFVect::<HomologicalIndex, F, M> {
            homological_index: -self.homological_index,
            dimension: self.dimension,
            differential: self.differential,
            ring_info: PhantomData,
            homsign: PhantomData,
            rest: self
                .rest
                .map(|real_rest| Box::new(real_rest.negate_homological_indices())),
        }
    }
}

impl<F, M> ChainFVect<HomologicalIndex, F, M>
where
    F: Ring + Clone,
    M: EffortfulMatrixStore<F>,
{
    pub fn negate_homological_indices(self) -> ChainFVect<CohomologicalIndex, F, M> {
        ChainFVect::<CohomologicalIndex, F, M> {
            homological_index: -self.homological_index,
            dimension: self.dimension,
            differential: self.differential,
            homsign: PhantomData,
            ring_info: PhantomData,
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
        use crate::base_ring::f2::F2;
        use crate::linear_algebra::{f2_vect::F2Matrix, matrix_store::MatrixStore};
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
