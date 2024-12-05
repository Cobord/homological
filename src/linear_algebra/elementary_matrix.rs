use core::fmt::Debug;
use core::ops::{DivAssign, MulAssign};
use std::collections::VecDeque;

use super::matrix_store::BasisIndexing;
use crate::base_ring::field_generals::{Field, Ring};

#[derive(PartialEq, Eq)]
pub enum ElementaryMatrix<F: Ring> {
    /// swap the two rows assuming that they are both within bounds
    SwapRows(BasisIndexing, BasisIndexing),
    /// AddAssignRow(x,y) means add row x to row y
    AddAssignRow(BasisIndexing, BasisIndexing),
    /// AddAssignMultipleRow(x,f,y) add row x * f to row y
    AddAssignMultipleRow(BasisIndexing, F, BasisIndexing),
    /// ScaleRow(f, x) row x gets scaled by f
    // not checked by the type that the scaling factor is invertible, but it should be
    ScaleRow(BasisIndexing, F),
}

impl<F: Ring + Debug> Debug for ElementaryMatrix<F> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::SwapRows(arg0, arg1) => {
                f.debug_tuple("SwapRows").field(arg0).field(arg1).finish()
            }
            Self::AddAssignRow(arg0, arg1) => f
                .debug_tuple("AddAssignRow")
                .field(arg0)
                .field(arg1)
                .finish(),
            Self::AddAssignMultipleRow(arg0, arg1, arg2) => f
                .debug_tuple("AddAssignMultipleRow")
                .field(arg0)
                .field(arg1)
                .field(arg2)
                .finish(),
            Self::ScaleRow(arg0, arg1) => {
                f.debug_tuple("ScaleRow").field(arg0).field(arg1).finish()
            }
        }
    }
}

impl<F: Ring + Clone> Clone for ElementaryMatrix<F> {
    fn clone(&self) -> Self {
        match self {
            Self::SwapRows(arg0, arg1) => Self::SwapRows(*arg0, *arg1),
            Self::AddAssignRow(arg0, arg1) => Self::AddAssignRow(*arg0, *arg1),
            Self::ScaleRow(arg0, arg1) => Self::ScaleRow(*arg0, arg1.clone()),
            Self::AddAssignMultipleRow(arg0, arg1, arg2) => {
                Self::AddAssignMultipleRow(*arg0, arg1.clone(), *arg2)
            }
        }
    }
}

impl<F: Ring> ElementaryMatrix<F> {
    fn support(&self) -> Vec<usize> {
        #[allow(clippy::match_same_arms)]
        match self {
            ElementaryMatrix::SwapRows(arg0, arg1) => vec![*arg0, *arg1],
            ElementaryMatrix::AddAssignRow(arg0, arg1) => vec![*arg0, *arg1],
            ElementaryMatrix::AddAssignMultipleRow(arg0, _arg1, arg2) => vec![*arg0, *arg2],
            ElementaryMatrix::ScaleRow(arg0, _) => vec![*arg0],
        }
    }

    fn commutes(&self, other: &Self) -> bool {
        let support_self = self.support();
        let support_other = other.support();
        if support_self
            .iter()
            .any(|in_self| support_other.contains(in_self))
        {
            #[allow(clippy::match_same_arms)]
            match self {
                ElementaryMatrix::SwapRows(arg0, arg1) => {
                    return *other == ElementaryMatrix::SwapRows(*arg0, *arg1)
                        || *other == ElementaryMatrix::SwapRows(*arg1, *arg0);
                }
                ElementaryMatrix::AddAssignRow(arg0, arg1) => {
                    #[allow(clippy::match_same_arms)]
                    return match other {
                        ElementaryMatrix::SwapRows(_, _) => false,
                        ElementaryMatrix::AddAssignRow(arg2, _arg3) if arg2 == arg1 => false,
                        ElementaryMatrix::AddAssignRow(_arg2, arg3) if arg0 == arg3 => false,
                        ElementaryMatrix::AddAssignRow(arg2, _arg3) if arg2 == arg0 => true,
                        ElementaryMatrix::AddAssignRow(_arg2, arg3) if arg1 == arg3 => true,
                        ElementaryMatrix::AddAssignRow(_arg2, _arg3) => true,
                        ElementaryMatrix::AddAssignMultipleRow(arg2, _, _arg3) if arg2 == arg1 => {
                            false
                        }
                        ElementaryMatrix::AddAssignMultipleRow(_arg2, _, arg3) if arg0 == arg3 => {
                            false
                        }
                        ElementaryMatrix::AddAssignMultipleRow(arg2, _, _arg3) if arg2 == arg0 => {
                            true
                        }
                        ElementaryMatrix::AddAssignMultipleRow(_arg2, _, arg3) if arg1 == arg3 => {
                            true
                        }
                        ElementaryMatrix::AddAssignMultipleRow(_arg2, _, _arg3) => true,

                        ElementaryMatrix::ScaleRow(_, _) => false,
                    };
                }
                ElementaryMatrix::AddAssignMultipleRow(arg0, _, arg1) => {
                    #[allow(clippy::match_same_arms)]
                    return match other {
                        ElementaryMatrix::SwapRows(_, _) => false,
                        ElementaryMatrix::AddAssignRow(arg2, _arg3) if arg2 == arg1 => false,
                        ElementaryMatrix::AddAssignRow(_arg2, arg3) if arg0 == arg3 => false,
                        ElementaryMatrix::AddAssignRow(arg2, _arg3) if arg2 == arg0 => true,
                        ElementaryMatrix::AddAssignRow(_arg2, arg3) if arg1 == arg3 => true,
                        ElementaryMatrix::AddAssignRow(_arg2, _arg3) => true,
                        ElementaryMatrix::AddAssignMultipleRow(arg2, _, _arg3) if arg2 == arg1 => {
                            false
                        }
                        ElementaryMatrix::AddAssignMultipleRow(_arg2, _, arg3) if arg0 == arg3 => {
                            false
                        }
                        ElementaryMatrix::AddAssignMultipleRow(arg2, _, _arg3) if arg2 == arg0 => {
                            true
                        }
                        ElementaryMatrix::AddAssignMultipleRow(_arg2, _, arg3) if arg1 == arg3 => {
                            true
                        }
                        ElementaryMatrix::AddAssignMultipleRow(_arg2, _, _arg3) => true,

                        ElementaryMatrix::ScaleRow(_, _) => false,
                    };
                }
                ElementaryMatrix::ScaleRow(_, _) => {
                    #[allow(clippy::match_same_arms)]
                    return match other {
                        ElementaryMatrix::ScaleRow(_, _) => true,
                        ElementaryMatrix::SwapRows(_, _) => false,
                        ElementaryMatrix::AddAssignRow(_, _) => false,
                        ElementaryMatrix::AddAssignMultipleRow(_, _, _) => false,
                    };
                }
            }
        }
        true
    }

    fn transpose(self) -> Self {
        match self {
            ElementaryMatrix::SwapRows(idx, jdx) => ElementaryMatrix::SwapRows(idx, jdx),
            ElementaryMatrix::AddAssignRow(arg0, arg1) => {
                ElementaryMatrix::AddAssignRow(arg1, arg0)
            }
            ElementaryMatrix::AddAssignMultipleRow(arg0, arg1, arg2) => {
                ElementaryMatrix::AddAssignMultipleRow(arg2, arg1, arg0)
            }
            ElementaryMatrix::ScaleRow(arg0, arg1) => ElementaryMatrix::ScaleRow(arg0, arg1),
        }
    }

    fn try_inverse(self) -> Option<Vec<Self>> {
        match self {
            ElementaryMatrix::SwapRows(arg0, arg1) => {
                Some(vec![ElementaryMatrix::SwapRows(arg0, arg1)])
            }
            ElementaryMatrix::AddAssignRow(arg0, arg1) => {
                let neg_one = -F::ring_one();
                Some(vec![ElementaryMatrix::AddAssignMultipleRow(
                    arg0, neg_one, arg1,
                )])
            }
            ElementaryMatrix::AddAssignMultipleRow(arg0, arg1, arg2) => {
                Some(vec![ElementaryMatrix::AddAssignMultipleRow(
                    arg0, -arg1, arg2,
                )])
            }
            ElementaryMatrix::ScaleRow(arg0, arg1) => {
                let arg1_inverse = arg1.try_inverse()?;
                Some(vec![ElementaryMatrix::ScaleRow(arg0, arg1_inverse)])
            }
        }
    }
}

impl<F: Field> ElementaryMatrix<F> {
    fn inverse(self) -> Vec<Self> {
        self.try_inverse().unwrap()
    }
}

#[allow(clippy::module_name_repetitions)]
pub struct ElementaryMatrixProduct<F: Ring> {
    pub(crate) dimension: BasisIndexing,
    pub(crate) steps: VecDeque<ElementaryMatrix<F>>,
}

impl<F: Ring + Debug> Debug for ElementaryMatrixProduct<F> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ElementaryMatrixProduct")
            .field("dimension", &self.dimension)
            .field("steps", &self.steps)
            .finish()
    }
}

impl<F: Ring + Clone> Clone for ElementaryMatrixProduct<F> {
    fn clone(&self) -> Self {
        Self {
            steps: self.steps.clone(),
            dimension: self.dimension,
        }
    }
}

impl<F: Ring> From<(BasisIndexing, ElementaryMatrix<F>)> for ElementaryMatrixProduct<F> {
    fn from(value: (BasisIndexing, ElementaryMatrix<F>)) -> Self {
        Self {
            steps: vec![value.1].into(),
            dimension: value.0,
        }
    }
}

impl<F: Ring> MulAssign for ElementaryMatrixProduct<F> {
    fn mul_assign(&mut self, rhs: Self) {
        self.steps.extend(rhs.steps);
    }
}

impl<F: Ring> ElementaryMatrixProduct<F> {
    pub(crate) fn extract_common_prefix(
        &mut self,
        other: &mut Self,
        mut extracted_out_so_far: Self,
    ) -> Self {
        if self.steps.is_empty() || other.steps.is_empty() {
            return extracted_out_so_far;
        }
        let index_of_step0 = other.steps.iter().position(|step| *step == self.steps[0]);
        if let Some(found_in_other) = index_of_step0 {
            for position_before_it in (0..found_in_other - 1).rev() {
                if other.steps[position_before_it].commutes(&other.steps[position_before_it + 1]) {
                    other.steps.swap(position_before_it, position_before_it + 1);
                }
            }
            if self.steps[0] == other.steps[0] {
                let z = self.steps.pop_front().expect("Already checked emptiness");
                let w = other.steps.pop_front().expect("Already checked emptiness");
                assert!(z == w, "Already checked equality");
                extracted_out_so_far.steps.push_back(z);
                return self.extract_common_prefix(other, extracted_out_so_far);
            }
        }
        extracted_out_so_far
    }

    pub(crate) fn extract_common_suffix(
        &mut self,
        other: &mut Self,
        mut extracted_out_so_far: Self,
    ) -> Self {
        if self.steps.is_empty() || other.steps.is_empty() {
            return extracted_out_so_far;
        }
        let last_of_self = self.steps.iter().last().expect("Already checked emptiness");
        let index_of_step_last = other
            .steps
            .iter()
            .rev()
            .position(|step| *step == *last_of_self);
        if let Some(found_in_other) = index_of_step_last {
            let other_len = other.steps.len();
            for position_after_it in found_in_other + 1..other_len - 1 {
                if other.steps[position_after_it - 1].commutes(&other.steps[position_after_it]) {
                    other.steps.swap(position_after_it - 1, position_after_it);
                }
            }
            if *last_of_self
                == *other
                    .steps
                    .iter()
                    .last()
                    .expect("Already checked emptiness")
            {
                let z = self.steps.pop_back().expect("Already checked emptiness");
                let w = other.steps.pop_back().expect("Already checked emptiness");
                assert!(z == w, "Already checked equality");
                extracted_out_so_far.steps.push_front(z);
                return self.extract_common_prefix(other, extracted_out_so_far);
            }
        }
        extracted_out_so_far
    }

    /// the transpose of a product of elementary matrices is
    /// also a product of elementary matrices
    #[must_use]
    pub fn transpose(self) -> Self {
        #[allow(clippy::redundant_closure_for_method_calls)]
        let new_steps = self
            .steps
            .into_iter()
            .map(|step| step.transpose())
            .rev()
            .collect();
        Self {
            steps: new_steps,
            dimension: self.dimension,
        }
    }

    /// no elementary factors
    /// so this is for the identity matrix of `dimension`
    #[must_use]
    pub fn new(dimension: BasisIndexing) -> Self {
        Self {
            dimension,
            steps: VecDeque::new(),
        }
    }

    /// invert this product of elementary matrices
    /// there may be a scaling a row by 0 which causes
    /// noninvertibility
    #[must_use]
    pub fn try_inverse(self) -> Option<Self> {
        let mut new_steps = VecDeque::with_capacity(self.steps.len());
        for cur_step in self.steps.into_iter().rev() {
            let cur_step_inverse = cur_step.try_inverse()?;
            new_steps.extend(cur_step_inverse);
        }
        Some(Self {
            steps: new_steps,
            dimension: self.dimension,
        })
    }

    /// are there factors in the product
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.steps.is_empty()
    }
}

impl<F: Field> ElementaryMatrixProduct<F> {
    /// invert this product of elementary matrices
    /// use `try_inverse` if not sure that it is solely consistent
    /// of invertible factors
    /// # Panics
    /// if one of the factors was not invertible
    /// such as multiplying a row by a zero divisor
    #[must_use]
    pub fn inverse(self) -> Self {
        #[allow(clippy::redundant_closure_for_method_calls)]
        let new_steps = self
            .steps
            .into_iter()
            .rev()
            .flat_map(|step| step.inverse())
            .collect();
        Self {
            steps: new_steps,
            dimension: self.dimension,
        }
    }
}

impl<F: Field> DivAssign for ElementaryMatrixProduct<F> {
    fn div_assign(&mut self, rhs: Self) {
        #[allow(clippy::redundant_closure_for_method_calls)]
        let inverse_rhs = rhs.steps.into_iter().rev().flat_map(|z| z.inverse());
        self.steps.extend(inverse_rhs);
    }
}
