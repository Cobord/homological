use crate::{
    field_generals::{Field, Ring},
    matrix_store::BasisIndexing,
};
use core::ops::{DivAssign, MulAssign};
use std::collections::VecDeque;

#[derive(PartialEq, Eq)]
pub(crate) enum ElementaryMatrix<F: Ring> {
    SwapRows(BasisIndexing, BasisIndexing),
    AddAssignRow(BasisIndexing, BasisIndexing), // AddAssignRow(x,y) add row x to row y
    ScaleRow(BasisIndexing, F), // not checked by the type that the scaling factor is invertible, but it should be
}

impl<F: Ring + Clone> Clone for ElementaryMatrix<F> {
    fn clone(&self) -> Self {
        match self {
            Self::SwapRows(arg0, arg1) => Self::SwapRows(*arg0, *arg1),
            Self::AddAssignRow(arg0, arg1) => Self::AddAssignRow(*arg0, *arg1),
            Self::ScaleRow(arg0, arg1) => Self::ScaleRow(*arg0, arg1.clone()),
        }
    }
}

impl<F: Ring> ElementaryMatrix<F> {
    fn commutes(&self, _other: &Self) -> bool {
        todo!()
    }

    fn transpose(self) -> Self {
        match self {
            ElementaryMatrix::SwapRows(idx, jdx) => ElementaryMatrix::SwapRows(idx, jdx),
            ElementaryMatrix::AddAssignRow(arg0, arg1) => {
                ElementaryMatrix::AddAssignRow(arg1, arg0)
            }
            ElementaryMatrix::ScaleRow(arg0, arg1) => ElementaryMatrix::ScaleRow(arg0, arg1),
        }
    }

    fn try_inverse(self) -> Option<Vec<Self>> {
        match self {
            ElementaryMatrix::SwapRows(_, _) => todo!(),
            ElementaryMatrix::AddAssignRow(_, _) => todo!(),
            ElementaryMatrix::ScaleRow(arg0, arg1) => {
                let arg1_inverse = arg1.try_inverse()?;
                Some(vec![ElementaryMatrix::ScaleRow(arg0, arg1_inverse)])
            }
        }
    }
}

impl<F: Field> ElementaryMatrix<F> {
    fn inverse(self) -> Vec<Self> {
        match self {
            ElementaryMatrix::SwapRows(_, _) => todo!(),
            ElementaryMatrix::AddAssignRow(_, _) => todo!(),
            ElementaryMatrix::ScaleRow(arg0, arg1) => {
                let arg1_inverse = F::one() / arg1;
                vec![ElementaryMatrix::ScaleRow(arg0, arg1_inverse)]
            }
        }
    }
}

pub struct ElementaryMatrixProduct<F: Ring> {
    pub(crate) dimension: BasisIndexing,
    pub(crate) steps: VecDeque<ElementaryMatrix<F>>,
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
                assert!(z == w);
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
                assert!(z == w);
                extracted_out_so_far.steps.push_front(z);
                return self.extract_common_prefix(other, extracted_out_so_far);
            }
        }
        extracted_out_so_far
    }

    pub(crate) fn transpose(self) -> Self {
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

    pub fn new(dimension: BasisIndexing) -> Self {
        Self {
            dimension,
            steps: VecDeque::new(),
        }
    }

    pub(crate) fn try_inverse(self) -> Option<Self> {
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
}

impl<F: Field> ElementaryMatrixProduct<F> {
    #[allow(dead_code)]
    pub(crate) fn inverse(self) -> Self {
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

impl<F: Field + Clone> DivAssign for ElementaryMatrixProduct<F> {
    fn div_assign(&mut self, rhs: Self) {
        let inverse_rhs = rhs.steps.into_iter().flat_map(|z| z.inverse()).rev();
        self.steps.extend(inverse_rhs);
    }
}
