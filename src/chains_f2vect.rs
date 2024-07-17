use std::marker::PhantomData;

use bitvector::BitVector;

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

#[allow(dead_code)]
pub struct F2Matrix {
    my_rows: Vec<BitVector>,
    num_cols: usize,
}

impl F2Matrix {
    pub fn new(num_rows: usize, num_cols: usize) -> Self {
        Self {
            my_rows: vec![BitVector::new(num_cols);num_rows],
            num_cols,
        }
    }

    pub fn num_rows(&self) -> usize {
        self.my_rows.len()
    }

    pub fn composed_eq_zero(&self, other: &Self) -> bool {
        for i in 0..other.num_cols {
            let mut after_other = BitVector::new(other.num_rows());
            for j in 0..other.num_rows() {
                if other.my_rows[j].contains(i) {
                    after_other.insert(j);
                }
            }
            for k in 0..self.num_rows() {
                let composed_entry_k = self.my_rows[k].intersection(&after_other);
                if composed_entry_k.len() % 2 != 0 {
                    return false;
                }
            }
        }
        true
    }

    #[allow(dead_code)]
    pub fn composed(&self,_other: &Self) -> Self {
        todo!()
    }
}

pub struct ChainF2Vect<R : HomSigns> {
    homological_index: i64,
    dimension: usize,
    differential: F2Matrix,
    homsign: PhantomData<R>,
    rest: Option<Box<ChainF2Vect<R>>>
}

impl<R> ChainF2Vect<R>
where
    R: HomSigns,
{
    pub fn concentrated_in_0(dimension: usize) -> Self {
        Self {
            homological_index: 0,
            dimension,
            differential: F2Matrix::new(0,dimension),
            homsign: PhantomData,
            rest: None
        }
    }

    #[allow(dead_code)]
    pub fn shift_all(&mut self,how_much: i64) {
        self.homological_index += how_much;
        if let Some(real_rest) = self.rest.as_deref_mut() {
            real_rest.shift_all(how_much);
        }
    }

    pub fn prepend_zero_spaces(&mut self,how_many: usize) {
        if how_many == 0 {
            return;
        }
        self.prepend_zero_space();
        self.prepend_zero_spaces(how_many-1);
    }

    fn prepend_zero_space(&mut self) {
        let new_index = if R::differential_increases() {self.homological_index-1} else {self.homological_index+1};
        let next_dimension = self.dimension;
        let mut fake_self = Self::concentrated_in_0(0);
        std::mem::swap(&mut fake_self, self);
        *self = Self {
            homological_index: new_index,
            dimension: 0,
            differential: F2Matrix::new(next_dimension,0),
            homsign: PhantomData,
            rest: Some(Box::new(fake_self))
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
        assert_eq!(self_index,other_index);
    }

    #[allow(dead_code)]
    pub fn prepend_space(&mut self, new_dimension: usize, new_differential: F2Matrix) {
        let new_index = if R::differential_increases() {self.homological_index-1} else {self.homological_index+1};
        let next_dimension = self.dimension;
        let should_be_next_dimension = new_differential.num_rows();
        assert_eq!(next_dimension,should_be_next_dimension);
        if let Some(next_differential) = self.rest.as_ref().map(|real_rest| &real_rest.differential) {
            assert!(F2Matrix::composed_eq_zero(next_differential,&new_differential));
        }
        let mut fake_self = Self::concentrated_in_0(0);
        std::mem::swap(&mut fake_self, self);
        *self = Self {
            homological_index: new_index,
            dimension: new_dimension,
            differential: new_differential,
            homsign: PhantomData,
            rest: Some(Box::new(fake_self))
        };
    }
}