use core::ops::{Add, Div, Mul, Neg, Sub};

use crate::linear_comb::Commutative;

pub trait Ring:
    Add<Output = Self>
    + Sub<Output = Self>
    + Neg<Output = Self>
    + Mul<Output = Self>
    + Eq
    + From<usize>
    + Sized
{
    #[allow(dead_code)]
    fn characteristic(primes: Box<dyn Iterator<Item = usize>>) -> usize {
        let zero_f = Self::from(0);
        for i in primes {
            if Self::from(i) == zero_f {
                return i;
            }
        }
        0
    }

    fn one() -> Self {
        1.into()
    }

    fn try_inverse(self) -> Option<Self> {
        None
    }
}

pub trait Field: Ring + Div<Output = Self> + Commutative {
    #[allow(dead_code)]
    fn try_inverse(self) -> Option<Self> {
        let zero_f = Self::from(0);
        if self == zero_f {
            None
        } else {
            Some(Self::one() / self)
        }
    }
}
