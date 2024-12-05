use core::ops::{Add, AddAssign, Div, Mul, Neg, Sub};

use crate::linear_comb::Commutative;

pub type IntegerType = i16;

pub trait Ring:
    Add<Output = Self>
    + AddAssign<Self>
    + Sub<Output = Self>
    + Neg<Output = Self>
    + Mul<Output = Self>
    + PartialEq
    + From<IntegerType>
    + Sized
{
    #[must_use]
    fn characteristic(primes: Box<dyn Iterator<Item = IntegerType>>) -> IntegerType {
        let zero_f = Self::from(0);
        for i in primes {
            if Self::from(i) == zero_f {
                return i;
            }
        }
        0
    }

    #[must_use]
    fn ring_one() -> Self {
        1.into()
    }

    fn try_inverse(self) -> Option<Self> {
        None
    }

    fn mul_assign_borrow(&mut self, other: &Self);
}

pub trait Field: Ring + Div<Output = Self> + Commutative {
    fn try_inverse(self) -> Option<Self> {
        let zero_f = Self::from(0);
        if self == zero_f {
            None
        } else {
            Some(Self::ring_one() / self)
        }
    }
}
