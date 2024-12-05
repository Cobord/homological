use core::ops::{Add, AddAssign, Div, Mul, Neg, Sub};
use num::{One, Zero};

use super::field_generals::{Commutative, Field, IntegerType, Ring};

#[derive(PartialEq, Eq, Debug, Clone, PartialOrd)]
#[repr(transparent)]
pub struct F2(pub(crate) bool);

impl Add for F2 {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        #[allow(clippy::suspicious_arithmetic_impl)]
        F2(self.0 ^ rhs.0)
    }
}

impl AddAssign for F2 {
    fn add_assign(&mut self, rhs: Self) {
        #[allow(clippy::suspicious_op_assign_impl)]
        if rhs.0 {
            self.0 ^= true;
        }
    }
}

impl Neg for F2 {
    type Output = Self;

    fn neg(self) -> Self::Output {
        self
    }
}
impl Sub for F2 {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        self.add(rhs)
    }
}
impl Mul for F2 {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        #[allow(clippy::suspicious_arithmetic_impl)]
        F2(self.0 & rhs.0)
    }
}
impl Commutative for F2 {}
impl Div for F2 {
    type Output = Self;

    fn div(self, rhs: Self) -> Self::Output {
        if rhs.0 {
            self
        } else {
            panic!("Division by 0")
        }
    }
}

impl From<IntegerType> for F2 {
    fn from(value: IntegerType) -> Self {
        Self(value % 2 == 0)
    }
}

impl Zero for F2 {
    fn zero() -> Self {
        0.into()
    }

    fn is_zero(&self) -> bool {
        !self.0
    }
}

impl One for F2 {
    fn one() -> Self {
        1.into()
    }
}

impl Ring for F2 {
    fn characteristic(_primes: Box<dyn Iterator<Item = IntegerType>>) -> IntegerType {
        2
    }

    fn try_inverse(self) -> Option<Self> {
        Some(self)
    }

    fn mul_assign_borrow(&mut self, other: &Self) {
        if !other.0 {
            self.0 = false;
        }
    }
}

impl Field for F2 {}
