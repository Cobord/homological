use core::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign};
use num::traits::One;

pub struct LazyLinear<N, T>
where
    N: Add<Output = N> + Neg<Output = N> + Mul<Output = N>,
{
    pub(crate) summands: Box<dyn Iterator<Item = (N, T)>>,
}

impl<N, T> LazyLinear<N, T>
where
    N: Add<Output = N> + Neg<Output = N> + Mul<Output = N> + 'static,
    T: 'static,
{
    #[must_use]
    pub fn new() -> Self {
        LazyLinear::<_, _> {
            summands: Box::new(vec![].into_iter()),
        }
    }

    pub fn map<T2>(self, f: impl Fn(T) -> T2 + 'static) -> LazyLinear<N, T2>
    where
        T2: 'static,
    {
        LazyLinear::<_, _> {
            summands: Box::new(self.summands.map(move |(z0, z1)| (z0, f(z1)))),
        }
    }

    pub fn change_coeffs<N2>(self, f: impl Fn(N) -> N2 + 'static) -> LazyLinear<N2, T>
    where
        N2: Add<Output = N2> + Neg<Output = N2> + Mul<Output = N2> + 'static,
    {
        LazyLinear::<_, _> {
            summands: Box::new(self.summands.map(move |(z0, z1)| (f(z0), z1))),
        }
    }
}

impl<N, T> Default for LazyLinear<N, T>
where
    N: Add<Output = N> + Neg<Output = N> + Mul<Output = N> + 'static,
    T: 'static,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<N, T> Add for LazyLinear<N, T>
where
    N: Add<Output = N> + Neg<Output = N> + Mul<Output = N> + 'static,
    T: 'static,
{
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self {
            summands: Box::new(self.summands.chain(rhs.summands)),
        }
    }
}

impl<N, T> AddAssign for LazyLinear<N, T>
where
    N: Add<Output = N> + Neg<Output = N> + Mul<Output = N> + 'static,
    T: 'static,
{
    fn add_assign(&mut self, rhs: Self) {
        let mut dummy_summands: Box<dyn Iterator<Item = (N, T)>> = Box::new([].into_iter());
        core::mem::swap(&mut dummy_summands, &mut self.summands);
        self.summands = Box::new(dummy_summands.chain(rhs.summands));
    }
}

impl<N, T> Neg for LazyLinear<N, T>
where
    N: Add<Output = N> + Neg<Output = N> + Mul<Output = N> + 'static,
    T: 'static,
{
    type Output = Self;

    fn neg(self) -> Self::Output {
        Self {
            summands: Box::new(self.summands.map(|(z0, z1)| (-z0, z1))),
        }
    }
}

impl<N, T> Sub for LazyLinear<N, T>
where
    N: Add<Output = N> + Neg<Output = N> + Mul<Output = N> + 'static,
    T: 'static,
{
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Self {
            summands: Box::new(self.summands.chain(rhs.summands.map(|(z0, z1)| (-z0, z1)))),
        }
    }
}

impl<N, T> SubAssign for LazyLinear<N, T>
where
    N: Add<Output = N> + Neg<Output = N> + Mul<Output = N> + 'static,
    T: 'static,
{
    fn sub_assign(&mut self, rhs: Self) {
        let mut dummy_summands: Box<dyn Iterator<Item = (N, T)>> = Box::new([].into_iter());
        core::mem::swap(&mut dummy_summands, &mut self.summands);
        self.summands = Box::new(dummy_summands.chain(rhs.summands.map(|(z0, z1)| (-z0, z1))));
    }
}

impl<N, T> Mul<N> for LazyLinear<N, T>
where
    N: Add<Output = N> + Neg<Output = N> + Mul<Output = N> + Clone + 'static,
    T: 'static,
{
    type Output = Self;

    fn mul(self, rhs: N) -> Self::Output {
        Self {
            summands: Box::new(self.summands.map(move |(z0, z1)| (z0 * rhs.clone(), z1))),
        }
    }
}

impl<N, T> MulAssign<N> for LazyLinear<N, T>
where
    N: Add<Output = N> + Neg<Output = N> + Mul<Output = N> + Clone + 'static,
    T: 'static,
{
    fn mul_assign(&mut self, rhs: N) {
        let mut dummy_summands: Box<dyn Iterator<Item = (N, T)>> = Box::new([].into_iter());
        core::mem::swap(&mut dummy_summands, &mut self.summands);
        self.summands = Box::new(dummy_summands.map(move |(z0, z1)| (z0 * rhs.clone(), z1)));
    }
}

impl<N, T> From<(N, T)> for LazyLinear<N, T>
where
    N: Add<Output = N> + Neg<Output = N> + Mul<Output = N> + 'static,
    T: 'static,
{
    fn from(value: (N, T)) -> Self {
        Self {
            summands: Box::new([value].into_iter()),
        }
    }
}

impl<N, T> From<T> for LazyLinear<N, T>
where
    N: Add<Output = N> + Neg<Output = N> + Mul<Output = N> + 'static + One,
    T: 'static,
{
    fn from(value: T) -> Self {
        (N::one(), value).into()
    }
}

pub trait TermMultiplier<N, T2 = Self>
where
    N: Add<Output = N> + Neg<Output = N> + Mul<Output = N> + 'static,
    Self: Sized,
{
    fn two_summand_mul(self, rhs: T2) -> LazyLinear<N, Self>;
}

impl<N, T, T2> Mul<LazyLinear<N, T2>> for LazyLinear<N, T>
where
    N: Add<Output = N> + Neg<Output = N> + Mul<Output = N> + Clone + 'static,
    T: 'static + TermMultiplier<N, T2> + Clone,
    T2: 'static + TermMultiplier<N, T2> + Clone,
{
    type Output = Self;

    fn mul(self, rhs: LazyLinear<N, T2>) -> Self::Output {
        let rhs_materialized = rhs.summands.collect::<Vec<_>>();
        Self {
            summands: Box::new(self.summands.flat_map(move |self_summand| {
                rhs_materialized
                    .clone()
                    .into_iter()
                    .flat_map(move |rhs_summand| {
                        let coeff = self_summand.0.clone() * rhs_summand.0;
                        let pieces = self_summand.1.clone().two_summand_mul(rhs_summand.1);
                        pieces
                            .summands
                            .map(move |piece| (piece.0 * coeff.clone(), piece.1))
                    })
            })),
        }
    }
}
