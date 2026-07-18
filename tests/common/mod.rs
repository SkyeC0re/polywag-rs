use approx::{AbsDiffEq, RelativeEq, relative_eq};
use bytemuck::Zeroable;
use core::fmt::Debug;
use core::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};
use core::ops::{Deref, DerefMut};
use core::{num::NonZeroUsize, slice};
use f256::f256;
use polywag::simd::{SimdAble, SimdField};
use std::fmt::Display;

pub trait TestableSimd:
    SimdAble + AbsDiffEq<Epsilon = Self> + RelativeEq<Epsilon = Self> + Into<F256>
{
}

impl<T> TestableSimd for T where
    T: SimdAble + AbsDiffEq<Epsilon = Self> + RelativeEq<Epsilon = Self> + Into<F256>
{
}

pub const TEST_EPS_PERCENTAGE: usize = 80;

/// Get the testing epsilon defined by `TEST_EPS_PERCENTAGE`. Specifically, given the test epsilon ratio $r$
/// and `T`'s machine epsilon $\epsilon < 1$, we define the test epsilon as:
/// $$
///     \epsilon_t = e^{r \ln(\epsilon)}
/// $$
pub fn test_eps<T: TestableSimd>() -> T {
    let ratio = T::from_usize(TEST_EPS_PERCENTAGE) / T::from_usize(100);

    T::exp(ratio * T::SF_EPS.ln())
}

pub fn eps_diff_eq<T: TestableSimd>(a: T, b: T, eps: T) -> bool {
    relative_eq!(a, b, epsilon = eps, max_relative = eps)
}

#[track_caller]
pub fn assert_eps_diff_eq<T: TestableSimd>(a: T, b: T, eps: T) {
    assert!(eps_diff_eq(a, b, eps), "{:?} != {:?}", a, b);
}

#[macro_export]
macro_rules! test_all_types {
    ($f:ident) => {
        paste! {
            #[test]
            fn [<f64_ $f>]() {
                $f::<f64>()
            }

            #[test]
            fn [<f32_ $f>]() {
                $f::<f32>()
            }
        }
    };
}

#[derive(Clone, Copy, PartialEq, PartialOrd)]
#[repr(transparent)]
pub struct F256(f256);

impl Add for F256 {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self(self.0 + rhs.0)
    }
}

impl Sub for F256 {
    type Output = Self;

    #[inline(always)]
    fn sub(self, rhs: Self) -> Self::Output {
        Self(self.0 - rhs.0)
    }
}

impl Mul for F256 {
    type Output = Self;

    #[inline(always)]
    fn mul(self, rhs: Self) -> Self::Output {
        Self(self.0 * rhs.0)
    }
}

impl Div for F256 {
    type Output = Self;

    #[inline(always)]
    fn div(self, rhs: Self) -> Self::Output {
        Self(self.0 / rhs.0)
    }
}

impl AddAssign for F256 {
    #[inline(always)]
    fn add_assign(&mut self, rhs: Self) {
        self.0 += rhs.0
    }
}

impl SubAssign for F256 {
    #[inline(always)]
    fn sub_assign(&mut self, rhs: Self) {
        self.0 -= rhs.0
    }
}

impl MulAssign for F256 {
    #[inline(always)]
    fn mul_assign(&mut self, rhs: Self) {
        self.0 *= rhs.0
    }
}

impl DivAssign for F256 {
    #[inline(always)]
    fn div_assign(&mut self, rhs: Self) {
        self.0 /= rhs.0
    }
}

impl Neg for F256 {
    type Output = Self;
    #[inline(always)]
    fn neg(self) -> Self::Output {
        Self(self.0.neg())
    }
}

impl Deref for F256 {
    type Target = f256;

    #[inline(always)]
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for F256 {
    #[inline(always)]
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl From<f32> for F256 {
    fn from(value: f32) -> Self {
        Self(f256::from(value))
    }
}

impl From<f64> for F256 {
    fn from(value: f64) -> Self {
        Self(f256::from(value))
    }
}

impl Debug for F256 {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        Display::fmt(&self.0, f)
    }
}

impl SimdField for F256 {
    const SF_ZERO: Self = Self(f256::ZERO);
    const SF_ONE: Self = Self(f256::ONE);
    const SF_EPS: Self = Self(f256::EPSILON);
    const SF_LANES: NonZeroUsize = NonZeroUsize::new(1).unwrap();

    type Element = Self;

    #[inline(always)]
    fn mul_add(self, m: Self, a: Self) -> Self {
        Self(self.0.mul_add(m.0, a.0))
    }

    #[inline(always)]
    fn splat(v: Self::Element) -> Self {
        v
    }

    #[inline(always)]
    fn as_slice(&self) -> &[Self] {
        unsafe { slice::from_raw_parts(self, 1) }
    }

    #[inline(always)]
    fn as_mut_slice(&mut self) -> &mut [Self] {
        unsafe { slice::from_raw_parts_mut(self, 1) }
    }

    #[inline(always)]
    fn reduce_add(self) -> Self::Element {
        self
    }

    #[inline(always)]
    fn abs(self) -> Self {
        Self((&self.0).abs())
    }

    #[inline(always)]
    fn max(self, other: Self) -> Self {
        Self(self.0.max(other.0))
    }

    #[inline(always)]
    fn exp(self) -> Self {
        Self((&self.0).exp())
    }

    #[inline(always)]
    fn ln(self) -> Self {
        Self((&self.0).ln())
    }

    #[inline(always)]
    fn recip(self) -> Self {
        Self((&self.0).recip())
    }
}

// The implementation does in fact store zero as the zero bit pattern.
unsafe impl Zeroable for F256 {}

impl SimdAble for F256 {
    type SimdT = Self;

    #[inline(always)]
    fn is_finite(&self) -> bool {
        self.0.is_finite()
    }

    fn from_usize(v: usize) -> Self {
        #[cfg(target_pointer_width = "16")]
        return Self(f256::from(v as u16));

        #[cfg(target_pointer_width = "32")]
        return Self(f256::from(v as u32));

        #[cfg(target_pointer_width = "64")]
        return Self(f256::from(v as u64));
    }
}
