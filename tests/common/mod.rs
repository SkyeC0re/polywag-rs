use approx::AbsDiffEq;
use polywag::simd::SimdAble;

pub trait TestableSimd: SimdAble + AbsDiffEq<Epsilon = Self> {}
impl<T> TestableSimd for T where T: SimdAble + AbsDiffEq<Epsilon = Self> {}
