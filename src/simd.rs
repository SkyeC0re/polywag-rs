use cfg_if::cfg_if;
use core::{
    fmt::Debug,
    num::NonZeroUsize,
    ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign},
    slice,
};

use pastey::paste;

macro_rules! wide_float_simd_field_impl {
   ( $($name:ty : $elm:ty : $lanes:expr),* ) => {
        $(paste! {
        impl SimdField for $name {
            const ZERO: Self = Self::ZERO;
            const ONE: Self = Self::ONE;
            const EPS: Self =  Self::splat(Self::Element::EPSILON);
            const LANES: NonZeroUsize = NonZeroUsize::new($lanes).unwrap();
            type Element = $elm;

            #[inline(always)]
            fn mul_add(self, m: Self, a: Self) -> Self {
               self.mul_add(m, a)
            }

            #[inline(always)]
            fn splat(v: Self::Element) -> Self {
                Self::splat(v)
            }

            #[inline(always)]
            fn as_slice(&self) -> &[$elm] {
                self.as_array()
            }

            #[inline(always)]
            fn as_mut_slice(&mut self) -> &mut [$elm] {
                self.as_mut_array()
            }

            #[inline(always)]
            fn reduce_add(self) -> Self::Element {
               self.reduce_add()
            }

            #[inline(always)]
            fn abs(self) -> Self {
                self.abs()
            }

            #[inline(always)]
            fn max(self, other: Self) -> Self {
                self.max(other)
            }

            #[inline(always)]
            fn exp(self) -> Self {
                self.exp()
            }

            #[inline(always)]
            fn ln(self) -> Self {
                self.ln()
            }
        }
        })*
    };
}

cfg_if! {
    if  #[cfg(target_feature="avx")] {
        pub type MaxSimdf32 = wide::f32x8;
        wide_float_simd_field_impl![
            MaxSimdf32 : f32 : 8
        ];
    } else if #[cfg(any(target_feature="sse", target_feature="simd128", all(target_feature="neon",target_arch="aarch64")))] {
        pub type MaxSimdf32 = wide::f32x4;
        wide_float_simd_field_impl![
            MaxSimdf32 : f32 : 4
        ];
    } else {
        pub type MaxSimdf32 = f32;
    }
}

cfg_if! {
    if  #[cfg(target_feature="avx512f")] {
        pub type MaxSimdf64 = wide::f64x8;
        wide_float_simd_field_impl![
            MaxSimdf64 : f64 : 8
        ];
    } else if  #[cfg(target_feature="avx")] {
        pub type MaxSimdf64 = wide::f64x4;
        wide_float_simd_field_impl![
            MaxSimdf64 : f64 : 4
        ];
    } else if #[cfg(any(target_feature="sse", target_feature="simd128", all(target_feature="neon",target_arch="aarch64")))] {
        pub type MaxSimdf64 = wide::f64x2;
        wide_float_simd_field_impl![
            MaxSimdf64 : f64 : 2
        ];
    } else {
        pub type MaxSimdf64 = f64;
    }
}

pub trait SimdField:
    Copy
    + Add<Output = Self>
    + AddAssign
    + Sub<Output = Self>
    + SubAssign
    + Mul<Output = Self>
    + MulAssign
    + Div<Output = Self>
    + DivAssign
    + Neg<Output = Self>
    + PartialEq
{
    const ZERO: Self;
    const ONE: Self;
    const EPS: Self;
    const LANES: NonZeroUsize;

    type Element: SimdField;

    fn splat(v: Self::Element) -> Self;

    fn as_slice(&self) -> &[Self::Element];

    fn as_mut_slice(&mut self) -> &mut [Self::Element];

    fn mul_add(self, m: Self, a: Self) -> Self;

    fn reduce_add(self) -> Self::Element;

    fn abs(self) -> Self;

    fn max(self, other: Self) -> Self;

    fn exp(self) -> Self;

    fn ln(self) -> Self;
}

macro_rules! primitive_float_simd_field_impl {
   ( $($t:ty),* ) => {
        $(

        impl SimdField for $t {
            const ZERO: Self = 0.0;
            const ONE: Self = 1.0;
            const EPS: Self = Self::EPSILON;
            const LANES: NonZeroUsize = NonZeroUsize::new(1).unwrap();
            type Element = Self;

            #[inline(always)]
            fn mul_add(self, m: Self, a: Self) -> Self {
                self.mul_add(m, a)
            }

            #[inline(always)]
            fn splat(v: Self::Element) -> Self {
                v
            }

            #[inline(always)]
            fn as_slice(&self) -> &[Self] {
               unsafe {slice::from_raw_parts(self, 1)}
            }

            #[inline(always)]
            fn as_mut_slice(&mut self) -> &mut [Self] {
                unsafe {slice::from_raw_parts_mut(self, 1)}
            }

            #[inline(always)]
            fn reduce_add(self) -> Self::Element {
               self
            }

            #[inline(always)]
            fn abs(self) -> Self {
                self.abs()
            }

            #[inline(always)]
            fn max(self, other: Self) -> Self {
                self.max(other)
            }

            #[inline(always)]
            fn exp(self) -> Self {
                self.exp()
            }

            #[inline(always)]
            fn ln(self) -> Self {
                self.ln()
            }
        }
        )*
    };
}

primitive_float_simd_field_impl![f64, f32];

pub trait SimdAble: SimdField<Element = Self> + PartialOrd + Debug {
    type SimdT: SimdField<Element = Self>;

    fn is_finite(self) -> bool;

    fn from_usize(v: usize) -> Self;
}

macro_rules! wide_simd_able_impl {
     ( $( $elm:ty : $simdt:ty),* ) => {
        $(
        impl SimdAble for $elm {
            type SimdT = $simdt;

            #[inline(always)]
            fn is_finite(self) -> bool {
                self.is_finite()
            }

            #[inline(always)]
            fn from_usize(v: usize) -> Self {
                v as _
            }
        }
        )*
    };
}

wide_simd_able_impl![
    f32 : MaxSimdf32,
    f64 : MaxSimdf64
];
