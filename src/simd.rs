use core::{
    num::NonZeroUsize,
    ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign},
    slice,
};
use std::fmt::Debug;
use wide::{f32x8, f64x8};

use pastey::paste;

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
    const LANES: NonZeroUsize;

    type Element: SimdField;

    fn splat(v: Self::Element) -> Self;

    fn as_slice(&self) -> &[Self::Element];

    fn as_mut_slice(&mut self) -> &mut [Self::Element];

    fn mul_add(self, m: Self, a: Self) -> Self;

    fn reduce_add(self) -> Self::Element;

    fn abs(self) -> Self;

    fn max(self, other: Self) -> Self;
}

macro_rules! forward_basic_ops_impl {
    ($t:ty) => {
        impl Add for $t {
            type Output = Self;

            #[inline(always)]
            fn add(self, other: Self) -> Self {
                Self(self.0 + other.0)
            }
        }

        impl AddAssign for $t {
            #[inline(always)]
            fn add_assign(&mut self, other: Self) {
                self.0 += other.0
            }
        }

        impl Sub for $t {
            type Output = Self;

            #[inline(always)]
            fn sub(self, other: Self) -> Self {
                Self(self.0 - other.0)
            }
        }

        impl SubAssign for $t {
            #[inline(always)]
            fn sub_assign(&mut self, other: Self) {
                self.0 -= other.0
            }
        }

        impl Mul for $t {
            type Output = Self;

            #[inline(always)]
            fn mul(self, other: Self) -> Self {
                Self(self.0 * other.0)
            }
        }

        impl MulAssign for $t {
            #[inline(always)]
            fn mul_assign(&mut self, other: Self) {
                self.0 *= other.0
            }
        }

        impl Div for $t {
            type Output = Self;

            #[inline(always)]
            fn div(self, other: Self) -> Self {
                Self(self.0 / other.0)
            }
        }

        impl DivAssign for $t {
            #[inline(always)]
            fn div_assign(&mut self, other: Self) {
                self.0 /= other.0
            }
        }

        impl Neg for $t {
            type Output = Self;

            #[inline(always)]
            fn neg(self) -> Self {
                Self(-self.0)
            }
        }

        impl PartialEq for $t {
            #[inline(always)]
            fn eq(&self, other: &Self) -> bool {
                self.0 == other.0
            }
        }
    };
}

macro_rules! primitive_float_simd_field_impl {
   ( $($t:ty),* ) => {
        $(

        impl SimdField for $t {
            const ZERO: Self = 0.0;
            const ONE: Self = 1.0;
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
        }
        )*
    };
}

primitive_float_simd_field_impl![f64, f32];

macro_rules! wide_float_numeric_impl {
   ( $($name:ident : $elm:ty : $simdt:ty : $lanes:expr),* ) => {
        $(paste! {

        #[repr(transparent)]
        #[derive(Clone, Copy)]
        pub struct [<$name>]($simdt);

        forward_basic_ops_impl!($name);

        impl SimdField for $name {
            const ZERO: Self = Self(<$simdt>::ZERO);
            const ONE: Self = Self(<$simdt>::ONE);
            const LANES: NonZeroUsize = NonZeroUsize::new($lanes).unwrap();
            type Element = $elm;

            #[inline(always)]
            fn mul_add(self, m: Self, a: Self) -> Self {
                Self(self.0.mul_add(m.0, a.0))
            }

            #[inline(always)]
            fn splat(v: Self::Element) -> Self {
                Self($simdt::splat(v))
            }

            #[inline(always)]
            fn as_slice(&self) -> &[$elm] {
                self.0.as_array()
            }

            #[inline(always)]
            fn as_mut_slice(&mut self) -> &mut [$elm] {
                self.0.as_mut_array()
            }

            #[inline(always)]
            fn reduce_add(self) -> Self::Element {
               self.0.reduce_add()
            }


            #[inline(always)]
            fn abs(self) -> Self {
                Self(self.0.abs())
            }

            #[inline(always)]
            fn max(self, other: Self) -> Self {
                Self(self.0.max(other.0))
            }

        }
        })*
    };
}

wide_float_numeric_impl![
    Simdf64 : f64 : f64x8 : 8,
    // TODO: Update to `f32x16` once LANES and reduce add is implemented.
    Simdf32: f32 : f32x8 : 8
];

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
    f32 : Simdf32,
    f64 : Simdf64
];
