// #![no_std]

use crate::{
    simd::{SimdAble, SimdField},
    storage::KP1Array,
};
use core::{
    mem::MaybeUninit,
    ops::{Deref, DerefMut},
    ptr,
};

mod polyfit;
mod storage;
pub use polyfit::*;
pub use storage::Fit;

pub mod simd;

/// A stack allocated polynomial of degree `K`.
#[repr(transparent)]
pub struct SPolynomial<T: SimdAble, const K: usize>(pub KP1Array<T, K>);

impl<T: SimdAble, const K: usize> SPolynomial<T, K> {
    /// Create a new zero polynomial.
    #[inline(always)]
    pub const fn new() -> Self {
        Self(KP1Array::zeroed())
    }

    #[inline]
    pub fn evaluate_array<const N: usize>(&self, xs: [T; N]) -> [T; N] {
        let mut xv = T::SimdT::SF_ZERO;
        unsafe {
            let mut arr = MaybeUninit::<[T; N]>::uninit();
            let mut data_ptr: *mut T = arr.as_mut_ptr() as _;
            for chunk in xs.chunks(T::SimdT::SF_LANES.get()) {
                ptr::copy_nonoverlapping(
                    chunk.as_ptr(),
                    xv.as_mut_slice().as_mut_ptr() as _,
                    chunk.len(),
                );

                let eval = eval_slice_horner(&self.0, xv);
                ptr::copy_nonoverlapping(eval.as_slice().as_ptr(), data_ptr, chunk.len());
                data_ptr = data_ptr.offset(T::SimdT::SF_LANES.get() as _);
            }

            arr.assume_init()
        }
    }
}

impl<T: SimdAble, const K: usize> Deref for SPolynomial<T, K> {
    type Target = [T];

    #[inline(always)]
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<T: SimdAble, const K: usize> DerefMut for SPolynomial<T, K> {
    #[inline(always)]
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

#[inline]
fn eval_slice_horner<SF: SimdField>(slice: &[SF::Element], x: SF) -> SF {
    let mut iter = slice.into_iter().rev().copied();
    let mut result = if let Some(end_coeff) = iter.next() {
        SF::splat(end_coeff)
    } else {
        return SF::SF_ZERO;
    };
    for ci in iter {
        result = SF::mul_add(result, x, SF::splat(ci));
    }

    result
}
