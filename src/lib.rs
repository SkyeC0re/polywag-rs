// #![no_std]

extern crate alloc;

use crate::{
    simd::{SimdAble, SimdField},
    storage::{Coeffs, KP1Array},
};
pub use bumpalo::{Bump, boxed::Box as BBox, collections::Vec as BVec};
use core::{
    marker::PhantomData,
    mem::{self, MaybeUninit},
    ops::{Deref, DerefMut},
    ptr::{self, NonNull},
};

mod polyfit;
mod storage;
pub use polyfit::*;

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

    // /// Computes the derivative of the polynomial in place.
    // #[inline]
    // pub fn deriv_in_place(&mut self) {
    //     if self.coeffs.len() == 0 {
    //         return;
    //     }

    //     for coeffs in self.coeffs.dims_mut() {
    //         for i in 1..coeffs.len() {
    //             unsafe {
    //                 *coeffs.get_unchecked_mut(i - 1) = *coeffs.get_unchecked(i) * T::from_usize(i);
    //             }
    //         }
    //     }

    //     self.coeffs.set_len(self.coeffs.len() - 1);
    // }
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
    let mut result = SF::SF_ZERO;
    for ci in slice.into_iter().rev().copied() {
        result = SF::mul_add(result, x, SF::splat(ci));
    }

    result
}
