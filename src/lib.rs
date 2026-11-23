//! A `no-std` library for online, positively weighted polynomial regression.
//!
//! # Basic Usage
//!
//! ```rust
//! extern crate polywag;
//! use polywag::{RegressionState, kp1_arr};
//!
//! pub fn main() {
//!     // Create a new regression state with a maximum regression polynomial degree 2 and without any samples.
//!     let mut r = RegressionState::<f32, 2>::new();
//!
//!     // Add some zeroeth-degree (`x, y`) samples.
//!     r.update(0, 1.0, 0.0, [2.0]);
//!     r.update(0, 1.0, 1.0, [3.0]);
//!     r.update(0, 1.0, 2.0, [4.0]);
//!
//!     // Compute the regression based on the current sample weights.
//!     let [fit] = r.compute_fit();
//!
//!     // This should produce the following approximate polynomial regressions up to degree 2:
//!
//!     // [ 3.0, 0.0, 0.0 ]
//!     fit.polynomials.deg(0);
//!
//!     // [ 2.0, 1.0, 0.0 ]
//!     fit.polynomials.deg(1);
//!
//!     // [ 2.0, 1.0, 0.0 ]
//!     fit.polynomials.deg(2);
//! }
//! ```
//!
//! # Numerical Stability
//!
//! Care was taken to keep the algorithm as stable as possible whilst still allowing the algorithm to be online
//! (online in the sense that samples are not stored individually and its performance characteristics are unaffected
//! by the amount of samples currently added to the regression state). To accomplish this, the main technique used here
//! is [Modified Gram Schmidt](https://en.wikipedia.org/wiki/Gram%E2%80%93Schmidt_process).
//!
//! # Bias
//!
//! We allow regressions to be generated using special bias samples, one for each derivative:
//!
//! ```rust
//! extern crate polywag;
//! use polywag::{RegressionState, kp1_arr};
//!
//! pub fn main() {
//!     let mut r = RegressionState::<f32, 2>::new();
//!
//!     // Add some bad samples (zero weight or not enough unique `x` values).
//!     r.update(0, 0.0, 1.0, [2.0]);
//!     r.update(0, 0.0, 1.0, [3.0]);
//!
//!     // Compute the regression based on the bad weight plus a bias at `x=0` in each dimension with weight 1.
//!     let [fit] = r.compute_fit_with_bias(kp1_arr![
//!         (1.0, [1.0]),
//!         (1.0, [2.0]),
//!         (1.0, [3.0]),
//!     ]);
//!
//!     // This should produce the following approximate polynomial regressions up to degree 2:
//!
//!     // [ 1.0, 0.0, 0.0 ]
//!     fit.polynomials.deg(0);
//!
//!     // [ 1.0, 2.0, 0.0 ]
//!     fit.polynomials.deg(1);
//!
//!     // [ 1.0, 2.0, 3.0 ]
//!     fit.polynomials.deg(2);
//! }
//! ```
//!
//!
//! The algorithm is written in such a way that if the bias samples by themselves do not yield
//! a numerical regression with infinities or `NaN`s then neither will the full regression (the
//! bias samples together with the regular samples) as a result of catastrophic cancellation.
//!
//!

#![no_std]

use crate::simd::{SimdAble, SimdField};
use bytemuck::zeroed;
use core::{
    mem::MaybeUninit,
    ops::{Deref, DerefMut},
    ptr,
};

mod polyfit;
mod storage;
pub use polyfit::*;
pub use storage::{FitPolynomials, KP1Array};

pub mod simd;

/// A stack allocated polynomial of degree `K`.
#[repr(transparent)]
pub struct SPolynomial<T: SimdAble, const K: usize>(pub KP1Array<T, K>);

impl<T: SimdAble, const K: usize> SPolynomial<T, K> {
    /// Create a new zero polynomial.
    #[inline(always)]
    pub const fn new() -> Self {
        Self(zeroed())
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
