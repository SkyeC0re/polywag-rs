#![no_std]

extern crate alloc;

use crate::{
    simd::{SimdAble, SimdField},
    storage::Coeffs,
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

/// A polynomial container with `R` range dimensions and functionality specified over externally provided workspaces.
#[repr(transparent)]
pub struct RawPolynomial<const R: usize, T: SimdAble> {
    coeffs: Coeffs<R, T>,
}

impl<const R: usize, T: SimdAble> RawPolynomial<R, T> {
    /// Create a new zero polynomial.
    #[inline(always)]
    pub const fn new() -> Self {
        Self {
            coeffs: Coeffs::new(),
        }
    }

    /// Evaluates the polynomial at the given values and allocates the results inside the workspace.
    #[inline]
    pub fn evaluate_slice<'a>(&self, ws: &'a Bump, xs: &[T]) -> [BBox<'a, [T]>; R] {
        let mut xv = T::SimdT::ZERO;
        self.coeffs.dims().map(|coeffs| unsafe {
            let mut bvec = BVec::with_capacity_in(xs.len(), ws);
            let mut data_ptr = bvec.as_mut_ptr();
            for chunk in xs.chunks(T::SimdT::LANES.get()) {
                ptr::copy_nonoverlapping(
                    chunk.as_ptr(),
                    xv.as_mut_slice().as_mut_ptr() as _,
                    chunk.len(),
                );

                let eval = eval_slice_horner(coeffs, xv);
                ptr::copy_nonoverlapping(eval.as_slice().as_ptr(), data_ptr, chunk.len());
                data_ptr = data_ptr.offset(T::SimdT::LANES.get() as _);
            }

            bvec.set_len(xs.len());
            bvec.into_boxed_slice()
        })
    }

    #[inline]
    pub fn evaluate_array<const N: usize>(&self, xs: [T; N]) -> [[T; N]; R] {
        let mut xv = T::SimdT::ZERO;
        self.coeffs.dims().map(|coeffs| unsafe {
            let mut arr = MaybeUninit::<[T; N]>::uninit();
            let mut data_ptr: *mut T = arr.as_mut_ptr() as _;
            for chunk in xs.chunks(T::SimdT::LANES.get()) {
                ptr::copy_nonoverlapping(
                    chunk.as_ptr(),
                    xv.as_mut_slice().as_mut_ptr() as _,
                    chunk.len(),
                );

                let eval = eval_slice_horner(coeffs, xv);
                ptr::copy_nonoverlapping(eval.as_slice().as_ptr(), data_ptr, chunk.len());
                data_ptr = data_ptr.offset(T::SimdT::LANES.get() as _);
            }

            arr.assume_init()
        })
    }

    /// Computes the derivative of the polynomial in place.
    #[inline]
    pub fn deriv_in_place(&mut self) {
        if self.coeffs.len() == 0 {
            return;
        }

        for coeffs in self.coeffs.dims_mut() {
            for i in 1..coeffs.len() {
                unsafe {
                    *coeffs.get_unchecked_mut(i - 1) = *coeffs.get_unchecked(i) * T::from_usize(i);
                }
            }
        }

        self.coeffs.set_len(self.coeffs.len() - 1);
    }

    /// Computes the anti-derivative of the polynomial in place with `C = 0`.
    #[inline]
    pub fn anti_deriv_in_place(&mut self) {
        if self.coeffs.len() == 0 {
            return;
        }

        self.coeffs.set_len(self.coeffs.len() + 1);

        for coeffs in self.coeffs.dims_mut() {
            for i in 1..coeffs.len() {
                unsafe {
                    *coeffs.get_unchecked_mut(i) = *coeffs.get_unchecked(i - 1) / T::from_usize(i);
                }
            }

            unsafe { *coeffs.get_unchecked_mut(0) = T::ZERO };
        }
    }

    /// Truncates higher order coefficients below or equal to the cutoff magnitude, starting from the highest degree coefficient
    /// until a coefficient is found that is greater than the cutoff magnitude.
    #[inline]
    pub fn truncate_high(&mut self, cutoff: T) {
        let mut new_len = 0;
        for coeffs in self.coeffs.dims() {
            for (c_i, c) in coeffs.iter().copied().enumerate().rev() {
                if c_i <= new_len {
                    break;
                }

                if c.abs() > cutoff {
                    new_len = c_i;
                    break;
                }
            }
        }
    }
}

impl<const R: usize, T: SimdAble> Deref for RawPolynomial<R, T> {
    type Target = Coeffs<R, T>;

    #[inline(always)]
    fn deref(&self) -> &Self::Target {
        &self.coeffs
    }
}

impl<const R: usize, T: SimdAble> DerefMut for RawPolynomial<R, T> {
    #[inline(always)]
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.coeffs
    }
}

/// A polynomial with `R` range dimensions and a managed workspace.
pub struct Polynomial<const R: usize, T: SimdAble> {
    inner: RawPolynomial<R, T>,
    ws: Bump,
}

impl<const R: usize, T: SimdAble> Polynomial<R, T> {
    /// Create a new zero polynomial.
    #[inline(always)]
    pub fn new() -> Self {
        Self {
            inner: RawPolynomial::new(),
            ws: Bump::new(),
        }
    }

    /// Evaluates the polynomial at the given values and allocates the results inside the workspace.
    #[inline(always)]
    pub fn evaluate_slice<'a>(&'a mut self, xs: &[T]) -> Reset<'a, [BBox<'a, [T]>; R]> {
        let ws = &mut self.ws;
        let ws_ptr = NonNull::from_mut(ws);

        Reset {
            data: mem::ManuallyDrop::new(self.inner.evaluate_slice(ws, xs)),
            ws: ws_ptr,
            _p: PhantomData,
        }
    }

    /// Deallocates the workspace, freeing up any allocated memory. After this is called, further operations on the polynomial will likely
    /// require workspace reallocation again.
    #[inline(always)]
    pub fn deallocate_workspace(&mut self) {
        self.ws = Bump::new();
    }
}

impl<const R: usize, T: SimdAble> Deref for Polynomial<R, T> {
    type Target = Coeffs<R, T>;

    #[inline(always)]
    fn deref(&self) -> &Self::Target {
        &self.inner.coeffs
    }
}

impl<const R: usize, T: SimdAble> DerefMut for Polynomial<R, T> {
    #[inline(always)]
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.inner.coeffs
    }
}

/// A temporary handle to data inside a workspace. The workspace will be cleared automatically once the handle is dropped.
pub struct Reset<'a, T> {
    data: mem::ManuallyDrop<T>,
    ws: NonNull<Bump>,
    _p: PhantomData<&'a mut Bump>,
}

impl<'a, T> Deref for Reset<'a, T> {
    type Target = T;

    #[inline(always)]
    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

impl<'a, T> DerefMut for Reset<'a, T> {
    #[inline(always)]
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.data
    }
}

impl<'a, T> Drop for Reset<'a, T> {
    #[inline(always)]
    fn drop(&mut self) {
        unsafe {
            mem::ManuallyDrop::drop(&mut self.data);
            self.ws.as_mut().reset();
        }
    }
}

impl<const R: usize, T: SimdAble> Polynomial<R, T> {}

#[inline]
fn eval_slice_horner<SF: SimdField>(slice: &[SF::Element], x: SF) -> SF {
    let mut result = SF::ZERO;
    for ci in slice.into_iter().rev().copied() {
        result = SF::mul_add(result, x, SF::splat(ci));
    }

    result
}
