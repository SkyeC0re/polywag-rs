// #![no_std]

extern crate alloc;

use crate::simd::{SimdAble, SimdField};
use alloc::vec::Vec;
pub use bumpalo::{Bump, boxed::Box as BBox, collections::Vec as BVec};
use core::{
    marker::PhantomData,
    mem,
    ops::{Deref, DerefMut},
    ptr::{self, NonNull},
};
use std::mem::MaybeUninit;

mod polyfit;
mod storage;
pub use polyfit::*;

pub mod simd;

/// A polynomial container with `R` range dimensions and functionality specified over externally provided workspaces.
#[repr(transparent)]
pub struct RawPolynomial<const R: usize, T: SimdAble> {
    coeffs: [Vec<T>; R],
}

impl<const R: usize, T: SimdAble> RawPolynomial<R, T> {
    /// Create a new zero polynomial.
    #[inline(always)]
    pub const fn new() -> Self {
        Self {
            coeffs: [const { Vec::new() }; R],
        }
    }

    /// Evaluates the polynomial at the given values and allocates the results inside the workspace.
    #[inline]
    pub fn evaluate_slice<'a>(&self, ws: &'a Bump, xs: &[T]) -> [BBox<'a, [T]>; R] {
        let mut xv = T::SimdT::ZERO;
        self.coeffs.each_ref().map(|coeffs| unsafe {
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

    #[inline(always)]
    pub fn evaluate_array<const N: usize>(&self, xs: [T; N]) -> [[T; N]; R] {
        let mut xv = T::SimdT::ZERO;
        self.coeffs.each_ref().map(|coeffs| unsafe {
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

    #[inline]
    pub fn deriv_in_place(&mut self) {
        for coeffs in &mut self.coeffs {
            if coeffs.len() == 0 {
                continue;
            }

            for i in 1..coeffs.len() {
                unsafe {
                    *coeffs.get_unchecked_mut(i - 1) = *coeffs.get_unchecked(i) * T::from_usize(i);
                }
            }

            coeffs.pop();
        }
    }

    #[inline]
    pub fn anti_deriv_in_place(&mut self) {
        for coeffs in &mut self.coeffs {
            if let Some(last) = coeffs.last() {
                coeffs.push(*last / T::from_usize(coeffs.len()));
            } else {
                continue;
            }

            for i in 1..(coeffs.len() - 1) {
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
        for coeffs in &mut self.coeffs {
            while coeffs.pop_if(|coeff| !(coeff.abs() > cutoff)).is_some() {}
        }
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

pub struct EvalY<'a, const R: usize, T: SimdAble> {
    ys: [BBox<'a, [T]>; R],
}

pub struct YIter<'a, const R: usize, T: SimdAble> {
    eval: &'a EvalY<'a, R, T>,
    i: usize,
    len: usize,
}

impl<'a, const R: usize, T: SimdAble> Iterator for YIter<'a, R, T> {
    type Item = [T; R];

    #[inline(always)]
    fn next(&mut self) -> Option<Self::Item> {
        if self.i >= self.len {
            return None;
        }

        let i = self.i;
        self.i += 1;

        Some(
            self.eval
                .ys
                .each_ref()
                .map(|ys| *unsafe { ys.get_unchecked(i) }),
        )
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

impl<const R: usize, T> PartialEq<RawPolynomial<R, T>> for RawPolynomial<R, T>
where
    T: SimdAble,
{
    fn eq(&self, other: &RawPolynomial<R, T>) -> bool {
        for (mut short, mut long) in self.coeffs.iter().zip(other.coeffs.iter()) {
            if short.len() > long.len() {
                mem::swap(&mut short, &mut long);
            }

            let mut z = T::SimdT::ZERO;
            for chunk in long[short.len()..].chunks(T::SimdT::LANES.get()) {
                unsafe {
                    ptr::copy_nonoverlapping(
                        chunk.as_ptr(),
                        z.as_mut_slice().as_mut_ptr(),
                        chunk.len(),
                    );
                }
                if z != T::SimdT::ZERO {
                    return false;
                }
            }

            let mut start = 0;
            let mut c1 = T::SimdT::ONE;
            let mut c2 = T::SimdT::ONE;
            while start < short.len() {
                let chunk_size = usize::min(T::SimdT::LANES.get(), short.len() - start);
                unsafe {
                    ptr::copy_nonoverlapping(
                        short.get_unchecked(start..).as_ptr(),
                        c1.as_mut_slice().as_mut_ptr(),
                        chunk_size,
                    );
                    ptr::copy_nonoverlapping(
                        long.get_unchecked(start..).as_ptr(),
                        c2.as_mut_slice().as_mut_ptr(),
                        chunk_size,
                    );
                }

                if c1 != c2 {
                    return false;
                }

                start += chunk_size;
            }
        }

        true
    }
}

#[inline]
fn eval_slice_horner<SF: SimdField>(slice: &[SF::Element], x: SF) -> SF {
    let mut result = SF::ZERO;
    for ci in slice.into_iter().rev().copied() {
        result = SF::mul_add(result, x, SF::splat(ci));
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_noisy() {
        let mut ws = Bump::new();

        let mut poly = RawPolynomial::<2, f32>::new();

        let fit_vals = Vec::from_iter((-50..50).map(|x| {
            println!("x {x}");

            let x = (x as f32) * 0.1 + 1.0;
            (1.0, x, [x, 77.5 * x * x - 1.0])
        }));

        poly.polyfit_from_iter(
            &ws,
            PolyfitCfg::<f32>::new()
                .with_max_deg(6)
                .with_halt_epsilon(-1e-2),
            fit_vals.iter().copied(),
        );

        let vals = poly.evaluate_slice(
            &ws,
            &fit_vals.iter().map(|(_, x, _)| *x).collect::<Vec<f32>>(),
        );

        let sum = vals[1]
            .iter()
            .zip(fit_vals.iter().copied())
            .map(|(&y_computed, (w, x, [_, y]))| w * (y_computed - y).powi(2))
            .sum::<f32>();

        println!("New Sum: {sum:?}");

        println!("{:.10?}", poly.coeffs[0]);
        println!("{:.10?}", poly.coeffs[1]);
    }
}
