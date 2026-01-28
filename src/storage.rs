extern crate alloc;

use alloc::vec::Vec;
use core::array;
use core::mem::MaybeUninit;
use core::ops::Index;
use core::ops::IndexMut;
use core::ptr;
use core::slice;

use crate::simd::SimdAble;

#[derive(Clone)]
pub struct Coeffs<const R: usize, T: SimdAble> {
    slab: Vec<T>,
    len: usize,
}

impl<const R: usize, T: SimdAble> Coeffs<R, T> {
    const MAX_CAP: usize = const {
        if R > isize::MAX as _ {
            panic!("Maximimum dimensions exceeded")
        } else if R == 0 {
            isize::MAX as _
        } else {
            isize::MAX as usize / R
        }
    };

    /// Create the zero polynomial.
    #[inline(always)]
    pub const fn new() -> Self {
        Self {
            slab: Vec::new(),
            len: 0,
        }
    }

    /// The number of coefficients (polynomial degree minus one) of the polynomial in all `R` dimensions.
    #[inline(always)]
    pub const fn len(&self) -> usize {
        self.len
    }

    #[inline(always)]
    pub(crate) fn ensure_capacity(&mut self, cap: usize) {
        if cap > Self::MAX_CAP {
            panic!("Maximum capacity exceeded");
        }

        self.slab
            .reserve((cap * R).saturating_sub(self.slab.capacity()));
    }

    /// Sets the number of coefficients of the polynomial in all `R` dimensions. If the new length is smaller
    /// than the current length then **higher order coefficients are truncated**. If the new length is larger than the
    /// current length then the added higher order coefficients are set to zero.
    pub fn set_len(&mut self, len: usize) {
        if self.len == len {
            return;
        }

        if self.len > len {
            // Safety: We need not call `Drop` on anything, as `T` is `SimdAble` which must be `Copy`.
            let mut i = 1;
            while i < R {
                unsafe {
                    ptr::copy(
                        self.slab.as_ptr().offset((self.len * i) as _),
                        self.slab.as_mut_ptr().offset((len * i) as _),
                        len,
                    );
                }
                i += 1;
            }
        } else {
            self.ensure_capacity(len * R);

            unsafe {
                self.slab
                    .spare_capacity_mut()
                    .get_unchecked_mut((self.len * R)..(len * R))
                    .fill(MaybeUninit::new(T::ZERO));
            }

            let mut i = R;
            while i > 1 {
                unsafe {
                    self.slab
                        .spare_capacity_mut()
                        .get_unchecked_mut((self.len * (i - 1))..(len * i))
                        .rotate_right(i - 1);
                }

                i -= 1;
            }
        }

        self.len = len;
    }

    /// Clears the current set of coefficients, effectively creating the zero polynomial.
    #[inline(always)]
    pub fn clear(&mut self) {
        // Safety: We need not call `Drop` on anything, as `T` is `SimdAble` which must be `Copy`.
        self.len = 0;
    }

    /// Returns a reference to the coefficients associated with the selected dimension.
    ///
    /// # Safety
    ///
    /// This function does no bounds checking on the selected dimension index and selecting an out of bounds index
    /// is considered undefined behaviour.
    #[inline(always)]
    pub unsafe fn dim_unchecked(&self, r_i: usize) -> &[T] {
        unsafe { slice::from_raw_parts(self.slab.as_ptr().offset((r_i * self.len) as _), self.len) }
    }

    /// Returns a reference to the coefficients associated with the selected dimension. Returns `None` if the index is out of bounds.
    #[inline(always)]
    pub fn dim(&self, r_i: usize) -> Option<&[T]> {
        if r_i >= R {
            return None;
        }

        Some(unsafe { self.dim_unchecked(r_i) })
    }

    /// Returns a reference to all coefficients for all dimensions.
    #[inline(always)]
    pub fn dims(&self) -> [&[T]; R] {
        array::from_fn(
            #[inline(always)]
            |r_i| unsafe { self.dim_unchecked(r_i) },
        )
    }

    /// Returns a mutable reference to the coefficients associated with the selected dimension.
    ///
    /// # Safety
    ///
    /// This function does no bounds checking on the selected dimension index and selecting an out of bounds index
    /// is considered undefined behaviour.
    #[inline(always)]
    pub unsafe fn dim_unchecked_mut(&mut self, r_i: usize) -> &mut [T] {
        unsafe {
            slice::from_raw_parts_mut(
                self.slab.as_mut_ptr().offset((r_i * self.len) as _),
                self.len,
            )
        }
    }

    /// Returns a mutable reference tto the coefficients associated with the selected dimension. Returns `None` if the index is out of bounds.
    #[inline(always)]
    pub fn dim_mut(&mut self, r_i: usize) -> Option<&mut [T]> {
        if r_i >= R {
            return None;
        }

        Some(unsafe { self.dim_unchecked_mut(r_i) })
    }

    /// Returns a mutable reference to all coefficients for all dimensions.
    #[inline(always)]
    pub fn dims_mut(&mut self) -> [&mut [T]; R] {
        array::from_fn(
            #[inline(always)]
            |r_i| unsafe {
                slice::from_raw_parts_mut(
                    self.slab.as_mut_ptr().offset((r_i * self.len) as _),
                    self.len,
                )
            },
        )
    }
}

impl<const R: usize, T: SimdAble> Index<usize> for Coeffs<R, T> {
    type Output = [T];

    #[inline(always)]
    fn index(&self, r_i: usize) -> &[T] {
        self.dim(r_i).expect("Index out of bounds")
    }
}

impl<const R: usize, T: SimdAble> IndexMut<usize> for Coeffs<R, T> {
    #[inline(always)]
    fn index_mut(&mut self, r_i: usize) -> &mut [T] {
        self.dim_mut(r_i).expect("Index out of bounds")
    }
}

#[cfg(test)]
mod tests {
    use super::Coeffs;

    #[test]
    fn scale_up_scale_down() {
        const R_DIMS: usize = 3;
        let mut coeffs = Coeffs::<R_DIMS, f32>::new();

        coeffs.set_len(5);

        for r_i in 0..R_DIMS {
            let slice = &mut coeffs[r_i];
            assert_eq!(slice.len(), 5);
            assert!(slice.iter().copied().all(|c| c == 0.0));

            for (c_i, c) in slice.iter_mut().enumerate() {
                *c = ((r_i + 1) * c_i) as _;
            }

            let slice = &coeffs[r_i];
            assert!(slice.len() == 5);
            assert!(
                slice
                    .iter()
                    .copied()
                    .enumerate()
                    .all(|(c_i, c)| c == ((r_i + 1) * c_i) as f32)
            );
        }

        coeffs.set_len(6);
        for r_i in 0..R_DIMS {
            let slice = &coeffs[r_i];
            assert_eq!(slice.len(), 6);
            assert!(
                slice
                    .iter()
                    .copied()
                    .enumerate()
                    .all(|(c_i, c)| if c_i < 5 {
                        c == ((r_i + 1) * c_i) as f32
                    } else {
                        c == 0.0
                    })
            );
        }

        coeffs.set_len(4);
        for r_i in 0..R_DIMS {
            let slice = &coeffs[r_i];
            assert_eq!(slice.len(), 4);
            assert!(
                slice
                    .iter()
                    .copied()
                    .enumerate()
                    .all(|(c_i, c)| c == ((r_i + 1) * c_i) as f32)
            );
        }

        coeffs.set_len(0);
        for r_i in 0..R_DIMS {
            let slice = &coeffs[r_i];
            assert_eq!(slice.len(), 0);
        }
    }
}
