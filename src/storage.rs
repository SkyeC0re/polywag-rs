extern crate alloc;
use core::fmt::Debug;
use core::mem::MaybeUninit;
use core::ops::Deref;
use core::ops::DerefMut;
use core::ptr;
use core::slice;

use crate::SPolynomial;
use crate::simd::SimdAble;

#[derive(Clone)]
#[repr(C)]
pub struct KP1Array<T: SimdAble, const K: usize>(T, [T; K]);

impl<T: SimdAble, const K: usize> Debug for KP1Array<T, K> {
    #[inline]
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        self.deref().fmt(f)
    }
}

impl<T: SimdAble, const K: usize> KP1Array<T, K> {
    pub const LEN: usize = K + 1;

    #[inline(always)]
    pub const fn zeroed() -> Self {
        unsafe { MaybeUninit::zeroed().assume_init() }
    }
}
impl<T: SimdAble, const K: usize> Deref for KP1Array<T, K> {
    type Target = [T];

    #[inline(always)]
    fn deref(&self) -> &Self::Target {
        unsafe { slice::from_raw_parts((self as *const Self).cast::<T>(), K + 1) }
    }
}

impl<T: SimdAble, const K: usize> DerefMut for KP1Array<T, K> {
    #[inline(always)]
    fn deref_mut(&mut self) -> &mut Self::Target {
        unsafe { slice::from_raw_parts_mut((self as *mut Self).cast::<T>(), K + 1) }
    }
}

#[derive(Clone)]
#[repr(C)]
pub struct TwoKP1Array<T: SimdAble, const K: usize>(T, [T; K], [T; K]);

impl<T: SimdAble, const K: usize> TwoKP1Array<T, K> {
    pub const LEN: usize = K + K + 1;

    #[inline(always)]
    pub const fn zeroed() -> Self {
        unsafe { MaybeUninit::zeroed().assume_init() }
    }
}

impl<T: SimdAble, const K: usize> Debug for TwoKP1Array<T, K> {
    #[inline]
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        self.deref().fmt(f)
    }
}

impl<T: SimdAble, const K: usize> Deref for TwoKP1Array<T, K> {
    type Target = [T];

    #[inline(always)]
    fn deref(&self) -> &Self::Target {
        unsafe { slice::from_raw_parts((self as *const Self).cast::<T>(), Self::LEN) }
    }
}

impl<T: SimdAble, const K: usize> DerefMut for TwoKP1Array<T, K> {
    #[inline(always)]
    fn deref_mut(&mut self) -> &mut Self::Target {
        unsafe { slice::from_raw_parts_mut((self as *mut Self).cast::<T>(), Self::LEN) }
    }
}

/// Stores all sums $\sum_{i=1}^{N_l} w_{l, i} x_{l, i}^k$ in the following form (l, k):
/// <br>(K, 0)
/// <br>(K - 1, 0), (K - 1, 1), (K - 1, 2)
/// <br>                    .
/// <br>                    .
/// <br>                    .
/// <br>(1, 0)    , (1, 1)    , (1, 2)    , ... , (1, 2K - 2)
/// <br>(0, 0)    , (0, 1)    , (1, 2)    , ... , (0, 2K - 2), (0, 2K - 1), (0, 2K)
#[derive(Clone)]
#[repr(C)]
pub(crate) struct XlkSums<T: SimdAble, const K: usize>(T, [[T; K]; 2], [[T; K]; K]);

impl<T: SimdAble, const K: usize> XlkSums<T, K> {
    const LEN: usize = const { (K + 1).checked_mul(K + 1).unwrap() };

    #[inline]
    pub const fn zeroed() -> Self {
        unsafe { MaybeUninit::zeroed().assume_init() }
    }

    #[inline(always)]
    pub const unsafe fn get_l_xks(&self, l: usize) -> &[T] {
        let kml = K - l;
        unsafe { slice::from_raw_parts(((&self.0) as *const T).add(kml * kml), (kml << 1) + 1) }
    }

    #[inline(always)]
    pub const unsafe fn get_l_xk(&self, l: usize, k: usize) -> &T {
        let kml = K - l;
        unsafe { &*((&self.0) as *const T).add(kml * kml + k) }
    }

    #[inline(always)]
    pub const unsafe fn get_l_xks_mut(&mut self, l: usize) -> &mut [T] {
        let kml = K - l;
        unsafe {
            slice::from_raw_parts_mut(((&mut self.0) as *mut T).add(kml * kml), (kml << 1) + 1)
        }
    }

    #[inline(always)]
    pub const unsafe fn get_l_xk_mut(&mut self, l: usize, k: usize) -> &mut T {
        let kml = K - l;
        unsafe { &mut *((&mut self.0) as *mut T).add(kml * kml + k) }
    }

    #[inline(always)]
    pub const fn as_raw_slice_mut(&mut self) -> &mut [T] {
        unsafe { slice::from_raw_parts_mut((&mut self.0) as *mut T, Self::LEN) }
    }
}

impl<T: SimdAble, const K: usize> Debug for XlkSums<T, K> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let mut list = f.debug_list();

        for l in 0..=K {
            list.entry(&unsafe { self.get_l_xks(l) });
        }

        list.finish()
    }
}

#[repr(C)]
pub(crate) struct PackedFitCoeffs<T: SimdAble, const K: usize>(T, [[T; K]; 2], [[T; K]; K]);

impl<T: SimdAble, const K: usize> PackedFitCoeffs<T, K> {
    const LEN: usize = const { (K + 1).checked_mul(K + 1).unwrap() };

    #[inline]
    pub const fn zeroed() -> Self {
        unsafe { MaybeUninit::zeroed().assume_init() }
    }

    #[inline(always)]
    pub const unsafe fn get_pk(&self, k: usize) -> &[T] {
        unsafe { slice::from_raw_parts(((&self.0) as *const T).add(k * k), k + 1) }
    }

    #[inline(always)]
    pub const unsafe fn get_pk_i(&self, k: usize, i: usize) -> &T {
        unsafe { &*((&self.0) as *const T).add(k * k + i) }
    }

    #[inline(always)]
    pub const unsafe fn get_pk_mut(&mut self, k: usize) -> &mut [T] {
        unsafe { slice::from_raw_parts_mut(((&mut self.0) as *mut T).add(k * k), k + 1) }
    }

    #[inline(always)]
    pub const unsafe fn get_pk_i_mut(&mut self, k: usize, i: usize) -> &mut T {
        unsafe { &mut *((&mut self.0) as *mut T).add(k * k + i) }
    }

    #[inline(always)]
    pub const fn as_raw_slice_mut(&mut self) -> &mut [T] {
        unsafe { slice::from_raw_parts_mut((&mut self.0) as *mut T, Self::LEN) }
    }
}

#[repr(C)]
#[derive(Debug, Clone)]
pub struct Fit<T: SimdAble, const K: usize>(T, [[T; K]; 2], [[T; K]; K]);

impl<T: SimdAble, const K: usize> Fit<T, K> {
    const LEN: usize = const { (K + 1).checked_mul(K + 1).unwrap() };

    #[inline]
    pub(crate) const fn zeroed() -> Self {
        unsafe { MaybeUninit::zeroed().assume_init() }
    }

    #[inline(always)]
    pub(crate) const unsafe fn get_pk(&self, k: usize) -> &[T] {
        unsafe { slice::from_raw_parts(((&self.0) as *const T).add(k * k), k + 1) }
    }

    #[inline(always)]
    pub(crate) const unsafe fn get_pk_i(&self, k: usize, i: usize) -> &T {
        unsafe { &*((&self.0) as *const T).add(k * k + i) }
    }

    #[inline(always)]
    pub(crate) const unsafe fn get_pk_mut(&mut self, k: usize) -> &mut [T] {
        unsafe { slice::from_raw_parts_mut(((&mut self.0) as *mut T).add(k * k), k + 1) }
    }

    #[inline(always)]
    pub(crate) const unsafe fn get_pk_i_mut(&mut self, k: usize, i: usize) -> &mut T {
        unsafe { &mut *((&mut self.0) as *mut T).add(k * k + i) }
    }

    pub(crate) const unsafe fn transfer_km1_k(&mut self, k: usize) {
        let start = (&mut self.0) as *mut T;
        unsafe {
            ptr::copy_nonoverlapping(start.add((k - 1) * (k - 1)), start.add(k * k), k);
        }
    }

    #[inline(always)]
    pub(crate) const fn as_raw_slice_mut(&mut self) -> &mut [T] {
        unsafe { slice::from_raw_parts_mut((&mut self.0) as *mut T, Self::LEN) }
    }

    #[inline]
    pub fn deg(&self, degree: usize) -> SPolynomial<T, K> {
        let mut poly = SPolynomial::new();
        let k_max = degree.min(K);

        unsafe {
            ptr::copy_nonoverlapping(self.get_pk(k_max).as_ptr(), poly.as_mut_ptr(), k_max + 1);
        }
        poly
    }

    #[inline]
    pub fn max_deg(&self) -> SPolynomial<T, K> {
        self.deg(K)
    }
}

#[cfg(test)]
mod tests {
    // TODO
}
