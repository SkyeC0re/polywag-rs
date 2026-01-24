use core::slice;

use crate::simd::SimdAble;

pub struct Storage<const R: usize, T: SimdAble> {
    slab: Vec<T>,
    len: usize,
}

impl<const R: usize, T: SimdAble> Storage<R, T> {
    const MAX_LEN: usize = const {
        if R == 0 {
            isize::MAX as _
        } else {
            isize::MAX as usize / R
        }
    };

    #[inline(always)]
    pub const fn new() -> Self {
        Self {
            slab: Vec::new(),
            len: 0,
        }
    }

    pub fn ensure_len(&mut self, len: usize) {
        if self.len >= len {
            return;
        }

        if len > Self::MAX_LEN {
            panic!("Maximum length exceeded");
        }

        self.slab.spare_capacity_mut()
    }
}
