use core::{
    marker::PhantomData,
    mem::{self, MaybeUninit},
    ptr::{NonNull, drop_in_place},
};

use crate::{
    simd::{SimdAble, SimdField},
    storage::{KP1Array, TwoKP1Array},
};

use super::{BVec, Bump, Polynomial, RawPolynomial};

#[derive(Debug, Clone, Copy)]
pub struct PolyfitCfg<T: SimdAble> {
    max_deg: u8,
    halt_epsilon: T,
}

impl<T: SimdAble> PolyfitCfg<T> {
    /// Create a new default configuration with the specified maximum regression polynomial degree.
    #[inline(always)]
    pub const fn new_with_max_deg(max_deg: u8) -> Self {
        Self {
            max_deg,
            halt_epsilon: <T as SimdField>::SF_EPS,
        }
    }

    /// Set the maximum degree that the regressional polynomial
    /// is allowed to have in any dimension.
    #[inline(always)]
    pub const fn with_max_deg(mut self, max_deg: u8) -> Self {
        self.max_deg = max_deg;
        self
    }

    /// Get the maximum degree that the regressional polynomial
    /// is allowed to have in any dimension.
    #[inline(always)]
    pub const fn max_deg(&self) -> u8 {
        self.max_deg
    }

    /// Set the halting epsilon on $gamma_k$, such that if $|gamma_k| < \eps$, then
    /// the algorithm will halt at degree $k$. This can happen if there are duplicate $x$ points
    /// for example, allowing the algorithm to short circuit rather than contaminate its existing
    /// fit.
    #[inline(always)]
    pub const fn with_halt_epsilon(mut self, halt_epsilon: T) -> Self {
        self.halt_epsilon = halt_epsilon;
        self
    }

    /// Get the halting epsilon.
    #[inline(always)]
    pub const fn halt_epsilon(&self) -> T {
        self.halt_epsilon
    }
}

#[derive(Clone, Copy)]
struct GroupedNodeData<const R: usize, T> {
    wv: T,
    xv: T,
    yvs: [T; R],
    p_kv: T,
    p_km1v: T,
}

struct GroupedNode<'a, const R: usize, T> {
    data: GroupedNodeData<R, T>,
    prev: Option<NonNull<Self>>,
    _p: PhantomData<&'a mut Self>,
}

struct GroupedNodeList<'a, const R: usize, T> {
    curr: Option<NonNull<GroupedNode<'a, R, T>>>,
}

struct GroupedNodeListIterMut<'a, const R: usize, T> {
    curr: Option<NonNull<GroupedNode<'a, R, T>>>,
}

impl<'a, const R: usize, T> GroupedNodeList<'a, R, T> {
    #[inline(always)]
    const fn new() -> Self {
        Self { curr: None }
    }

    #[inline(always)]
    fn push(&mut self, ws: &'a Bump, data: GroupedNodeData<R, T>) {
        let new_block = ws.alloc(GroupedNode {
            data,
            prev: self.curr,
            _p: PhantomData,
        });

        self.curr = Some(NonNull::from_mut(new_block));
    }

    #[inline(always)]
    pub fn iter(&mut self) -> GroupedNodeListIterMut<'a, R, T> {
        GroupedNodeListIterMut { curr: self.curr }
    }
}

impl<'a, const R: usize, T> Drop for GroupedNodeList<'a, R, T> {
    #[inline(always)]
    fn drop(&mut self) {
        while let Some(mut node) = self.curr {
            unsafe {
                self.curr = node.as_mut().prev;
                drop_in_place(node.as_ptr());
            }
        }
    }
}

impl<'a, const R: usize, T> Iterator for GroupedNodeListIterMut<'a, R, T> {
    type Item = &'a mut GroupedNodeData<R, T>;

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(node) = self.curr {
            let node = unsafe { &mut *node.as_ptr() };
            self.curr = node.prev;
            return Some(&mut node.data);
        }

        None
    }
}

impl<const R: usize, T: SimdAble> RawPolynomial<R, T> {
    pub fn polyfit_from_iter<I: Iterator<Item = (T, T, [T; R])>>(
        &mut self,
        ws: &Bump,
        cfg: PolyfitCfg<T>,
        iter: I,
    ) {
        let mut data_list = GroupedNodeList::new();
        let mut curr_data = GroupedNodeData {
            wv: T::SimdT::SF_ZERO,
            xv: T::SimdT::SF_ZERO,
            yvs: [T::SimdT::SF_ZERO; R],
            p_km1v: T::SimdT::SF_ZERO,
            p_kv: T::SimdT::SF_ONE,
        };

        let mut data_len = 0;
        let mut lane_i = 0;
        let mut weight_sum = T::SimdT::SF_ZERO;
        for (w, x, ys) in iter {
            unsafe {
                *curr_data.wv.as_mut_slice().get_unchecked_mut(lane_i) = w;
                *curr_data.xv.as_mut_slice().get_unchecked_mut(lane_i) = x;

                for (y, yv) in ys.into_iter().zip(curr_data.yvs.iter_mut()) {
                    *yv.as_mut_slice().get_unchecked_mut(lane_i) = y;
                }
            }
            lane_i += 1;

            if lane_i < T::SimdT::SF_LANES.get() {
                continue;
            }

            weight_sum += curr_data.wv;
            data_list.push(ws, curr_data);
            data_len += T::SimdT::SF_LANES.get();
            lane_i = 0;
        }

        let len = lane_i;
        if len != 0 {
            unsafe {
                // Enforce that extreneous SIMD lanes contribute nothing to the computation.
                curr_data
                    .wv
                    .as_mut_slice()
                    .get_unchecked_mut(len..)
                    .fill(T::SF_ZERO);
            }

            weight_sum += curr_data.wv;
            data_list.push(ws, curr_data);
            data_len += len;
        }

        let weight_sum = weight_sum.reduce_add();
        if !(weight_sum.is_finite() && weight_sum > cfg.halt_epsilon) {
            self.coeffs.clear();
            return;
        }

        let weight_sum_recip = T::SimdT::splat(T::SF_ONE / weight_sum);

        let (d_0s, gamma_0, b_0) = {
            let mut d_0vs = [T::SimdT::SF_ZERO; R];
            let mut gamma_0v = T::SimdT::SF_ZERO;
            let mut b_0v = T::SimdT::SF_ZERO;

            for chunk in data_list.iter() {
                chunk.wv *= weight_sum_recip;
                gamma_0v += chunk.wv;
                b_0v = T::SimdT::mul_add(chunk.wv, chunk.xv, b_0v);
                for r_i in 0..R {
                    unsafe {
                        let d_0v = d_0vs.get_unchecked_mut(r_i);
                        let yv = chunk.yvs.get_unchecked(r_i);
                        *d_0v = T::SimdT::mul_add(chunk.wv, *yv, *d_0v);
                    }
                }
            }

            (
                d_0vs.map(T::SimdT::reduce_add),
                gamma_0v.reduce_add(),
                b_0v.reduce_add(),
            )
        };

        let max_coeffs = data_len.min(cfg.max_deg as usize + 1);
        if max_coeffs == 0 {
            self.coeffs.clear();
            return;
        }

        // In each iteration the highest element of each dimension is overwritten, not modified. As such
        // we need not worry about stale non-zero values still being present.
        self.coeffs.set_len(max_coeffs);

        let mut p_k = BVec::with_capacity_in(max_coeffs, ws);
        p_k.push(T::SF_ONE);
        let mut p_km1 = BVec::with_capacity_in(max_coeffs, ws);

        for r_i in 0..R {
            unsafe {
                let d_0 = *d_0s.get_unchecked(r_i) / gamma_0;
                *self.coeffs.dim_unchecked_mut(r_i).get_unchecked_mut(0) = d_0;
            }
        }

        let mut gamma_km1 = gamma_0;
        let mut minus_b_km1 = -b_0 / gamma_0;
        let mut minus_c_km1 = T::SF_ZERO;
        let mut d_ks = d_0s;

        for k in 1..max_coeffs {
            let (gamma_k, b_k) = {
                let mut d_kvs = [T::SimdT::SF_ZERO; R];
                let mut gamma_kv = T::SimdT::SF_ZERO;
                let mut b_kv = T::SimdT::SF_ZERO;
                let minus_b_km1v = T::SimdT::splat(minus_b_km1);
                let minus_c_km1v = T::SimdT::splat(minus_c_km1);

                for chunk in data_list.iter() {
                    mem::swap(&mut chunk.p_km1v, &mut chunk.p_kv);
                    chunk.p_kv =
                        (chunk.xv + minus_b_km1v).mul_add(chunk.p_km1v, minus_c_km1v * chunk.p_kv);

                    let wp = chunk.wv * chunk.p_kv;

                    for r_i in 0..R {
                        unsafe {
                            let d_kv = d_kvs.get_unchecked_mut(r_i);
                            let yv = chunk.yvs.get_unchecked_mut(r_i);
                            // We repeatedly subtract from the current y values the previous regression, as although
                            // this adds an additional computational step it, on average, decreases the magnitude
                            // of the y values, which appears to lead to less error on higher order fits.
                            *yv = T::SimdT::mul_add(
                                T::SimdT::splat(-*d_ks.get_unchecked(r_i)),
                                chunk.p_km1v,
                                *yv,
                            );
                            *d_kv = wp.mul_add(*yv, *d_kv);
                        }
                    }
                    let wpp = wp * chunk.p_kv;
                    gamma_kv += wpp;
                    b_kv = wpp.mul_add(chunk.xv, b_kv);
                }

                d_ks = d_kvs.map(T::SimdT::reduce_add);
                (gamma_kv.reduce_add(), b_kv.reduce_add())
            };

            if !(gamma_k.is_finite() && gamma_k.abs() > cfg.halt_epsilon) {
                self.coeffs.set_len(k);
                break;
            }

            mem::swap(&mut p_k, &mut p_km1);
            p_k.push(T::SF_ZERO);
            for i in 0..k {
                let p_ki = unsafe { p_k.get_unchecked_mut(i) };
                let p_km1i = unsafe { p_km1.get_unchecked_mut(i) };

                *p_ki = minus_c_km1.mul_add(*p_ki, minus_b_km1 * (*p_km1i));
            }
            p_k.push(T::SF_ZERO);
            for i in (0..k).rev() {
                let p_kip1 = unsafe { p_k.get_unchecked_mut(i + 1) };
                let p_km1i = unsafe { p_km1.get_unchecked_mut(i) };

                *p_kip1 += *p_km1i;
            }

            for r_i in 0..R {
                unsafe {
                    let d_k = d_ks.get_unchecked_mut(r_i);

                    *d_k /= gamma_k;

                    let coeffs = self.coeffs.dim_unchecked_mut(r_i);
                    *coeffs.get_unchecked_mut(k) = *d_k;
                    for i in (0..k).rev() {
                        let coeff = coeffs.get_unchecked_mut(i);
                        let p_ki = *p_k.get_unchecked(i);

                        *coeff = d_k.mul_add(p_ki, *coeff);
                    }
                }
            }

            minus_c_km1 = -gamma_k / gamma_km1;
            minus_b_km1 = -b_k / gamma_k;
            gamma_km1 = gamma_k;
        }
    }
}

impl<const R: usize, T: SimdAble> Polynomial<R, T> {
    #[inline(always)]
    pub fn polyfit_from_iter<I: Iterator<Item = (T, T, [T; R])>>(
        &mut self,
        cfg: PolyfitCfg<T>,
        iter: I,
    ) {
        self.inner.polyfit_from_iter(&self.ws, cfg, iter);
        self.ws.reset();
    }
}

#[repr(C)]
pub struct OnlinePolyfit<const K: usize, const D: usize, T: SimdAble> {
    factorials_1_up: [T; K],
    xks: TwoKP1Array<K, T>,
    yxks: [KP1Array<K, T>; D],
}

impl<const K: usize, const D: usize, T: SimdAble> OnlinePolyfit<K, D, T> {
    pub fn new() -> Self {
        let mut factorial_value: usize = 1;
        let mut mult: usize = 1;
        Self {
            factorials_1_up: [(); K].map(|_| {
                factorial_value *= mult;
                mult += 1;

                T::from_usize(factorial_value)
            }),
            xks: unsafe { MaybeUninit::zeroed().assume_init() },
            yxks: unsafe { MaybeUninit::zeroed().assume_init() },
        }
    }

    /// Update the regression state from `x'` to `x = x' + delta_x`, effectively shifting the domain of the
    /// regression to the left by `delta_x`.
    pub fn rotate(&mut self, delta_x: T) {
        let old_xks = self.xks.clone();
        let mut coeffs = TwoKP1Array::<K, T>::new(T::SF_ZERO);
        unsafe { *coeffs.get_unchecked_mut(0) = T::SF_ONE }
        for i in 1..TwoKP1Array::<K, T>::LEN {
            unsafe {
                *coeffs.get_unchecked_mut(i) = T::SF_ONE;
                for j in (1..i).rev() {
                    let prev_coeff = *coeffs.get_unchecked(j - 1);
                    let curr_coeff = coeffs.get_unchecked_mut(j);
                    *curr_coeff = curr_coeff.mul_add(delta_x, prev_coeff);

                    let curr_xks = self.xks.get_unchecked_mut(i);
                    *curr_xks = T::mul_add(*curr_coeff, *old_xks.get_unchecked(j), *curr_xks);
                }

                let curr_coeff = coeffs.first_mut().unwrap_unchecked();
                *curr_coeff *= delta_x;

                let curr_xks = self.xks.get_unchecked_mut(i);
                *curr_xks = T::mul_add(*curr_coeff, *old_xks.first().unwrap_unchecked(), *curr_xks);
            }
        }

        for d in 0..D {
            let yxks = &mut self.yxks[d];
            let old_yxks = yxks.clone();
            let mut coeffs = KP1Array::<K, T>::new(T::SF_ZERO);
            unsafe { *coeffs.get_unchecked_mut(0) = T::SF_ONE }

            for i in 1..KP1Array::<K, T>::LEN {
                unsafe {
                    *coeffs.get_unchecked_mut(i) = T::SF_ONE;
                    for j in (1..i).rev() {
                        let prev_coeff = *coeffs.get_unchecked(j - 1);
                        let curr_coeff = coeffs.get_unchecked_mut(j);
                        *curr_coeff = curr_coeff.mul_add(delta_x, prev_coeff);

                        let curr_yxks = yxks.get_unchecked_mut(i);
                        *curr_yxks =
                            T::mul_add(*curr_coeff, *old_yxks.get_unchecked(j), *curr_yxks);
                    }

                    let curr_coeff = coeffs.first_mut().unwrap_unchecked();
                    *curr_coeff *= delta_x;

                    let curr_yxks = yxks.get_unchecked_mut(i);
                    *curr_yxks = T::mul_add(
                        *curr_coeff,
                        *old_yxks.first().unwrap_unchecked(),
                        *curr_yxks,
                    );
                }
            }
        }
    }

    /// Scale all sample weights by `r`.
    pub fn scale(&mut self, r: T) {
        for xk in self.xks.iter_mut() {
            *xk *= r;
        }

        for yxks in &mut self.yxks {
            for yxk in yxks.iter_mut() {
                *yxk *= r;
            }
        }
    }

    pub fn update_at_zero(&mut self, derivative: usize, w: T, ys: [T; D]) {
        if derivative > K {
            return;
        }

        let factorial = if derivative == 0 {
            T::SF_ONE
        } else {
            self.factorials_1_up[derivative - 1]
        };
        let w_factorial = w * factorial;

        self.xks[derivative] *= w_factorial;

        for (dim, y) in ys.into_iter().enumerate() {
            let yxk = unsafe {
                self.yxks
                    .get_unchecked_mut(dim)
                    .get_unchecked_mut(derivative)
            };

            *yxk = T::mul_add(w_factorial, y, *yxk);
        }
    }

    pub fn compute_fit(&self) -> [KP1Array<K, T>; D] {
        let mut fit = [(); D].map(|_| KP1Array::<K, T>::new(T::SF_ZERO));
        let mut yxks = self.yxks.clone();

        let mut p_km1 = KP1Array::<K, T>::new(T::SF_ZERO);
        let mut p_km1 = &mut p_km1;

        let mut p_k = KP1Array::<K, T>::new(T::SF_ZERO);
        let mut p_k = &mut p_k;

        let gamma_0 = unsafe { *self.xks.get_unchecked(0) };
        let gamma_0_recip = gamma_0.recip();

        for dim in 0..D {
            let yxks_dim = unsafe { yxks.get_unchecked_mut(0) };
            let yx0_dim = unsafe { yxks_dim.get_unchecked_mut(0) };
            let d_0 = *yx0_dim * gamma_0_recip;

            // Subtracting here makes zero mathematical differrence:
            // <y(x), P_j> = <y(x) - d_k P_k, P_j>    j != k
            // but makes the fit more stable at higher dimensions when the polynomials are not exactly orthogonal.
            *yx0_dim = d_0.mul_add(-unsafe { *self.xks.get_unchecked(0) }, *yx0_dim);

            unsafe { *fit.get_unchecked_mut(dim).get_unchecked_mut(0) = d_0 }
        }

        let mut min_c_km1 = T::SF_ZERO;
        let mut gamma_km1_recip = gamma_0_recip;
        unsafe { *p_km1.get_unchecked_mut(0) = T::SF_ONE };
        for k in 1..=K {
            let mut min_b_km1 = *unsafe { p_km1.get_unchecked(0) };
            // B_k = <xP_k, P_k> / gamma_k
            //     = <x^(k+1) + [x^(k-1)]P_k x^(k-1), P_k>
            //     = <x^(k+1), P_k> / gamma_k + [x^(k-1)]P_k
            for i in 0..k {
                min_b_km1 = T::mul_add(
                    gamma_km1_recip,
                    *unsafe { self.xks.get_unchecked(k + 1 + i) },
                    min_b_km1,
                );
            }
            min_b_km1 = -min_b_km1;

            // gamma_k = <P_k, P_k> = <x^k, P_k>
            let mut gamma_k = unsafe { *self.xks.get_unchecked(k + k) };
            unsafe { *p_k.get_unchecked_mut(k) = T::SF_ONE };

            for i in (1..k).rev() {
                let [pkm1_im1, pkm1_i] =
                    *unsafe { mem::transmute::<_, &[T; 2]>(p_km1.get_unchecked(i - 1)) };
                let pkm1_contribution = pkm1_i.mul_add(min_b_km1, pkm1_im1);

                let pk_i = unsafe { p_k.get_unchecked_mut(i) };
                *pk_i = pk_i.mul_add(min_c_km1, pkm1_contribution);

                gamma_k = pk_i.mul_add(unsafe { *self.xks.get_unchecked(k + i) }, gamma_k);
            }
            unsafe {
                let pkm1_0 = *p_km1.get_unchecked(0);
                let pk_0 = p_k.get_unchecked_mut(0);
                *pk_0 = pk_0.mul_add(min_c_km1, min_b_km1 * pkm1_0);

                gamma_k = pk_0.mul_add(*self.xks.get_unchecked(k), gamma_k);
            }

            let gamma_k_recip = gamma_k.recip();
            for dim in 0..D {
                let yxks_dim = unsafe { yxks.get_unchecked_mut(0) };

                let mut d_k = unsafe { *yxks_dim.get_unchecked(k) };
                for i in (0..k).rev() {
                    d_k = unsafe {
                        yxks_dim
                            .get_unchecked(i)
                            .mul_add(*p_k.get_unchecked(i), d_k)
                    }
                }
                d_k *= gamma_k_recip;

                let fit_dim = unsafe { fit.get_unchecked_mut(dim) };
                for i in 0..k {
                    unsafe {
                        // Subtracting here makes zero mathematical differrence:
                        // <y(x), P_j> = <y(x) - d_k P_k, P_j>    j != k
                        // but makes the fit more stable at higher dimensions when the polynomials are not exactly orthogonal.
                        let yxk_dim_i = yxks_dim.get_unchecked_mut(i);
                        let d_k_coeff = d_k * *p_k.get_unchecked(i);
                        *yxk_dim_i = d_k_coeff.mul_add(-*self.xks.get_unchecked(i), *yxk_dim_i);

                        *fit_dim.get_unchecked_mut(i) += d_k_coeff;
                    }
                }
                unsafe {
                    let yxk_dim_k = yxks_dim.get_unchecked_mut(k);
                    // Subtracting here makes zero mathematical differrence:
                    // <y(x), P_j> = <y(x) - d_k P_k, P_j>    j != k
                    // but makes the fit more stable at higher dimensions when the polynomials are not exactly orthogonal.

                    *yxk_dim_k = d_k.mul_add(-*self.xks.get_unchecked(k), *yxk_dim_k);

                    *fit_dim.get_unchecked_mut(k) = d_k;
                }
            }

            min_c_km1 = gamma_km1_recip * gamma_k;
            gamma_km1_recip = gamma_k_recip;
            mem::swap::<&mut KP1Array<K, T>>(&mut p_k, &mut p_km1);
        }

        fit
    }

    // pub fn compute_fit_with_bias(&self, bias: [(T, T); K]) -> [KP1Array<K, T>; D] {
    //     todo!()
    // }
}
