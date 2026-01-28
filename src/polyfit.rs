use core::{
    f32,
    marker::PhantomData,
    mem,
    ptr::{NonNull, drop_in_place},
};

use crate::simd::{SimdAble, SimdField};

use super::{BVec, Bump, Polynomial, RawPolynomial};

#[derive(Debug, Clone, Copy)]
pub struct PolyfitCfg<T: SimdAble> {
    max_deg: u8,
    halt_epsilon: T,
}

impl<T: SimdAble> PolyfitCfg<T> {
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

impl PolyfitCfg<f32> {
    pub const fn new() -> Self {
        Self {
            max_deg: u8::MAX,
            halt_epsilon: f32::EPSILON,
        }
    }
}

impl PolyfitCfg<f64> {
    pub const fn new() -> Self {
        Self {
            max_deg: u8::MAX,
            halt_epsilon: f64::EPSILON,
        }
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
            wv: T::SimdT::ZERO,
            xv: T::SimdT::ZERO,
            yvs: [T::SimdT::ZERO; R],
            p_km1v: T::SimdT::ZERO,
            p_kv: T::SimdT::ONE,
        };

        let mut data_len = 0;
        let mut lane_i = 0;
        let mut weight_sum = T::SimdT::ZERO;
        for (w, x, ys) in iter {
            unsafe {
                *curr_data.wv.as_mut_slice().get_unchecked_mut(lane_i) = w;
                *curr_data.xv.as_mut_slice().get_unchecked_mut(lane_i) = x;

                for (y, yv) in ys.into_iter().zip(curr_data.yvs.iter_mut()) {
                    *yv.as_mut_slice().get_unchecked_mut(lane_i) = y;
                }
            }
            lane_i += 1;

            if lane_i < T::SimdT::LANES.get() {
                continue;
            }

            weight_sum += curr_data.wv;
            data_list.push(ws, curr_data);
            data_len += T::SimdT::LANES.get();
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
                    .fill(T::ZERO);
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

        let weight_sum_recip = T::SimdT::splat(T::ONE / weight_sum);

        let (d_0s, gamma_0, b_0) = {
            let mut d_0vs = [T::SimdT::ZERO; R];
            let mut gamma_0v = T::SimdT::ZERO;
            let mut b_0v = T::SimdT::ZERO;

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

        // Each iteration the highest element of each dimension is overwritten, not modified. As such
        // we need not worry about stale non-zero values still being present.
        self.coeffs.set_len(max_coeffs);

        let mut p_k = BVec::with_capacity_in(max_coeffs, ws);
        p_k.push(T::ONE);
        let mut p_km1 = BVec::with_capacity_in(max_coeffs, ws);

        for r_i in 0..R {
            unsafe {
                let d_0 = *d_0s.get_unchecked(r_i) / gamma_0;
                *self.coeffs.dim_unchecked_mut(r_i).get_unchecked_mut(0) = d_0;
            }
        }

        let mut gamma_km1 = gamma_0;
        let mut minus_b_km1 = -b_0 / gamma_0;
        let mut minus_c_km1 = T::ZERO;
        let mut d_ks = d_0s;

        for k in 1..max_coeffs {
            let (gamma_k, b_k) = {
                let mut d_kvs = [T::SimdT::ZERO; R];
                let mut gamma_kv = T::SimdT::ZERO;
                let mut b_kv = T::SimdT::ZERO;
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
            p_k.push(T::ZERO);
            for i in 0..k {
                let p_ki = unsafe { p_k.get_unchecked_mut(i) };
                let p_km1i = unsafe { p_km1.get_unchecked_mut(i) };

                *p_ki = minus_c_km1.mul_add(*p_ki, minus_b_km1 * (*p_km1i));
            }
            p_k.push(T::ZERO);
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
