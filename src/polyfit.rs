use core::cmp::{max, min};
use core::mem::{self, MaybeUninit};
use core::ops::DerefMut;
use core::{ptr, slice};

use bytemuck::{Zeroable, zeroed};

use crate::storage::{Fit, FitErrors, FitResult, YxlkSums};
use crate::{
    SPolynomial,
    simd::SimdAble,
    storage::{KP1Array, TwoKP1Array, XlkSums},
};

unsafe fn shift_power_sums<T: SimdAble>(
    power_sums: &mut [T],
    power_sums_ref_store: &mut [T],
    coeffs: &mut [T],
    delta_x: T,
) {
    unsafe {
        ptr::copy_nonoverlapping(
            power_sums.as_ptr(),
            power_sums_ref_store.as_mut_ptr(),
            power_sums.len(),
        );
        *coeffs.get_unchecked_mut(0) = T::SF_ONE;
        let mut k = 1;
        while k < power_sums.len() {
            let mut shifted_k_sum = *power_sums.get_unchecked_mut(k);
            *coeffs.get_unchecked_mut(k) = T::SF_ONE;
            for j in (1..k).rev() {
                let prev_coeff = *coeffs.get_unchecked(j - 1);
                let curr_coeff = coeffs.get_unchecked_mut(j);
                *curr_coeff = curr_coeff.mul_add(delta_x, prev_coeff);

                shifted_k_sum = T::mul_add(
                    *curr_coeff,
                    *power_sums_ref_store.get_unchecked(j),
                    shifted_k_sum,
                );
            }

            let curr_coeff = coeffs.get_unchecked_mut(0);
            *curr_coeff *= delta_x;

            *power_sums.get_unchecked_mut(k) = T::mul_add(
                *curr_coeff,
                *power_sums_ref_store.get_unchecked(0),
                shifted_k_sum,
            );

            k += 1;
        }
    }
}

#[derive(Clone)]
#[repr(C)]
pub struct OnlinePolyfit<T: SimdAble, const K: usize, const D: usize = 1> {
    factorials_1_up: [T; K],
    xlks: XlkSums<T, K>,
    /// Y_1[x^<array index>]
    yxlks: [YxlkSums<T, K>; D],
    /// Sum of all w_(l, i) y_(l, i)^2 for error calculation.
    yys: [T; D],
    max_l_insertion: usize,
}

impl<T: SimdAble, const K: usize, const D: usize> OnlinePolyfit<T, K, D> {
    pub fn new() -> Self {
        let mut factorial_value: T = T::SF_ONE;
        let mut mult: T = T::SF_ONE;
        Self {
            factorials_1_up: [(); K].map(|_| {
                factorial_value *= mult;
                mult += T::SF_ONE;

                factorial_value
            }),
            xlks: XlkSums::zeroed(),
            yxlks: unsafe { MaybeUninit::zeroed().assume_init() },
            yys: unsafe { MaybeUninit::zeroed().assume_init() },
            max_l_insertion: 0,
        }
    }

    /// Shift the regression state from `x'` to `x = x' + delta_x`, effectively shifting the domain of the
    /// regression to the left by `delta_x`. Practically, if you had inserted at `x_i` previously, the shift would
    /// make it as though you inserted at `x_i + delta_x` instead and the fit after the shift (`P(x)`) relates
    /// to the fit before the shift (`P'(x')`) as follows:
    ///
    ///  `P(x) = P'(x - delta_x)`
    pub fn shift(&mut self, delta_x: T) {
        let mut old_sums = TwoKP1Array::<T, K>::zeroed();
        let mut coeffs = TwoKP1Array::<T, K>::zeroed();

        // Iterating in reverse order guarantees that we are stepping through the `xlk` and `yxlk` elements in
        // storage order. `l=K` also only contains the zoroeth degree powers and does not need to be considered.
        for l in (0..K).rev() {
            unsafe {
                let xlks = self.xlks.get_l_xks_mut(l);
                // // Safety: xlks has (K - l) * 2 + 1 elements.
                // let rescale = T::exp2(T::round(
                //     (T::log2(*xlks.get_unchecked(2))) / T::from_usize(2),
                // ));
                // let rescale_recip = rescale.recip();
                // for xlk in xlks.iter_mut() {
                //     *xlk *= rescale;
                // }
                shift_power_sums(xlks, &mut old_sums, &mut coeffs, delta_x);
                // for xlk in xlks.iter_mut() {
                //     *xlk *= rescale_recip;
                // }

                for yxks in &mut self.yxlks {
                    let yxlks = yxks.get_l_yxks_mut(l);
                    // for yxlk in yxlks.iter_mut() {
                    //     *yxlk *= rescale;
                    // }
                    shift_power_sums(yxlks, &mut old_sums, &mut coeffs, delta_x);
                    // for yxlk in yxlks.iter_mut() {
                    //     *yxlk *= rescale_recip;
                    // }
                }
            }
        }
    }

    /// Scale all sample weights by `scale`.
    pub fn scale(&mut self, scale: T) {
        for xlk in self.xlks.as_raw_slice_mut() {
            *xlk *= scale;
        }

        for yxks in &mut self.yxlks {
            for v in yxks.as_raw_slice_mut() {
                *v *= scale;
            }
        }

        for yys in &mut self.yys {
            *yys *= scale;
        }
    }

    pub fn update_at_zero(&mut self, derivative: usize, w: T, ys: [T; D]) {
        let l = derivative;
        if l > K {
            return;
        }
        self.max_l_insertion = max(self.max_l_insertion, l);

        unsafe {
            *self.xlks.get_l_xk_mut(l, 0) += w;
            for (d, y) in ys.into_iter().enumerate() {
                let yxl0 = self.yxlks.get_unchecked_mut(d).get_l_yxk_mut(l, 0);
                *yxl0 = w.mul_add(y, *yxl0);

                let yys = self.yys.get_unchecked_mut(d);
                *yys = w.mul_add(y * y, *yys);
            }
        }
    }

    pub fn update(&mut self, derivative: usize, w: T, x: T, ys: [T; D]) {
        let l = derivative;
        if l > K {
            return;
        }
        self.max_l_insertion = max(self.max_l_insertion, l);

        let xks = unsafe { self.xlks.get_l_xks_mut(l) };
        let mut x_pow = T::SF_ONE;
        for xk in xks {
            *xk = w.mul_add(x_pow, *xk);
            x_pow *= x;
        }
        for (d, y) in ys.into_iter().enumerate() {
            let mut x_pow = T::SF_ONE;
            for yxlk in unsafe { self.yxlks.get_unchecked_mut(d).get_l_yxks_mut(l) } {
                *yxlk = w.mul_add(y * x_pow, *yxlk);
                x_pow *= x;
            }

            let yys = unsafe { self.yys.get_unchecked_mut(d) };
            *yys = w.mul_add(y * y, *yys);
        }
    }

    /// Returns the sum of all sample weights:
    /// $$\sum_{l=0}^{K}\sum_{i=1}^{N_l} w_{l, i}$$
    pub fn weight(&self) -> T {
        unsafe {
            let mut sum = *self.xlks.get_l_xk(K, 0);
            for l in (0..K).rev() {
                sum += *self.xlks.get_l_xk(l, 0);
            }

            sum
        }
    }

    /// Returns the total squared error
    /// which will be observed by the zero polynomial.
    pub fn addressable_error(&self) -> [T; D] {
        self.yys
    }

    fn safe_rescale_coeff(&self) -> T {
        return T::SF_ONE;
        if K == 0 {
            return T::SF_ONE;
        }

        // Safety: For all `l < K` the power sums go up to at least `k=2`.
        unsafe {
            let mut x0s = *self.xlks.get_l_xk(0, 0);
            let mut x2s = *self.xlks.get_l_xk(0, 2);

            for l in 1..(self.max_l_insertion + 1).min(K) {
                x0s += *self.xlks.get_l_xk(l, 0);
                x2s += *self.xlks.get_l_xk(l, 2);
            }

            let rescale = T::exp2(T::round((T::log2(x0s) - T::log2(x2s)) / T::from_usize(2)));

            if T::is_finite(&rescale) {
                rescale
            } else {
                T::SF_ONE
            }
        }
    }

    fn compute_fit_inner(&self, bias: KP1Array<(T, [T; D]), K>) -> [FitResult<T, K>; D] {
        let mut fit_res = self.yys.map(|yys| {
            let mut errors = FitErrors::zeroed();
            unsafe { *errors.errors_mut().get_unchecked_mut(0) = yys }
            FitResult::<T, K> {
                fit: Fit::zeroed(),
                errors,
            }
        });

        if D == 0 {
            // Why are you the way that you are.
            return fit_res;
        }

        unsafe {
            let rescale = self.safe_rescale_coeff();
            let rescale_recip = rescale.recip();
            let rescale_recip_sq = rescale_recip * rescale_recip;
            let w_b = bias.get_unchecked(0).0;
            let gamma_0 = (*self.xlks.get_l_xk(0, 0) + w_b).max(w_b);
            let gamma_0_recip = gamma_0.recip();
            // <x^k, x^{k_prime}>
            let mut inner_products =
                MaybeUninit::<KP1Array<KP1Array<T, K>, K>>::zeroed().assume_init();
            {
                *inner_products.get_unchecked_mut(0).get_unchecked_mut(0) = gamma_0;
                let mut rescale_k = T::SF_ONE;
                for k in 1..(K + 1) {
                    rescale_k *= rescale;

                    // 2^{r(k + k_prime)}
                    let mut rescale_kkp = rescale_k;
                    for k_prime in 0..=k {
                        let kk_prime = k + k_prime;

                        let mut left = T::from_usize(k);
                        let mut right = T::from_usize(k_prime);
                        let mut mul = rescale_kkp;
                        let mut xkxk_prime = *self.xlks.get_l_xk(0, kk_prime) * rescale_kkp;
                        for l in 1..(k_prime.min(self.max_l_insertion) + 1) {
                            mul *= left * right * rescale_recip_sq;
                            left -= T::SF_ONE;
                            right -= T::SF_ONE;

                            xkxk_prime = self
                                .xlks
                                .get_l_xk(l, kk_prime - l - l)
                                .mul_add(mul, xkxk_prime);
                        }

                        *inner_products
                            .get_unchecked_mut(k)
                            .get_unchecked_mut(k_prime) = xkxk_prime;
                        *inner_products
                            .get_unchecked_mut(k_prime)
                            .get_unchecked_mut(k) = xkxk_prime;

                        rescale_kkp *= rescale;
                    }
                }
            }

            let mut yxks = zeroed::<[KP1Array<T, K>; D]>();
            for d in 0..D {
                let yxlks = self.yxlks.get_unchecked(d);

                let yxks = yxks.get_unchecked_mut(d);
                *yxks.get_unchecked_mut(0) = *yxlks.get_l_yxk(0, 0);

                let mut rescale_k = T::SF_ONE;
                for k in 1..(K + 1) {
                    rescale_k *= rescale;

                    let mut right = T::from_usize(k);
                    let mut mul = rescale_k;
                    let mut yxk = *yxlks.get_l_yxk(0, k) * rescale_k;
                    for l in 1..(k.min(self.max_l_insertion) + 1) {
                        mul *= right * rescale_recip;
                        right -= T::SF_ONE;

                        yxk = yxlks.get_l_yxk(l, k - l).mul_add(mul, yxk);
                    }

                    *yxks.get_unchecked_mut(k) = yxk;
                }
            }

            let bias_0 = bias.get_unchecked(0);
            for dim in (0..D).rev() {
                let yxks_dim = yxks.get_unchecked_mut(dim);
                let yx0_dim = yxks_dim.get_unchecked_mut(0);
                let fit_res_dim = fit_res.get_unchecked_mut(dim);
                let gamma_d_0 = bias_0.0.mul_add(*bias_0.1.get_unchecked(dim), *yx0_dim);
                let d_0 = gamma_d_0 * gamma_0_recip;

                // Subtracting from Y_1[x^k] here makes zero mathematical differrence since:
                // <y(x), P_j> = <y(x) - d_k P_k, P_j>    j != k
                // but makes the fit more stable at higher dimensions when the polynomials are not exactly orthogonal.
                // Specifically, since we calculate <y(x), P_j> as Y_1[P_j], if, at every k we have for all k':
                // Y'_1[x^k'] = <y(x) - d_0 P_0 ... - d_(k-1) P_(k-1), x^k'> = <y(x) - d_0 P_0 ... - d_min(k-1, k') P_min(k-1, k'), x^k'>
                // then Y'_1[P_k] = Y_1[P_k].
                *yx0_dim = T::SF_ZERO;
                let inner_products = inner_products.get_unchecked(0);
                for k_upper in 1..(K + 1) {
                    let yxks_dim_ku = yxks_dim.get_unchecked_mut(k_upper);
                    *yxks_dim_ku = inner_products
                        .get_unchecked(k_upper)
                        .mul_add(-d_0, *yxks_dim_ku);
                }

                // Compute zeroeth degree error sum of all w_(l, i) [y_(l, i) - P_0^(l)(x_(l, i))]^2. (P_0^(l) here is shorthand for the
                // l-th derivative of P_0).
                let err = fit_res_dim.errors.errors_mut().get_unchecked_mut(0);
                *err = d_0.mul_add(-gamma_d_0, *err);

                *fit_res_dim.fit.get_pk_i_mut(0, 0) = d_0;
            }

            {
                // MGS Procedure for larger k's. We abuse the storage of the zeroeth-dimension's
                // polynomials to accomplish this.
                let upper_coeffs = &mut fit_res.get_unchecked_mut(0).fit;
                let inner_products = inner_products.get_unchecked(0);
                for k_larger in 1..(K + 1) {
                    let xkp0 = *inner_products.get_unchecked(k_larger);
                    *upper_coeffs.get_pk_i_mut(k_larger, 0) = -xkp0 * gamma_0_recip;
                }
            }

            // [<x^0, P_k>, ... , <x^K, P_k>]
            let mut xkpk = KP1Array::<T, K>::zeroed();
            // P_k - x^k (P_k without the x^k term).
            let mut p_k = KP1Array::<T, K>::zeroed();
            // k!
            let mut bias_factor = T::SF_ONE;
            for k in 1..=K {
                ptr::copy_nonoverlapping(
                    fit_res.get_unchecked(0).fit.get_pk(k).as_ptr(),
                    p_k.as_mut_ptr(),
                    k,
                );

                println!("{p_k:?}");

                for k_left in (0..=K).rev() {
                    let inner_products = inner_products.get_unchecked(k_left);
                    let xkpk = xkpk.get_unchecked_mut(k_left);
                    *xkpk = *inner_products.get_unchecked(k);
                    for k_right in (0..k).rev() {
                        *xkpk = inner_products
                            .get_unchecked(k_right)
                            .mul_add(*p_k.get_unchecked(k_right), *xkpk);
                    }
                }

                let mut gamma_k = *xkpk.get_unchecked(k);
                for k_prime in (0..k).rev() {
                    gamma_k = p_k
                        .get_unchecked(k_prime)
                        .mul_add(*xkpk.get_unchecked(k_prime), gamma_k);
                }

                bias_factor *= T::from_usize(k);
                let bias_k = bias.get_unchecked(k);
                let w_b = bias_k.0;
                gamma_k = (bias_factor * bias_factor).mul_add(w_b, gamma_k).max(w_b);

                let gamma_k_recip = gamma_k.recip();

                for dim in (0..D).rev() {
                    let yxks_dim = yxks.get_unchecked_mut(dim);
                    let fit_res_dim = fit_res.get_unchecked_mut(dim);
                    fit_res_dim.fit.transfer_km1_k(k);
                    let fit_pk = fit_res_dim.fit.get_pk_mut(k);

                    let yxk_dim = yxks_dim.get_unchecked_mut(k);
                    *yxk_dim = w_b.mul_add(bias_factor * *bias_k.1.get_unchecked(dim), *yxk_dim);

                    let mut gamma_d_k = *yxk_dim;
                    for i in (0..k).rev() {
                        gamma_d_k = yxks_dim
                            .get_unchecked(i)
                            .mul_add(*p_k.get_unchecked(i), gamma_d_k)
                    }
                    let d_k = gamma_d_k * gamma_k_recip;

                    // Subtract the resulting fit from Y_1(x^j) for all j to gradually decrease their
                    // overall values as k becomes larger.
                    for k_upper in 0..=K {
                        let yxks_dim_k = yxks_dim.get_unchecked_mut(k_upper);
                        *yxks_dim_k = d_k.mul_add(-*xkpk.get_unchecked(k_upper), *yxks_dim_k);
                    }

                    // Transfer and refine error sum of all w_(l, i) [y_(l, i) - P^(l)(x_(l, i))]^2.
                    let err = fit_res_dim.errors.error(k - 1);
                    *fit_res_dim.errors.errors_mut().get_unchecked_mut(k) =
                        d_k.mul_add(-gamma_d_k, err);

                    for k_prime in 0..k {
                        let fit_dim_k = fit_pk.get_unchecked_mut(k_prime);
                        *fit_dim_k = d_k.mul_add(*p_k.get_unchecked(k_prime), *fit_dim_k);
                    }

                    *fit_pk.get_unchecked_mut(k) = d_k;
                }

                {
                    // MGS Procedure for larger k's. We abuse the storage of the zeroeth-dimension's
                    // polynomials to accomplish this.
                    let upper_coeffs = &mut fit_res.get_unchecked_mut(0).fit;
                    for k_upper in (k + 1)..(K + 1) {
                        let pk_upper = upper_coeffs.get_pk_mut(k_upper);
                        let mut c = *xkpk.get_unchecked(k_upper);
                        for k_prime in 0..k {
                            c = pk_upper
                                .get_unchecked(k_prime)
                                .mul_add(*xkpk.get_unchecked(k_prime), c);
                        }
                        c = -c * gamma_k_recip;
                        *pk_upper.get_unchecked_mut(k) += c;
                        for k_prime in 0..=k {
                            let pk_upper_kp = pk_upper.get_unchecked_mut(k_prime);

                            *pk_upper_kp = p_k.get_unchecked(k_prime).mul_add(c, *pk_upper_kp);
                        }
                    }
                }
            }

            for fit_res in fit_res.iter_mut() {
                for k in 1..(K + 1) {
                    let fit = fit_res.fit.get_pk_mut(k);
                    let mut mul = T::SF_ONE;
                    for i in 1..=k {
                        mul *= rescale;
                        *fit.get_unchecked_mut(i) *= mul;
                    }
                }
            }
        }

        fit_res
    }

    #[inline]
    pub fn compute_fit(&self) -> [FitResult<T, K>; D] {
        self.compute_fit_inner(zeroed())
    }

    #[inline]
    pub fn compute_fit_with_bias(&self, bias: KP1Array<(T, [T; D]), K>) -> [FitResult<T, K>; D] {
        self.compute_fit_inner(bias)
    }
}
