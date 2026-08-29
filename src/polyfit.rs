use core::cmp::{max, min};
use core::mem::{self, MaybeUninit};
use core::ptr;

use crate::storage::{Fit, FitErrors, FitResult};
use crate::{
    SPolynomial,
    simd::SimdAble,
    storage::{KP1Array, TwoKP1Array, XlkSums},
};

#[derive(Clone)]
#[repr(C)]
pub struct OnlinePolyfit<T: SimdAble, const K: usize, const D: usize = 1> {
    factorials_1_up: [T; K],
    xlks: XlkSums<T, K>,
    /// Y_1[x^<array index>]
    yxks: [KP1Array<T, K>; D],
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
            yxks: unsafe { MaybeUninit::zeroed().assume_init() },
            yys: unsafe { MaybeUninit::zeroed().assume_init() },
            max_l_insertion: 0,
        }
    }

    /// Update the regression state from `x'` to `x = x' + delta_x`, effectively shifting the domain of the
    /// regression to the left by `delta_x`.
    pub fn shift(&mut self, delta_x: T) {
        let old_xlks = self.xlks.clone();
        let mut coeffs = TwoKP1Array::<T, K>::zeroed();

        // Iterating in reverse order guarantees that we are stepping through the `xlk` elements in
        // storage order.
        for l in (0..=K).rev() {
            unsafe {
                *coeffs.get_unchecked_mut(0) = T::SF_ONE;
                let xlks = self.xlks.get_l_xks_mut(l);
                let old_xlks = old_xlks.get_l_xks(l);
                for k in 1..(((K - l) << 1) + 1) {
                    let xlk = xlks.get_unchecked_mut(k);
                    *coeffs.get_unchecked_mut(k) = T::SF_ONE;
                    for j in (1..k).rev() {
                        let prev_coeff = *coeffs.get_unchecked(j - 1);
                        let curr_coeff = coeffs.get_unchecked_mut(j);
                        *curr_coeff = curr_coeff.mul_add(delta_x, prev_coeff);

                        *xlk = T::mul_add(*curr_coeff, *old_xlks.get_unchecked(j), *xlk);
                    }

                    let curr_coeff = coeffs.get_unchecked_mut(0);
                    *curr_coeff *= delta_x;

                    *xlk = T::mul_add(*curr_coeff, *old_xlks.get_unchecked(0), *xlk);
                }
            }
        }

        let mut coeffs = KP1Array::<T, K>::zeroed();
        for d in 0..D {
            let yxks = &mut self.yxks[d];
            let old_yxks = yxks.clone();

            unsafe {
                *coeffs.get_unchecked_mut(0) = T::SF_ONE;
                for k in 1..KP1Array::<T, K>::LEN {
                    let yxk = yxks.get_unchecked_mut(k);
                    *coeffs.get_unchecked_mut(k) = T::SF_ONE;
                    for j in (1..k).rev() {
                        let prev_coeff = *coeffs.get_unchecked(j - 1);
                        let curr_coeff = coeffs.get_unchecked_mut(j);
                        *curr_coeff = curr_coeff.mul_add(delta_x, prev_coeff);

                        *yxk = T::mul_add(*curr_coeff, *old_yxks.get_unchecked(j), *yxk);
                    }

                    let curr_coeff = coeffs.get_unchecked_mut(0);
                    *curr_coeff *= delta_x;

                    *yxk = T::mul_add(*curr_coeff, *old_yxks.get_unchecked(0), *yxk);
                }
            }
        }
    }

    /// Scale all sample weights by `scale`.
    pub fn scale(&mut self, scale: T) {
        for xlk in self.xlks.as_raw_slice_mut() {
            *xlk *= scale;
        }

        for yxks in &mut self.yxks {
            for yxk in yxks.iter_mut() {
                *yxk *= scale;
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

        let factorial = if l == 0 {
            T::SF_ONE
        } else {
            self.factorials_1_up[l - 1]
        };

        let w_factorial = w * factorial;
        unsafe { *self.xlks.get_l_xk_mut(l, 0) += w }

        for (dim, y) in ys.into_iter().enumerate() {
            let yxk = unsafe { self.yxks.get_unchecked_mut(dim).get_unchecked_mut(l) };
            *yxk = T::mul_add(w_factorial, y, *yxk);

            let yys = unsafe { self.yys.get_unchecked_mut(dim) };
            *yys = w.mul_add(y * y, *yys);
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

        let l_factorial = if l == 0 {
            T::SF_ONE
        } else {
            self.factorials_1_up[l - 1]
        };

        for (dim, y) in ys.into_iter().enumerate() {
            let yxk_dim = unsafe { self.yxks.get_unchecked_mut(dim) };

            let mut factorial = l_factorial;
            let mut x_pow = T::SF_ONE;
            let mut k = l;
            for yxk in unsafe { yxk_dim.get_unchecked_mut(l..=K) } {
                let w_factorial = w * factorial;
                *yxk = w_factorial.mul_add(y * x_pow, *yxk);
                k += 1;
                factorial = (factorial / T::from_usize(k - l)) * T::from_usize(k);
                x_pow *= x;
            }

            let yys = unsafe { self.yys.get_unchecked_mut(dim) };
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

    fn compute_fit_inner(
        xlks: &XlkSums<T, K>,
        yxks: &mut [KP1Array<T, K>; D],
        yys: [T; D],
        max_l: usize,
    ) -> [FitResult<T, K>; D] {
        let mut fit_res = yys.map(|yys| {
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
            let gamma_0 = *xlks.get_l_xk(0, 0);
            let gamma_0_recip = gamma_0.recip();
            // <x^k, x^{k_prime}>
            let mut inner_products =
                MaybeUninit::<KP1Array<KP1Array<T, K>, K>>::zeroed().assume_init();
            {
                *inner_products.get_unchecked_mut(0).get_unchecked_mut(0) = gamma_0;
                for k in 1..(K + 1) {
                    for k_prime in 0..=k {
                        let mut left = T::from_usize(k);
                        let mut right = T::from_usize(k_prime);
                        let mut mul = T::SF_ONE;

                        let kk_prime = k + k_prime;
                        let mut xkxk_prime = *xlks.get_l_xk(0, kk_prime);

                        for l in 1..(k_prime + 1) {
                            mul *= left * right;
                            left -= T::SF_ONE;
                            right -= T::SF_ONE;

                            xkxk_prime =
                                xlks.get_l_xk(l, kk_prime - l - l).mul_add(mul, xkxk_prime);
                        }

                        *inner_products
                            .get_unchecked_mut(k)
                            .get_unchecked_mut(k_prime) = xkxk_prime;
                        *inner_products
                            .get_unchecked_mut(k_prime)
                            .get_unchecked_mut(k) = xkxk_prime;
                    }
                }
            }

            for dim in (0..D).rev() {
                let yxks_dim = yxks.get_unchecked_mut(dim);
                let yx0_dim = yxks_dim.get_unchecked_mut(0);
                let fit_res_dim = fit_res.get_unchecked_mut(dim);
                let gamma_d_0 = *yx0_dim;
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
                    *upper_coeffs.get_pk_i_mut(k_larger, 0) =
                        -*inner_products.get_unchecked(k_larger) * gamma_0_recip;
                }
            }

            // [<x^0, P_k>, ... , <x^K, P_k>]
            let mut xkpk = KP1Array::<T, K>::zeroed();
            // P_k - x^k (P_k without the x^k term).
            let mut p_k = KP1Array::<T, K>::zeroed();
            for k in 1..=K {
                ptr::copy_nonoverlapping(
                    fit_res.get_unchecked(0).fit.get_pk(k).as_ptr(),
                    p_k.as_mut_ptr(),
                    k,
                );

                for k_left in 0..=K {
                    let inner_products = inner_products.get_unchecked(k_left);
                    let xkpk = xkpk.get_unchecked_mut(k_left);
                    *xkpk = *inner_products.get_unchecked(k);
                    for k_right in (0..k).rev() {
                        *xkpk = inner_products
                            .get_unchecked(k_right)
                            .mul_add(*p_k.get_unchecked(k_right), *xkpk);
                    }
                }

                let gamma_k = *xkpk.get_unchecked(k);
                // for k_prime in (0..k).rev() {
                //     gamma_k = xkpk
                //         .get_unchecked(k_prime)
                //         .mul_add(*p_k.get_unchecked(k_prime), gamma_k)
                // }
                let gamma_k_recip = gamma_k.recip();

                for dim in (0..D).rev() {
                    let yxks_dim = yxks.get_unchecked_mut(dim);
                    let fit_res_dim = fit_res.get_unchecked_mut(dim);
                    fit_res_dim.fit.transfer_km1_k(k);
                    let fit_pk = fit_res_dim.fit.get_pk_mut(k);

                    let mut gamma_d_k = *yxks_dim.get_unchecked(k);
                    for i in (0..k).rev() {
                        gamma_d_k = yxks_dim
                            .get_unchecked(i)
                            .mul_add(*p_k.get_unchecked(i), gamma_d_k)
                    }
                    let d_k = gamma_d_k * gamma_k_recip;
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
                    for k_larger in (k + 1)..(K + 1) {
                        let pk_larger = upper_coeffs.get_pk_mut(k_larger);
                        let mut c = *xkpk.get_unchecked(k_larger);
                        for k_prime in 0..k {
                            c = pk_larger
                                .get_unchecked(k_prime)
                                .mul_add(*xkpk.get_unchecked(k_prime), c);
                        }
                        c *= -gamma_k_recip;

                        *pk_larger.get_unchecked_mut(k) += c;
                        for k_prime in 0..=k {
                            let pk_larger_kp = pk_larger.get_unchecked_mut(k_prime);

                            *pk_larger_kp = p_k.get_unchecked(k_prime).mul_add(c, *pk_larger_kp);
                        }
                    }
                }
            }
        }

        fit_res
    }

    #[inline]
    pub fn compute_fit(&self) -> [FitResult<T, K>; D] {
        Self::compute_fit_inner(
            &self.xlks,
            &mut self.yxks.clone(),
            self.yys,
            self.max_l_insertion,
        )
    }
}
