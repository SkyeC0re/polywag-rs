use core::mem::{self, MaybeUninit};

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
    yxks: [KP1Array<T, K>; D],
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
            max_l_insertion: 0,
        }
    }

    /// Update the regression state from `x'` to `x = x' + delta_x`, effectively shifting the domain of the
    /// regression to the left by `delta_x`.
    pub fn rotate(&mut self, delta_x: T) {
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
    #[inline]
    pub fn scale(&mut self, scale: T) {
        for xlk in self.xlks.as_raw_slice_mut() {
            *xlk *= scale;
        }

        for yxks in &mut self.yxks {
            for yxk in yxks.iter_mut() {
                *yxk *= scale;
            }
        }
    }

    pub fn update_at_zero(&mut self, derivative: usize, w: T, ys: [T; D]) {
        let l = derivative;
        if l > K {
            return;
        }

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
        }

        self.max_l_insertion = l;
    }

    pub fn update(&mut self, derivative: usize, w: T, x: T, ys: [T; D]) {
        let l = derivative;
        if l > K {
            return;
        }

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
        }

        self.max_l_insertion = l;
    }

    #[inline]
    pub fn compute_fit(&self) -> [SPolynomial<T, K>; D] {
        Self::compute_fit_inner(&self.xlks, &mut self.yxks.clone())
    }

    fn compute_fit_inner(
        xlks: &XlkSums<T, K>,
        yxks: &mut [KP1Array<T, K>; D],
    ) -> [SPolynomial<T, K>; D] {
        let mut fit = [(); D].map(|_| SPolynomial::<T, K>::new());

        let mut p_km1 = KP1Array::<T, K>::zeroed();
        let mut p_km1 = &mut p_km1;

        let mut p_k = KP1Array::<T, K>::zeroed();
        let mut p_k = &mut p_k;

        unsafe {
            let gamma_0 = *xlks.get_l_xk(0, 0);
            let gamma_0_recip = gamma_0.recip();

            for dim in 0..D {
                let yxks_dim = yxks.get_unchecked_mut(dim);
                let yx0_dim = yxks_dim.get_unchecked_mut(0);
                let d_0 = *yx0_dim * gamma_0_recip;

                // Subtracting from Y_1[x^k] here makes zero mathematical differrence since:
                // <y(x), P_j> = <y(x) - d_k P_k, P_j>    j != k
                // but makes the fit more stable at higher dimensions when the polynomials are not exactly orthogonal.
                *yx0_dim = T::SF_ZERO;

                *fit.get_unchecked_mut(dim).get_unchecked_mut(0) = d_0;
            }

            let mut min_c_km1 = T::SF_ZERO;
            let mut gamma_km1_recip = gamma_0_recip;
            *p_km1.get_unchecked_mut(0) = T::SF_ONE;

            // [<x^0, x^k>, ... , <x^(k-1), x^k>]
            let mut inner_products_km1 = [T::SF_ZERO; K];
            for k in 1..=K {
                // B_(k-1) = <xP_(k-1), P_(k-1)> / gamma_(k-1)
                //     = <x^k + [x^(k-2)]P_(k-1) x^(k-2), P_(k-1)>
                //     = <x^k, P_(k-1)> / gamma_(k-1) + [x^(k-2)]P_(k-1)

                // <x^k, [x^0]P_(k-1) x^0> contains only the l=0 term. We abuse this to
                // provide a direct initial value.
                let mut min_b_km1 = *p_km1.get_unchecked(0) * *xlks.get_l_xk(0, k);

                for k_prime in 1..k {
                    let mut a = k_prime;
                    let mut b = k;
                    let mut mul_coeff = 1;

                    let max_deg = k + k_prime;
                    min_b_km1 = p_km1
                        .get_unchecked(k_prime)
                        .mul_add(*xlks.get_l_xk(0, max_deg), min_b_km1);

                    for l in 1..=k_prime {
                        mul_coeff *= a * b;
                        a -= 1;
                        b -= 1;
                        min_b_km1 = p_km1.get_unchecked(k_prime).mul_add(
                            T::from_usize(mul_coeff) * *xlks.get_l_xk(l, max_deg - (l << 1)),
                            min_b_km1,
                        );
                    }
                }
                if k > 1 {
                    min_b_km1 = min_b_km1.mul_add(gamma_km1_recip, *p_km1.get_unchecked(k - 2));
                } else {
                    min_b_km1 *= gamma_km1_recip;
                }
                min_b_km1 = -min_b_km1;

                let pkm1_0 = *p_km1.get_unchecked(0);
                let pk_0 = p_k.get_unchecked_mut(0);
                *pk_0 = pk_0.mul_add(min_c_km1, min_b_km1 * pkm1_0);

                let x0ks = *xlks.get_l_xk(0, k);
                // gamma_k = <P_k, P_k> = <x^k, P_k>
                // <x^k, [x^0]P_k x^0> contains only the l=0 term. We abuse this to
                // provide a direct initial value.
                let mut gamma_k = *pk_0 * x0ks;

                *inner_products_km1.get_unchecked_mut(0) = x0ks;

                for k_prime in 1..k {
                    let [pkm1_im1, pkm1_i] =
                        *mem::transmute::<_, &[T; 2]>(p_km1.get_unchecked(k_prime - 1));
                    let pkm1_contribution = pkm1_i.mul_add(min_b_km1, pkm1_im1);

                    let pk_i = p_k.get_unchecked_mut(k_prime);
                    *pk_i = pk_i.mul_add(min_c_km1, pkm1_contribution);

                    let mut a = k_prime;
                    let mut b = k;
                    let mut mul_coeff = 1;
                    let max_deg = k + k_prime;
                    let x0ks = *xlks.get_l_xk(0, max_deg);
                    gamma_k = pk_i.mul_add(*xlks.get_l_xk(0, max_deg), gamma_k);
                    let inner_product_k_prime_k = inner_products_km1.get_unchecked_mut(k_prime);
                    *inner_product_k_prime_k = x0ks;

                    for l in 1..=k_prime {
                        mul_coeff *= a * b;
                        a -= 1;
                        b -= 1;
                        let l_contrib =
                            T::from_usize(mul_coeff) * *xlks.get_l_xk(l, max_deg - (l << 1));
                        gamma_k = pk_i.mul_add(l_contrib, gamma_k);

                        *inner_product_k_prime_k += l_contrib;
                    }
                }

                *p_k.get_unchecked_mut(k) = T::SF_ONE;
                {
                    let mut a = k;
                    let mut mul_coeff = 1;
                    let max_deg = k << 1;

                    gamma_k += *xlks.get_l_xk(0, max_deg);
                    for l in 1..=k {
                        mul_coeff *= a * a;
                        a -= 1;
                        gamma_k = T::from_usize(mul_coeff)
                            .mul_add(*xlks.get_l_xk(l, max_deg - (l << 1)), gamma_k);
                    }
                }

                let gamma_k_recip = gamma_k.recip();
                for dim in 0..D {
                    let yxks_dim = yxks.get_unchecked_mut(dim);
                    let fit_dim = fit.get_unchecked_mut(dim);

                    {
                        // We subtract the accumulated <x^k, d_0 P_0> , ... , <x^k, d_(k-1) P_(k-1)> here as part of the
                        // <y(x), P_j> = <y(x) - d_k P_k, P_j>    j != k
                        // strategy for Y_1[x^k].
                        let yxks_dim_k = yxks_dim.get_unchecked_mut(k);
                        for k_prime in 0..k {
                            *yxks_dim_k = T::mul_add(
                                -*fit_dim.get_unchecked(k_prime),
                                *inner_products_km1.get_unchecked(k_prime),
                                *yxks_dim_k,
                            );
                        }
                    }

                    let mut d_k = *yxks_dim.get_unchecked(k);
                    for i in (0..k).rev() {
                        d_k = yxks_dim
                            .get_unchecked(i)
                            .mul_add(*p_k.get_unchecked(i), d_k)
                    }
                    // Here d_k is actually still gamma_k * d_k, which we subtract to complete the the
                    // <y(x), P_j> = <y(x) - d_k P_k, P_j>    j != k
                    // strategy for Y_1[x^k].
                    *yxks_dim.get_unchecked_mut(k) -= d_k;

                    d_k *= gamma_k_recip;

                    for k_prime in 0..k {
                        let fit_dim_k = fit_dim.get_unchecked_mut(k_prime);
                        *fit_dim_k = d_k.mul_add(*p_k.get_unchecked(k_prime), *fit_dim_k);
                    }

                    *fit_dim.get_unchecked_mut(k) = d_k;
                }

                min_c_km1 = -gamma_km1_recip * gamma_k;
                gamma_km1_recip = gamma_k_recip;
                mem::swap::<&mut KP1Array<T, K>>(&mut p_k, &mut p_km1);
            }
        }

        fit
    }

    #[inline]
    pub fn compute_fit_with_bias(
        &self,
        scale: T,
        rotate: T,
        bias: [(T, [T; D]); K],
    ) -> [SPolynomial<T, K>; D] {
        let mut mutable_data = self.clone();
        if scale != T::SF_ONE {
            mutable_data.scale(scale);
        }

        if rotate != T::SF_ZERO {
            mutable_data.rotate(rotate);
        }

        for (deriv, (w, ys)) in bias.into_iter().enumerate() {
            mutable_data.update_at_zero(deriv, w, ys);
        }

        Self::compute_fit_inner(&mutable_data.xlks, &mut mutable_data.yxks)
    }
}
