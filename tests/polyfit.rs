mod common;

use approx::{abs_diff_eq, assert_abs_diff_eq};
use core::ops::Add;
use pastey::paste;
use polywag::{OnlinePolyfit, simd::SimdField};

use crate::common::{F256, TestableSimd, assert_eps_diff_eq, eps_diff_eq, test_eps};

/// Tests a function to ensure a minimal least squares fit is produced (up to our selected epsilon).
fn online_minimal_fit<T: TestableSimd, const K: usize>(
    w: fn(T) -> T,
    y: fn(T) -> T,
    samples: usize,
) {
    let mut regressor = OnlinePolyfit::<T, K>::new();
    let mut f256_regressor = OnlinePolyfit::<F256, K>::new();
    let scaling = T::SF_ONE / T::from_usize(samples);

    for x in 0..=samples {
        let x = scaling * T::from_usize(x);
        let w = w(x);
        let y = y(x);
        regressor.update(0, w, x, [y]);
        f256_regressor.update(0, w.into(), x.into(), [y.into()]);
    }

    let [mut f256_poly] = f256_regressor
        .compute_fit()
        .each_ref()
        .map(|r| r.fit.max_deg());

    let avg_err = scaling.into()
        * (0..=samples)
            .into_iter()
            .map(|x| {
                let x = scaling * T::from_usize(x);
                let [y_p] = f256_poly.evaluate_array([x.into()]);
                let err = y(x).into() - y_p;
                w(x).into() * err * err
            })
            .reduce(F256::add)
            .unwrap_or(F256::SF_ZERO);

    let f256_eps: F256 = T::SF_EPS.into();
    for i in 0..f256_poly.len() {
        for jitter in [-f256_eps, f256_eps] {
            let p_bckp = f256_poly[i];
            f256_poly[i] += jitter;
            let jitter_avg_err = scaling.into()
                * (0..=samples)
                    .into_iter()
                    .map(|x| {
                        let x = scaling * T::from_usize(x);
                        let [y_p] = f256_poly.evaluate_array([x.into()]);
                        let err = y(x).into() - y_p;
                        w(x).into() * err * err
                    })
                    .reduce(F256::add)
                    .unwrap_or(F256::SF_ZERO);

            // For every jitter of `T` epsilon, it should be impossible to improve the `f256` fit (under the assumption that `f256`` is much more precise than `T`).
            assert!(
                avg_err <= jitter_avg_err,
                "Non optimal f256 fit produced. Improved from {avg_err:?} to {jitter_avg_err:?} when adding {jitter:?} to fit polynomial coefficient {i}"
            );
            f256_poly[i] = p_bckp;
        }
    }

    let [poly] = regressor.compute_fit().each_ref().map(|r| r.fit.max_deg());

    let poly_err = scaling.into()
        * (0..=samples)
            .into_iter()
            .map(|x| {
                let x = scaling * T::from_usize(x);
                let y_p = poly.evaluate_array([x])[0].into();
                let err = y(x).into() - y_p;
                w(x).into() * err * err
            })
            .reduce(F256::add)
            .unwrap_or(F256::SF_ZERO);

    let divergence = (avg_err - poly_err).abs();
    let eps = test_eps::<T>().into();
    assert!(
        (avg_err - poly_err).abs() < eps,
        "Result diverges from optimal fit solution: {divergence:?} >= {eps:?}"
    );
}

/// Least squares regressions should reproduce polynomials up to the maximum fitting degree exactly.
fn online_multi_dem_increasing_deg_fit<T: TestableSimd>() {
    let half = T::SF_ONE / T::from_usize(2);
    let quarter = T::SF_ONE / T::from_usize(4);

    let mut regressor = OnlinePolyfit::<T, 2, 3>::new();

    for x in 0..10 {
        let x = T::from_usize(x);

        regressor.update(
            0,
            T::SF_ONE,
            x,
            [
                T::SF_ONE,
                T::SF_ONE + half * x,
                T::SF_ONE + half * x + quarter * x * x,
            ],
        );
    }

    let poly = regressor.compute_fit().each_ref().map(|r| r.fit.max_deg());

    let eps = test_eps();
    assert_abs_diff_eq!(poly[0][0], T::SF_ONE, epsilon = eps);
    assert_abs_diff_eq!(poly[0][1], T::SF_ZERO, epsilon = eps);
    assert_abs_diff_eq!(poly[0][2], T::SF_ZERO, epsilon = eps);

    assert_abs_diff_eq!(poly[1][0], T::SF_ONE, epsilon = eps);
    assert_abs_diff_eq!(poly[1][1], half, epsilon = eps);
    assert_abs_diff_eq!(poly[1][2], T::SF_ZERO, epsilon = eps);

    assert_abs_diff_eq!(poly[2][0], T::SF_ONE, epsilon = eps);
    assert_abs_diff_eq!(poly[2][1], half, epsilon = eps);
    assert_abs_diff_eq!(poly[2][2], quarter, epsilon = eps);
}

fn saturated_zero_error_fit<T: TestableSimd, const KP1: usize>(
    polynomial: [T; KP1],
    samples: &[T],
) {
    let mut r = OnlinePolyfit::<T, KP1>::new();

    let eval = |x: T| {
        let mut y = T::SF_ZERO;
        for c in polynomial.into_iter().rev() {
            y = y.mul_add(x, c);
        }
        y
    };

    for x in samples {
        r.update(0, T::SF_ONE, *x, [eval(*x)]);
    }

    let [fit] = r.compute_fit();
    let eps = test_eps::<T>();

    let p = fit.fit.deg(KP1 - 1);
    for &x in samples {
        let expected = eval(x);
        let found = p.evaluate_array([x])[0];
        assert!(
            eps_diff_eq(found, expected, eps),
            "P({:?}) diverged: {:?} != {:?}",
            x,
            found,
            expected
        );
    }

    // Instead of testing that the total error is exactly zero, we instead test
    // that the error associated with the fit is zero relative to the total addressable error.
    let [addressable_error] = r.addressable_error();
    assert_eps_diff_eq(
        addressable_error - fit.errors.error(KP1 - 1),
        addressable_error,
        eps,
    );

    // assert_abs_diff_eq!(fit.errors.error(KP1 - 1), T::SF_ZERO, epsilon = eps);
}

fn test_optimal_fit_test_polynomial<T: TestableSimd>() {
    let samples = 100;
    online_minimal_fit::<T, 1>(
        |_| T::from_usize(1),
        |x| x * x + T::from_usize(5) * x - T::from_usize(1),
        samples,
    )
}
test_all_types!(test_optimal_fit_test_polynomial);

fn test_optimal_fit_test_reciprocal<T: TestableSimd>() {
    let samples = 100;
    online_minimal_fit::<T, 3>(
        |_| T::from_usize(1),
        |x| T::from_usize(1) / (x + T::from_usize(1)),
        samples,
    )
}
test_all_types!(test_optimal_fit_test_reciprocal);

fn test_optimal_fit_test_exp<T: TestableSimd>() {
    let samples = 100;
    online_minimal_fit::<T, 3>(|x| x, |x| T::exp(x), samples)
}
test_all_types!(test_optimal_fit_test_exp);

fn test_optimal_fit_test_ln<T: TestableSimd>() {
    let samples = 100;
    online_minimal_fit::<T, 5>(|x| x, |x| T::ln(x + T::SF_ONE), samples)
}
test_all_types!(test_optimal_fit_test_ln);

fn test_optimal_fit_test_discontinuity<T: TestableSimd>() {
    let samples = 100;

    online_minimal_fit::<T, 4>(
        |_| T::from_usize(1),
        |x| {
            let half = T::SF_ONE / T::from_usize(2);
            if x < half { T::SF_ZERO } else { T::SF_ONE }
        },
        samples,
    )
}
test_all_types!(test_optimal_fit_test_discontinuity);

fn test_online_multi_dem_increasing_deg_fit<T: TestableSimd>() {
    online_multi_dem_increasing_deg_fit::<T>()
}
test_all_types!(test_online_multi_dem_increasing_deg_fit);

fn test_saturated_zero_error_fit_d0_1<T: TestableSimd>() {
    saturated_zero_error_fit::<T, _>([-T::from_usize(1)], &[T::from_usize(3)])
}
test_all_types!(test_saturated_zero_error_fit_d0_1);

fn test_saturated_zero_error_fit_d0_2<T: TestableSimd>() {
    saturated_zero_error_fit::<T, _>(
        [T::from_usize(1)],
        &[
            -T::from_usize(3),
            T::from_usize(5),
            -T::from_usize(2),
            T::from_usize(1),
        ],
    )
}
test_all_types!(test_saturated_zero_error_fit_d0_2);

fn test_saturated_zero_error_fit_d1_1<T: TestableSimd>() {
    saturated_zero_error_fit::<T, _>(
        [-T::from_usize(1), T::from_usize(2)],
        &[T::from_usize(3), -T::from_usize(2), T::from_usize(0)],
    )
}
test_all_types!(test_saturated_zero_error_fit_d1_1);

fn test_saturated_zero_error_fit_d1_2<T: TestableSimd>() {
    let offset = -T::from_usize(50);
    let scale = T::from_usize(1);
    let sample_positions: Vec<T> = (0..100)
        .into_iter()
        .rev()
        .map(|i| (T::from_usize(i) - offset) * scale)
        .collect();

    saturated_zero_error_fit::<T, _>([T::from_usize(1), -T::from_usize(1)], &sample_positions)
}
test_all_types!(test_saturated_zero_error_fit_d1_2);

fn test_saturated_zero_error_fit_d2_1<T: TestableSimd>() {
    let offset = -T::from_usize(50);
    let scale = T::from_usize(10).recip();
    let sample_positions: Vec<T> = (0..100)
        .into_iter()
        .rev()
        .map(|i| (T::from_usize(i) - offset) * scale)
        .collect();

    saturated_zero_error_fit::<T, _>(
        [T::from_usize(1), T::from_usize(20), T::from_usize(300)],
        &sample_positions,
    )
}
test_all_types!(test_saturated_zero_error_fit_d2_1);
