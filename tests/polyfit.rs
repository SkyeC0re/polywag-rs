mod common;

use approx::assert_abs_diff_eq;
use core::ops::Add;
use pastey::paste;
use polywag::{
    PolyfitCfg, Polynomial,
    simd::{SimdAble, SimdField},
};

use crate::common::{F256, TestableSimd};

fn test_eps<T: TestableSimd>() -> T {
    let ratio = T::from_usize(85) / T::from_usize(100);

    T::exp(ratio * T::SF_EPS.ln())
}

#[track_caller]
fn empty_fit<T: TestableSimd>() {
    let mut poly = Polynomial::<3, T>::new();
    poly.polyfit_from_iter(PolyfitCfg::new_with_max_deg(100), [].into_iter());

    assert_eq!(poly.len(), 0)
}

/// Least squares regressions should reproduce polynomials up to the maximum fitting degree exactly.
#[track_caller]
fn multi_dem_increasing_deg_fit<T: TestableSimd>() {
    let mut poly = Polynomial::<3, T>::new();

    let half = T::SF_ONE / T::from_usize(2);
    let quarter = T::SF_ONE / T::from_usize(4);

    poly.polyfit_from_iter(
        PolyfitCfg::new_with_max_deg(2),
        (0..10).into_iter().map(|x| {
            let x = T::from_usize(x);
            (
                T::SF_ONE,
                x,
                [
                    T::SF_ONE,
                    T::SF_ONE + half * x,
                    T::SF_ONE + half * x + quarter * x * x,
                ],
            )
        }),
    );

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

/// Equal weights with duplicate `x` points with a symmetric offset should produce a least squares fit
/// exactly halfway in between.
#[track_caller]
fn split_linear_fit<T: TestableSimd>() {
    let mut poly = Polynomial::<1, T>::new();

    let iter = (0..10).into_iter().map(|x| T::from_usize(x));
    poly.polyfit_from_iter(
        PolyfitCfg::new_with_max_deg(1),
        iter.clone()
            .map(|x| (T::SF_ONE, x, [x + T::SF_ONE]))
            .chain(iter.map(|x| (T::SF_ONE, x, [x - T::SF_ONE]))),
    );

    let eps = test_eps();
    assert_abs_diff_eq!(poly[0][0], T::SF_ZERO, epsilon = eps);
    assert_abs_diff_eq!(poly[0][1], T::SF_ONE, epsilon = eps);
}

/// Degeneracies, such as duplicate `x` samples or zero weighted, should decrease the maximum fitted degree.
#[track_caller]
fn degeneracy_short_circuit<T: TestableSimd>() {
    let mut poly = Polynomial::<1, T>::new();
    poly.polyfit_from_iter(
        PolyfitCfg::new_with_max_deg(10),
        (0..20).into_iter().map(|x| {
            let (w, x) = match (x % 10 == 0, x % 2 == 0) {
                (true, _) => (T::SF_ONE, T::from_usize(x)),
                // Zero weight degeneracy
                (false, true) => (T::SF_ZERO, T::from_usize(x)),
                // Duplicate `x` degeneracy
                (false, false) => (T::SF_ONE, T::from_usize(x - (x % 10))),
            };

            (w, x, [x + T::SF_ONE])
        }),
    );

    assert_eq!(poly[0].len(), 2);

    let eps = test_eps();
    assert_abs_diff_eq!(poly[0][0], T::SF_ONE, epsilon = eps);
    assert_abs_diff_eq!(poly[0][1], T::SF_ONE, epsilon = eps);
}

/// Tests a function to ensure a minimal least squares fit is produced (up to our selected epsilon).
#[track_caller]
fn test_minimal_fit<T: TestableSimd>(w: fn(T) -> T, y: fn(T) -> T, samples: usize, max_deg: u8) {
    let mut f256_poly = Polynomial::<1, F256>::new();
    let scaling = T::SF_ONE / T::from_usize(samples);
    f256_poly.polyfit_from_iter(
        PolyfitCfg::new_with_max_deg(max_deg),
        (0..=samples).into_iter().map(|x| {
            let x = scaling * T::from_usize(x);
            (w(x).into(), x.into(), [y(x).into()])
        }),
    );

    let avg_err = scaling.into()
        * (0..=samples)
            .into_iter()
            .map(|x| {
                let x = scaling * T::from_usize(x);
                let y_p = f256_poly.evaluate_array([x.into()])[0][0];
                let err = y(x).into() - y_p;
                w(x).into() * err * err
            })
            .reduce(F256::add)
            .unwrap_or(F256::SF_ZERO);

    let f256_eps: F256 = T::SF_EPS.into();
    for i in 0..f256_poly[0].len() {
        for jitter in [-f256_eps, f256_eps] {
            let p_bckp = f256_poly[0][i];
            f256_poly[0][i] += jitter;
            let jitter_avg_err = scaling.into()
                * (0..=samples)
                    .into_iter()
                    .map(|x| {
                        let x = scaling * T::from_usize(x);
                        let y_p = f256_poly.evaluate_array([x.into()])[0][0];
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
            f256_poly[0][i] = p_bckp;
        }
    }

    let mut poly = Polynomial::<1, T>::new();
    poly.polyfit_from_iter(
        PolyfitCfg::new_with_max_deg(max_deg),
        (0..=samples).into_iter().map(|x| {
            let x = scaling * T::from_usize(x);
            (w(x), x, [y(x)])
        }),
    );

    let poly_err = scaling.into()
        * (0..=samples)
            .into_iter()
            .map(|x| {
                let x = scaling * T::from_usize(x);
                let y_p = poly.evaluate_array([x])[0][0].into();
                let err = y(x).into() - y_p;
                w(x).into() * err * err
            })
            .reduce(F256::add)
            .unwrap_or(F256::SF_ZERO);

    assert!(
        (avg_err - poly_err).abs() < test_eps::<T>().into(),
        "Result diverges from optimal fit solution"
    );
}

macro_rules! test_type {
    ($t:ty) => {
        paste! {

            #[test]
            fn [<$t _empty_fit>]() {
                empty_fit::<$t>();
            }

            #[test]
            fn [<$t _multi_dem_increasing_deg_fit>]() {
                multi_dem_increasing_deg_fit::<$t>()
            }

            #[test]
            fn [<$t _split_linear_fit>]() {
                split_linear_fit::<$t>()
            }

            #[test]
            fn [<$t _degeneracy_short_circuit>]() {
                degeneracy_short_circuit::<$t>()
            }

            #[test]
            fn [<$t _optimal_fit_test_1>]() {
                let samples = 20;
                test_minimal_fit::<$t>(
                    |_| $t::from_usize(1),
                    |x| x*x + $t::from_usize(5) * x - $t::from_usize(1),
                    samples,
                    1,
                )
            }

            #[test]
            fn [<$t _optimal_fit_test_2>]() {
                let samples = 20;
                test_minimal_fit::<$t>(
                    |_| $t::from_usize(1),
                    |x| $t::from_usize(1) / (x + $t::from_usize(1)),
                    samples,
                    3,
                )
            }

            #[test]
            fn [<$t _optimal_fit_test_3>]() {
                let samples = 20;
                test_minimal_fit::<$t>(
                    |x| x,
                    |x| $t::exp(x),
                    samples,
                    3,
                )
            }
        }
    };
}

test_type!(f32);
test_type!(f64);
