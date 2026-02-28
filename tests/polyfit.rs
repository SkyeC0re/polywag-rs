mod common;

use approx::assert_abs_diff_eq;
use pastey::paste;
use polywag::{PolyfitCfg, Polynomial, simd::SimdAble};

use crate::common::TestableSimd;

fn eps<T: TestableSimd>() -> T {
    let ratio = T::from_usize(85) / T::from_usize(100);

    T::exp(ratio * T::EPS.ln())
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

    let half = T::ONE / T::from_usize(2);
    let quarter = T::ONE / T::from_usize(4);

    poly.polyfit_from_iter(
        PolyfitCfg::new_with_max_deg(2),
        (0..10).into_iter().map(|x| {
            let x = T::from_usize(x);
            (
                T::ONE,
                x,
                [
                    T::ONE,
                    T::ONE + half * x,
                    T::ONE + half * x + quarter * x * x,
                ],
            )
        }),
    );

    let eps = eps();
    assert_abs_diff_eq!(poly[0][0], T::ONE, epsilon = eps);
    assert_abs_diff_eq!(poly[0][1], T::ZERO, epsilon = eps);
    assert_abs_diff_eq!(poly[0][2], T::ZERO, epsilon = eps);

    assert_abs_diff_eq!(poly[1][0], T::ONE, epsilon = eps);
    assert_abs_diff_eq!(poly[1][1], half, epsilon = eps);
    assert_abs_diff_eq!(poly[1][2], T::ZERO, epsilon = eps);

    assert_abs_diff_eq!(poly[2][0], T::ONE, epsilon = eps);
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
            .map(|x| (T::ONE, x, [x + T::ONE]))
            .chain(iter.map(|x| (T::ONE, x, [x - T::ONE]))),
    );

    let eps = eps();
    assert_abs_diff_eq!(poly[0][0], T::ZERO, epsilon = eps);
    assert_abs_diff_eq!(poly[0][1], T::ONE, epsilon = eps);
}

/// Degeneracies, such as duplicate `x` samples or zero weighted, should decrease the maximum fitted degree.
#[track_caller]
fn degeneracy_short_circuit<T: TestableSimd>() {
    let mut poly = Polynomial::<1, T>::new();
    poly.polyfit_from_iter(
        PolyfitCfg::new_with_max_deg(10),
        (0..20).into_iter().map(|x| {
            let (w, x) = match (x % 10 == 0, x % 2 == 0) {
                (true, _) => (T::ONE, T::from_usize(x)),
                // Zero weight degeneracy
                (false, true) => (T::ZERO, T::from_usize(x)),
                // Duplicate `x` degeneracy
                (false, false) => (T::ONE, T::from_usize(x - (x % 10))),
            };

            (w, x, [x + T::ONE])
        }),
    );

    assert_eq!(poly[0].len(), 2);

    let eps = eps();
    assert_abs_diff_eq!(poly[0][0], T::ONE, epsilon = eps);
    assert_abs_diff_eq!(poly[0][1], T::ONE, epsilon = eps);
}

/// Degeneracies, such as duplicate `x` samples or zero weighted, should decrease the maximum fitted degree.
#[track_caller]
fn test_minimal_fit<T: TestableSimd>(
    w: fn(T) -> T,
    y: fn(T) -> T,
    samples: usize,
    max_deg: u8,
    jitter: T,
) {
    let mut poly = Polynomial::<1, T>::new();

    let scaling = T::ONE / T::from_usize(samples);
    poly.polyfit_from_iter(
        PolyfitCfg::new_with_max_deg(max_deg),
        (0..=samples).into_iter().map(|x| {
            let x = scaling * T::from_usize(x);
            (w(x), x, [y(x)])
        }),
    );

    let avg_err = scaling
        * (0..=samples)
            .into_iter()
            .map(|x| {
                let x = scaling * T::from_usize(x);
                let y_p = poly.evaluate_array([x])[0][0];
                let err = y(x) - y_p;
                w(x) * err * err
            })
            .reduce(T::add)
            .unwrap_or(T::ZERO);

    for i in 0..poly[0].len() {
        for jitter in [-jitter, jitter] {
            let p_bckp = poly[0][i];
            poly[0][i] += jitter;
            let jitter_avg_err = scaling
                * (0..=samples)
                    .into_iter()
                    .map(|x| {
                        let x = scaling * T::from_usize(x);
                        let y_p = poly.evaluate_array([x])[0][0];
                        let err = y(x) - y_p;
                        w(x) * err * err
                    })
                    .reduce(T::add)
                    .unwrap_or(T::ZERO);

            assert!(
                avg_err <= jitter_avg_err + eps(),
                "Non optimal fit produced. Improved from {avg_err:?} to {jitter_avg_err:?} when adding {jitter:?} to fit polynomial coefficient {i}"
            );
            poly[0][i] = p_bckp;
        }
    }
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
                    $t::from_usize(100) * eps::<$t>(),
                )
            }
        }
    };
}

test_type!(f32);
test_type!(f64);
