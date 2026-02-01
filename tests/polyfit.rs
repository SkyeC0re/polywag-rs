mod common;

use approx::assert_abs_diff_eq;
use pastey::paste;
use polywag::{PolyfitCfg, Polynomial};

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
        }
    };
}

test_type!(f32);
test_type!(f64);
