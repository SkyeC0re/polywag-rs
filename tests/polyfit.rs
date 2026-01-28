use polywag::{PolyfitCfg, Polynomial};

#[test]
fn empty_fit() {
    let mut poly = Polynomial::<3, f64>::new();
    poly.polyfit_from_iter(PolyfitCfg::<f64>::new(), [].into_iter());

    assert_eq!(poly.len(), 0)
}
