use criterion::{Criterion, criterion_group, criterion_main};
use std::hint::black_box;

use polywag::{PolyfitCfg, RawPolynomial};

fn criterion_benchmark(c: &mut Criterion) {
    c.bench_function("fit 1000 f32", |b| {
        let mut poly = RawPolynomial::new();

        let xs = (0..1000).map(|i| (i as f32) * 0.01).collect::<Vec<_>>();
        let ys = Vec::from_iter(xs.iter().copied().map(|x| x * x + x + 1.0));

        let cfg = PolyfitCfg::<f32>::new_with_max_deg(2);

        let mut ws = polywag::Bump::new();
        poly.polyfit_from_iter(
            &ws,
            cfg,
            xs.iter().zip(ys.iter()).map(|(x, y)| (1.0, *x, [*y])),
        );
        ws.reset();

        b.iter(|| {
            black_box(poly.polyfit_from_iter(
                &ws,
                cfg,
                xs.iter().zip(ys.iter()).map(|(x, y)| (1.0, *x, [*y])),
            ));

            ws.reset();
        });
    });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
