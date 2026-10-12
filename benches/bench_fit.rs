use criterion::{BatchSize, Criterion, criterion_group, criterion_main};
use polywag::{RegressionState, simd::SimdAble};
use rand::RngExt;
use std::hint::black_box;

fn random_state_f32<const K: usize>(samples: usize, step: f32) -> RegressionState<f32, K> {
    let mut rng = rand::rng();

    let mut x = 0.0;
    let mut r = RegressionState::<f32, K>::new();
    let w = 1.0 / samples as f32;
    for _ in 0..samples {
        r.update(0, w, x, [rng.random()]);
        x += step;
    }

    r
}

fn criterion_benchmark(c: &mut Criterion) {
    c.bench_function("f32 k=2 compute_fit", |b| {
        b.iter_batched_ref(
            || random_state_f32::<2>(100, 0.01),
            |r| {
                black_box(r.compute_fit());
            },
            BatchSize::SmallInput,
        );
    });

    c.bench_function("f32 k=3 compute_fit", |b| {
        b.iter_batched_ref(
            || random_state_f32::<3>(100, 0.01),
            |r| {
                black_box(r.compute_fit());
            },
            BatchSize::SmallInput,
        );
    });

    c.bench_function("f32 k=2 shift", |b| {
        b.iter_batched_ref(
            || random_state_f32::<2>(100, 0.01),
            |r| {
                black_box(r.shift(-0.1));
            },
            BatchSize::SmallInput,
        );
    });

    c.bench_function("f32 k=3 shift", |b| {
        b.iter_batched_ref(
            || random_state_f32::<3>(100, 0.01),
            |r| {
                black_box(r.shift(-0.1));
            },
            BatchSize::SmallInput,
        );
    });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
