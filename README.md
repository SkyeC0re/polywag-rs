![Maintenance](https://img.shields.io/badge/maintenance-actively--developed-brightgreen.svg)

# polywag

A `no-std` library for online, positively weighted least squares polynomial regression. At a high
level this library allows a user to fit the best (in a weighted least squares sense) polynomial
associated with a set of `w`(weight), `x`, `y` samples. Specifically, given a set of weighted samples
$(w_{l, i}, x_{l, i}, y_{l, i})$, where:
- $y_{l, i}$ a sampled value of a the $l$-th derivative of a function at $x_{l, i}$.
- $w_{l, i} > 0$ is the sample's associated weight.

Then the obtained regression polynomial $P$ of degree $k$ minimizes the following sum:

$$
\sum_{l=0}^k \sum_{i=1}^{N_l} w_{l, i} \left[ y_{l, i} - P^{(l)}(x_{l, i})  \right]^2
$$

## Basic Usage

```rust
extern crate polywag;
use polywag::{RegressionState, kp1_arr};

pub fn main() {
    // Create a new regression state with a maximum regression polynomial degree 2
    // and without any samples. The regression always has a single x-dimension and
    // by default a single y-dimension.
    let mut r = RegressionState::<f32, 2>::new();

    // Add some zeroeth-degree (`w, x, [y]`) samples.
    r.update(0, 1.0, 0.0, [2.0]);
    r.update(0, 1.0, 1.0, [3.0]);
    r.update(0, 1.0, 2.0, [4.0]);

    // Compute the regression based on the current sample weights.
    let [fit] = r.compute_fit();

    // This should produce the following approximate polynomial regressions up to degree 2:

    // [ 3.0 ]
    fit.polynomials.array_p::<0>();
    // &[3.0]
    fit.polynomials.slice_p(0);

    // [ 2.0, 1.0 ]
    fit.polynomials.array_p::<1>();
    // &[ 2.0, 1.0]
    fit.polynomials.slice_p(1);

    // [ 2.0, 1.0, 0.0 ]
    fit.polynomials.array_p::<2>();
    // &[ 2.0, 1.0, 0.0 ]
    fit.polynomials.slice_p(2);

    // [ 2.0, 1.0, 0.0, 0.0 ]
    fit.polynomials.array_p::<3>();
    // &[ 2.0, 1.0, 0.0 ]
    fit.polynomials.slice_p(3);
}
```

## Numerical Stability

Care was taken to keep the algorithm as stable as possible whilst still allowing the algorithm to be online
(online in the sense that samples are not stored individually and its performance characteristics are unaffected
by the amount of samples currently added to the regression state). To accomplish this, the main technique used here
is [Modified Gram Schmidt](https://en.wikipedia.org/wiki/Gram%E2%80%93Schmidt_process).

## Bias

We allow regressions to be generated using special bias samples, one for each derivative:

```rust
extern crate polywag;
use polywag::{RegressionState, kp1_arr};

pub fn main() {
    let mut r = RegressionState::<f32, 2>::new();

    // Add some bad samples (zero weight or not enough unique `x` values).
    r.update(0, 0.0, 1.0, [2.0]);
    r.update(0, 0.0, 1.0, [3.0]);

    // Compute the regression based on the bad weight plus a bias at `x=0`
    // in each dimension with weight 1.
    let [fit] = r.compute_fit_with_bias(kp1_arr![
        (1.0, [1.0]),
        (1.0, [2.0]),
        (1.0, [3.0]),
    ]);

    // This should produce the following approximate polynomial regressions up to degree 2:

    // &[ 1.0 ]
    fit.polynomials.slice_p(0);

    // &[ 1.0, 2.0 ]
    fit.polynomials.slice_p(1);

    // &[ 1.0, 2.0, 3.0 ]
    fit.polynomials.slice_p(2);
}
```


The algorithm is written in such a way that if the bias samples by themselves do not yield
a numerical regression with infinities or `NaN`s then neither will the full regression (the
bias samples together with the regular samples) as a result of catastrophic cancellation. In practice,
this simply means that if the bias samples produce reasonable results by themselves, then the
full regression should produce reasonable results as well.

## Rolling Regression

As part of the online nature of the regression state, it is possible create an infinite response filter
with exponential decay as follows:

```rust
extern crate polywag;
use polywag::{RegressionState, kp1_arr};

pub fn main() {
    let mut r = RegressionState::<f32, 2>::new();
    // Bias the initial regressions ever so slightly towards the zero polynomial,
    // until there are sufficient samples in the regression state to fit a second
    // degree polynomial.
    r.update(0, 1.0, 0.0, [0.0]);
    r.update(1, 1.0, 0.0, [0.0]);
    r.update(2, 1.0, 0.0, [0.0]);

    for t in 1..100 {
        let t = t as f32;
        // Sample weights decay to 0.9 of their value every time unit.
        r.scale(0.9);

        r.update(0, 1.0, t, [f32::sin(t)]);
        // The current filter response.
        let [fit] = r.compute_fit();
    }
}
```

The code above effectively produces the second degree polynomial $P$ which minimizes
the least squares sum of the exponentially decaying sine samples and original bias:

$$
\sum_{l=0}^2 0.9^{T} \epsilon \left[ P^{(l)}(t) \right]^2 + \sum_{t=1} 0.9^{T-t} \left[ \sin(t) - P(t)  \right]^2
$$

where $T$ is the current time.



## Version 

```
0.1.0
```

## License

`MIT`
