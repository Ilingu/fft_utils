pub mod complex;
mod fft;
mod tests;

use complex::AlgebraicComplex;
use fft::fft_core;

fn to_complex(v: Vec<f64>) -> Vec<AlgebraicComplex> {
    v.into_iter().map(|x| x.into()).collect()
}

pub fn ifft(values: Vec<AlgebraicComplex>) -> Vec<AlgebraicComplex> {
    let ifftres = fft_core(values, true);
    let normalizing = 1.0 / ifftres.len() as f64;
    ifftres
        .into_iter()
        .map(|v| v.times(normalizing))
        .collect::<Vec<_>>()
}

pub fn fft(values: Vec<AlgebraicComplex>) -> Vec<AlgebraicComplex> {
    fft_core(values, false)
}

pub fn fft_convolution(mut a: Vec<f64>, mut b: Vec<f64>) -> Vec<f64> {
    let final_len = a.len() + b.len() - 1;
    a.resize(final_len, 0.0);
    b.resize(final_len, 0.0);

    let (ahat, bhat) = (fft(to_complex(a)), fft(to_complex(b)));
    let pointwise = ahat.into_iter().zip(bhat).map(|(a, b)| a * b).collect();

    ifft(pointwise).into_iter().map(|x| x.r).collect()
}
