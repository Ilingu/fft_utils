use std::f64::consts::PI;

use crate::complex::{AlgebraicComplex, ExponentialComplex};

#[macro_export]
macro_rules! compvec {
    ( $( $x:expr ),* ) => {
        {
            use fft_utils::complex::ExponentialComplex;
            let mut temp_vec = Vec::new();
            $(
                temp_vec.push(AlgebraicComplex::from($x));
            )*
            temp_vec
        }
    };
}

pub fn fft_core(mut values: Vec<AlgebraicComplex>, inverse: bool) -> Vec<AlgebraicComplex> {
    pad_with_zeroes(&mut values);
    let n = values.len(); // n is a power of 2
    if n == 1 {
        return values;
    }

    let nth_root = {
        let sign = if inverse { -1.0 } else { 1.0 };
        ExponentialComplex::new(sign * 2.0 * PI / n as f64, 1.0)
    };

    let (mut pe, mut po) = (Vec::with_capacity(n / 2), Vec::with_capacity(n / 2)); // n power of 2
    for (i, p) in values.into_iter().enumerate() {
        if i % 2 == 0 {
            pe.push(p);
        } else {
            po.push(p);
        }
    }

    let (ye, yo) = (fft_core(pe, inverse), fft_core(po, inverse));
    let mut y: Vec<AlgebraicComplex> = vec![0.0.into(); n];
    for j in 0..(n / 2) {
        let jth_nth_root = AlgebraicComplex::from(nth_root.pow(j as f64));
        y[j] = ye[j] + jth_nth_root * yo[j];
        y[j + n / 2] = ye[j] - jth_nth_root * yo[j];
    }

    y
}

/// pad vec with zero until its length is the next power power of two after its current len
fn pad_with_zeroes(values: &mut Vec<AlgebraicComplex>) {
    let n = values.len();
    let newn = n.checked_next_power_of_two().expect("Overflow");
    values.resize(newn, 0.0.into());
}
