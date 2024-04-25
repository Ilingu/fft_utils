#[cfg(test)]
mod benchmark {
    use std::time::Instant;

    use nanorand::{Rng, WyRand};

    use crate::fft_convolution;

    fn generate_random_list(size: usize) -> Vec<f64> {
        let mut rng = WyRand::new();
        (0..size)
            .map(|_| rng.generate_range(0_usize..=1000) as f64)
            .collect()
    }

    #[test]
    fn convolution_bench() {
        let (a, b) = (generate_random_list(100_000), generate_random_list(100_000));

        let now = Instant::now();
        let _ = fft_convolution(a, b);
        println!("{:.2?}", now.elapsed());
    }

    /// code stollen from https://github.com/TecnoKrash/elenia/blob/master/src/convolution.rs to see if my implementation of
    /// fftconvolution was in the same order of magnitude
    mod elenia_conv {
        use std::f64::consts::PI;
        use std::ops::{Add, Mul, Sub};
        #[derive(Debug)]

        pub struct C {
            pub re: f64,
            pub im: f64,
        }

        impl Add for C {
            type Output = Self;

            fn add(self, other: Self) -> Self {
                Self {
                    re: self.re + other.re,
                    im: self.im + other.im,
                }
            }
        }
        impl Sub for C {
            type Output = Self;

            fn sub(self, other: Self) -> Self {
                Self {
                    re: self.re - other.re,
                    im: self.im - other.im,
                }
            }
        }
        impl Mul for C {
            type Output = Self;

            fn mul(self, other: Self) -> Self {
                Self {
                    re: self.re * other.re - self.im * other.im,
                    im: self.im * other.re + other.im * self.re,
                }
            }
        }

        impl Copy for C {}

        impl Clone for C {
            fn clone(&self) -> Self {
                *self
            }
        }

        impl C {
            pub fn c(x: f64) -> C {
                C { re: x, im: 0.0 }
            }
            pub fn rc(x: C) -> f64 {
                x.re
            }
        }

        fn expo(x: f64, p: u32) -> f64 {
            if p == 0 {
                return 1.0;
            }
            if p == 1 {
                return x;
            }
            let e2 = expo(x, p / 2);
            if p % 2 == 0 {
                e2 * e2
            } else {
                e2 * e2 * x
            }
        }

        // Apply the fast fourier transform to the list in entry
        pub fn fft(p: Vec<C>, invert: bool) -> Vec<C> {
            let n: i32 = p.len() as i32;

            if n == 1 {
                return p;
            }

            let np = f64::from(n);
            let n2 = (n / 2) as usize;

            let mut w: Vec<C> = vec![];

            let mut theta = (2.0 * PI) / np;

            if invert {
                theta = -theta;
            }

            for j in 0..n2 {
                w.push(C {
                    re: (theta * (f64::from(j as i32))).cos(),
                    im: (theta * (f64::from(j as i32))).sin(),
                })
            }

            let mut pe: Vec<C> = vec![];
            let mut po: Vec<C> = vec![];

            for i in 0..(p.len()) {
                if i % 2 == 0 {
                    pe.push(p[i]);
                } else {
                    po.push(p[i]);
                }
            }

            let ye = fft(pe, invert);
            let yo = fft(po, invert);
            let mut y = vec![C { re: 0.0, im: 0.0 }; p.len()];

            for j in 0..(n2) {
                y[j] = ye[j] + w[j] * yo[j];
                y[j + n2] = ye[j] - w[j] * yo[j];
            }
            y
        }

        // return the same vector but adds zeros to the end to have p1.len() = 2^k
        fn add_zeros(p: &mut Vec<f64>) {
            let l = p.len() as u32;
            let k = u32::BITS - l.leading_zeros();

            if l != (1 << (k - 1)) {
                let k2: u32 = 1 << k;
                for _i in l..k2 {
                    p.push(0.0);
                }
            }
        }

        fn c_vec(p: &mut Vec<f64>) -> Vec<C> {
            let mut result: Vec<C> = Vec::with_capacity(p.len());
            for i in 0..p.len() {
                result.push(C::c(p[i]));
            }
            result
        }

        pub fn convolution_2d(p1: &mut Vec<f64>, p2: &mut Vec<f64>) -> Vec<f64> {
            let lp1 = p1.len();
            let lp2 = p2.len();

            for _i in 1..lp1 {
                p2.push(0.0)
            }

            for _j in 1..lp2 {
                p1.push(0.0)
            }

            add_zeros(p1);
            add_zeros(p2);

            let cp1 = c_vec(p1);
            let cp2 = c_vec(p2);

            let fp1 = fft(cp1, false);
            let fp2 = fft(cp2, false);

            let n = fp1.len() as u32;
            let np = f64::from(n);

            let mut point = Vec::with_capacity(fp1.len());

            for i in 0..fp1.len() {
                point.push(fp1[i] * fp2[i]);
            }

            let result_c = fft(point, true);
            let mut result: Vec<f64> = Vec::with_capacity(fp1.len());

            for k in 0..(lp1 + lp2 - 1) {
                result.push((1.0 / np) * result_c[k].re);
            }
            result
        }
    }

    #[test]
    fn convolution_bench_elenia() {
        use crate::tests::benchmark::elenia_conv::convolution_2d;

        let (mut a, mut b) = (generate_random_list(100_000), generate_random_list(100_000));

        let now = Instant::now();
        let _ = convolution_2d(&mut a, &mut b);
        println!("{:.2?}", now.elapsed());
    }
}
