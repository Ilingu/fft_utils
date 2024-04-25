use std::ops::{Add, Div, Mul, Sub};
#[derive(Debug, Clone, Copy)]
pub struct ExponentialComplex {
    pub phase: f64,
    pub modulus: f64,
}

impl ExponentialComplex {
    pub fn new(phase: f64, modulus: f64) -> Self {
        Self { phase, modulus }
    }

    pub fn pow(&self, scalar: f64) -> Self {
        Self {
            phase: scalar * self.phase,
            modulus: if scalar == 1.0 {
                1.0
            } else {
                self.modulus.powf(scalar)
            },
        }
    }

    pub fn times(&self, scalar: f64) -> Self {
        Self {
            modulus: scalar * self.modulus,
            phase: self.phase,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct AlgebraicComplex {
    pub r: f64,
    pub im: f64,
}

impl AlgebraicComplex {
    pub fn new(real: f64, imaginary: f64) -> Self {
        Self {
            r: real,
            im: imaginary,
        }
    }

    pub fn conjugate(&self) -> Self {
        Self {
            r: self.r,
            im: -self.im,
        }
    }

    pub fn times(&self, scalar: f64) -> Self {
        Self {
            r: scalar * self.r,
            im: scalar * self.im,
        }
    }

    pub fn modulus(&self) -> f64 {
        ((self.r * self.r) + (self.im * self.im)).sqrt()
    }

    pub fn modulus_square(&self) -> f64 {
        (self.r * self.r) + (self.im * self.im)
    }

    pub fn phase(&self) -> f64 {
        (self.im).atan2(self.r)
    }
}

/* Type conversion */

impl From<f64> for ExponentialComplex {
    fn from(val: f64) -> Self {
        AlgebraicComplex::from(val).into()
    }
}

impl From<f64> for AlgebraicComplex {
    fn from(val: f64) -> Self {
        Self { r: val, im: 0.0 }
    }
}

impl From<AlgebraicComplex> for ExponentialComplex {
    fn from(val: AlgebraicComplex) -> Self {
        Self {
            phase: val.phase(),
            modulus: val.modulus(),
        }
    }
}

impl From<ExponentialComplex> for AlgebraicComplex {
    fn from(val: ExponentialComplex) -> Self {
        Self {
            r: val.modulus * val.phase.cos(),
            im: val.modulus * val.phase.sin(),
        }
    }
}

/* Operation implementation */

impl Add for AlgebraicComplex {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self {
            r: self.r + rhs.r,
            im: self.im + rhs.im,
        }
    }
}

impl Sub for AlgebraicComplex {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Self {
            r: self.r - rhs.r,
            im: self.im - rhs.im,
        }
    }
}

impl Mul for AlgebraicComplex {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        Self {
            r: self.r * rhs.r - self.im * rhs.im,
            im: self.r * rhs.im + rhs.r * self.im,
        }
    }
}

impl Div for AlgebraicComplex {
    type Output = Self;

    fn div(self, rhs: Self) -> Self::Output {
        (self * rhs.conjugate()).times(1.0 / rhs.modulus_square())
    }
}

impl Add for ExponentialComplex {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        (AlgebraicComplex::from(self) + AlgebraicComplex::from(rhs)).into()
    }
}

impl Sub for ExponentialComplex {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        (AlgebraicComplex::from(self) - AlgebraicComplex::from(rhs)).into()
    }
}

impl Mul for ExponentialComplex {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        (AlgebraicComplex::from(self) * AlgebraicComplex::from(rhs)).into()
    }
}

impl Div for ExponentialComplex {
    type Output = Self;

    fn div(self, rhs: Self) -> Self::Output {
        (AlgebraicComplex::from(self) / AlgebraicComplex::from(rhs)).into()
    }
}
