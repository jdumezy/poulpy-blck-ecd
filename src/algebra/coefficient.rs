use std::ops::{Add, Mul, Neg, Sub};

use num_complex::Complex;

use crate::scalar::BlockScalar;

#[derive(Clone, Copy, Debug, PartialEq)]
/// Exact dyadic or approximate complex coefficient used by compiled maps.
pub enum Coefficient<F> {
    /// Gaussian integer numerator divided by a power of two.
    Exact {
        /// Real numerator.
        re: i128,
        /// Imaginary numerator.
        im: i128,
        /// Base-two logarithm of the denominator.
        log_den: u32,
    },
    /// Approximate complex coefficient.
    Approx(Complex<F>),
}

impl<F: BlockScalar> Coefficient<F> {
    /// Constructs and normalizes an exact Gaussian dyadic coefficient.
    pub fn exact(re: i128, im: i128, log_den: u32) -> Self {
        let mut re = re;
        let mut im = im;
        let mut log_den = log_den;
        if re == 0 && im == 0 {
            return Self::Exact {
                re: 0,
                im: 0,
                log_den: 0,
            };
        }
        while log_den != 0 && re % 2 == 0 && im % 2 == 0 {
            re /= 2;
            im /= 2;
            log_den -= 1;
        }
        Self::Exact { re, im, log_den }
    }

    /// Constructs an exact real integer coefficient.
    pub fn integer(value: i128) -> Self {
        Self::exact(value, 0, 0)
    }

    /// Constructs an exact Gaussian integer coefficient.
    pub fn gaussian_integer(re: i128, im: i128) -> Self {
        Self::exact(re, im, 0)
    }

    /// Constructs an approximate complex coefficient.
    pub fn approximate(re: F, im: F) -> Self {
        Self::Approx(Complex::new(re, im))
    }

    /// Returns the exact zero coefficient.
    pub fn zero() -> Self {
        Self::integer(0)
    }

    /// Returns the exact unit coefficient.
    pub fn one() -> Self {
        Self::integer(1)
    }

    /// Reports whether this coefficient retains an exact representation.
    pub fn is_exact(&self) -> bool {
        matches!(self, Self::Exact { .. })
    }

    /// Returns the exact numerator and denominator parts when available.
    pub fn exact_parts(&self) -> Option<(i128, i128, u32)> {
        match *self {
            Self::Exact { re, im, log_den } => Some((re, im, log_den)),
            Self::Approx(_) => None,
        }
    }

    /// Converts this coefficient to its complex floating-point value.
    pub fn value(self) -> Complex<F> {
        match self {
            Self::Approx(value) => value,
            Self::Exact { re, im, log_den } => {
                let scale = F::from_f64(0.5)
                    .expect("BlockScalar must represent 0.5")
                    .powi(i32::try_from(log_den).expect("dyadic denominator is too large"));
                Complex::new(
                    F::from_i128(re).expect("BlockScalar must represent i128") * scale,
                    F::from_i128(im).expect("BlockScalar must represent i128") * scale,
                )
            }
        }
    }

    /// Returns the complex conjugate of this coefficient.
    pub fn conjugate(self) -> Self {
        match self {
            Self::Exact { re, im, log_den } => Self::exact(re, -im, log_den),
            Self::Approx(value) => Self::Approx(value.conj()),
        }
    }

    /// Multiplies this coefficient by an integer while preserving exactness when possible.
    pub fn scale_integer(self, scalar: i128) -> Self {
        match self {
            Self::Exact { re, im, log_den } => {
                if let (Some(re), Some(im)) = (re.checked_mul(scalar), im.checked_mul(scalar)) {
                    Self::exact(re, im, log_den)
                } else {
                    Self::Approx(
                        self.value()
                            * F::from_i128(scalar).expect("BlockScalar must represent i128"),
                    )
                }
            }
            Self::Approx(value) => {
                Self::Approx(value * F::from_i128(scalar).expect("BlockScalar must represent i128"))
            }
        }
    }

    /// Divides this coefficient by a positive integer, preserving exactness when possible.
    pub fn div_usize(self, divisor: usize) -> Self {
        assert_ne!(divisor, 0, "coefficient division by zero");
        match self {
            Self::Exact { re, im, log_den } if divisor.is_power_of_two() => {
                Self::exact(re, im, log_den + divisor.trailing_zeros())
            }
            Self::Exact { re, im, log_den }
                if re % divisor as i128 == 0 && im % divisor as i128 == 0 =>
            {
                Self::exact(re / divisor as i128, im / divisor as i128, log_den)
            }
            _ => Self::Approx(
                self.value() / F::from_usize(divisor).expect("BlockScalar must represent usize"),
            ),
        }
    }

    fn add_exact(lhs: (i128, i128, u32), rhs: (i128, i128, u32)) -> Option<Self> {
        let (lhs_re, lhs_im, lhs_den) = lhs;
        let (rhs_re, rhs_im, rhs_den) = rhs;
        let den = lhs_den.max(rhs_den);
        let lhs_shift = den - lhs_den;
        let rhs_shift = den - rhs_den;
        let lhs_re = lhs_re.checked_shl(lhs_shift)?;
        let lhs_im = lhs_im.checked_shl(lhs_shift)?;
        let rhs_re = rhs_re.checked_shl(rhs_shift)?;
        let rhs_im = rhs_im.checked_shl(rhs_shift)?;
        Some(Self::exact(
            lhs_re.checked_add(rhs_re)?,
            lhs_im.checked_add(rhs_im)?,
            den,
        ))
    }

    /// Compares two coefficients within an absolute complex-norm tolerance.
    pub fn approx_eq(self, rhs: Self, tolerance: F) -> bool {
        (self.value() - rhs.value()).norm() <= tolerance
    }
}

impl<F: BlockScalar> Add for Coefficient<F> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        match (self.exact_parts(), rhs.exact_parts()) {
            (Some(lhs), Some(rhs_exact)) => Self::add_exact(lhs, rhs_exact)
                .unwrap_or_else(|| Self::Approx(self.value() + rhs.value())),
            _ => Self::Approx(self.value() + rhs.value()),
        }
    }
}

impl<F: BlockScalar> Neg for Coefficient<F> {
    type Output = Self;

    fn neg(self) -> Self::Output {
        match self {
            Self::Exact { re, im, log_den } => Self::exact(-re, -im, log_den),
            Self::Approx(value) => Self::Approx(-value),
        }
    }
}

impl<F: BlockScalar> Sub for Coefficient<F> {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        self + -rhs
    }
}

impl<F: BlockScalar> Mul for Coefficient<F> {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        match (self.exact_parts(), rhs.exact_parts()) {
            (Some((ar, ai, ad)), Some((br, bi, bd))) => {
                let re = ar
                    .checked_mul(br)
                    .and_then(|x| ai.checked_mul(bi).and_then(|y| x.checked_sub(y)));
                let im = ar
                    .checked_mul(bi)
                    .and_then(|x| ai.checked_mul(br).and_then(|y| x.checked_add(y)));
                match (re, im, ad.checked_add(bd)) {
                    (Some(re), Some(im), Some(log_den)) => Self::exact(re, im, log_den),
                    _ => Self::Approx(self.value() * rhs.value()),
                }
            }
            _ => Self::Approx(self.value() * rhs.value()),
        }
    }
}
