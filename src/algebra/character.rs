use anyhow::{Result, ensure};

use super::{AffineFunction, BlockEncoding, CleaningMode, Coefficient, NativeOperation};
use crate::scalar::BlockScalar;

fn root_of_unity<F: BlockScalar>(order: usize, exponent: i128) -> Coefficient<F> {
    let exponent = exponent.rem_euclid(order as i128) as usize;
    if exponent == 0 {
        return Coefficient::one();
    }
    if order.is_multiple_of(2) && exponent == order / 2 {
        return Coefficient::integer(-1);
    }
    if order.is_multiple_of(4) {
        if exponent == order / 4 {
            return Coefficient::gaussian_integer(0, 1);
        }
        if exponent == 3 * order / 4 {
            return Coefficient::gaussian_integer(0, -1);
        }
    }
    let angle = F::from_usize(2 * exponent).expect("BlockScalar must represent usize") * F::PI()
        / F::from_usize(order).expect("BlockScalar must represent usize");
    Coefficient::approximate(angle.cos(), angle.sin())
}

fn check_values<F: BlockScalar>(values: &[Coefficient<F>], expected: usize) -> Result<()> {
    ensure!(
        values.len() == expected,
        "function has {} values, expected {expected}",
        values.len()
    );
    Ok(())
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
/// Root-of-unity character encoding of the additive group modulo an integer.
pub struct Bru {
    modulus: usize,
}

impl Bru {
    /// Constructs a root-of-unity encoding for the requested modulus.
    pub fn new(modulus: usize) -> Result<Self> {
        ensure!(modulus >= 2, "BRU modulus must be at least 2");
        Ok(Self { modulus })
    }

    /// Returns the additive modulus and alphabet size.
    pub fn modulus(&self) -> usize {
        self.modulus
    }
}

impl<F: BlockScalar> BlockEncoding<F> for Bru {
    fn alphabet_size(&self) -> usize {
        self.modulus
    }

    fn encode_into(&self, value: usize, output: &mut [Coefficient<F>]) -> Result<()> {
        ensure!(value < self.modulus, "BRU value {value} is out of range");
        ensure!(
            output.len() == self.modulus - 1,
            "BRU output has width {}, expected {}",
            output.len(),
            self.modulus - 1
        );
        for (coordinate, coefficient) in output.iter_mut().enumerate() {
            *coefficient = root_of_unity(self.modulus, ((coordinate + 1) * value) as i128);
        }
        Ok(())
    }

    fn interpolate(&self, values: &[Coefficient<F>]) -> Result<AffineFunction<F>> {
        check_values(values, self.modulus)?;
        let coefficient = |k: usize| {
            values
                .iter()
                .enumerate()
                .fold(Coefficient::zero(), |acc, (value, &output)| {
                    acc + output * root_of_unity(self.modulus, -((k * value) as i128))
                })
                .div_usize(self.modulus)
        };
        Ok(AffineFunction {
            bias: coefficient(0),
            weights: (1..self.modulus).map(coefficient).collect(),
        })
    }

    fn native_operation(&self) -> Option<NativeOperation> {
        Some(NativeOperation::AddMod)
    }

    fn native_product(&self, lhs: usize, rhs: usize) -> Option<usize> {
        (lhs < self.modulus && rhs < self.modulus).then_some((lhs + rhs) % self.modulus)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
/// Multiplicative-character encoding of a prime field, extended with zero.
pub struct Lbru {
    prime: usize,
    generator: usize,
    logarithms: Vec<usize>,
}

impl Lbru {
    /// Constructs an encoding using an automatically selected primitive root.
    pub fn new(prime: usize) -> Result<Self> {
        let generator = primitive_root(prime)?;
        Self::with_generator(prime, generator)
    }

    /// Constructs an encoding using the supplied primitive root.
    pub fn with_generator(prime: usize, generator: usize) -> Result<Self> {
        ensure!(is_prime(prime), "LBRU modulus {prime} is not prime");
        ensure!(
            generator > 0 && generator < prime,
            "invalid generator {generator}"
        );
        let mut logarithms = vec![usize::MAX; prime];
        let mut value = 1;
        for logarithm in 0..prime - 1 {
            ensure!(
                logarithms[value] == usize::MAX,
                "{generator} does not generate the multiplicative group modulo {prime}"
            );
            logarithms[value] = logarithm;
            value = mul_mod(value, generator, prime);
        }
        ensure!(value == 1, "invalid generator cycle modulo {prime}");
        ensure!(
            logarithms
                .iter()
                .skip(1)
                .all(|&logarithm| logarithm != usize::MAX),
            "{generator} is not a primitive root modulo {prime}"
        );
        Ok(Self {
            prime,
            generator,
            logarithms,
        })
    }

    /// Returns the prime modulus and alphabet size.
    pub fn modulus(&self) -> usize {
        self.prime
    }

    /// Returns the primitive root used to order nonzero field elements.
    pub fn generator(&self) -> usize {
        self.generator
    }
}

impl<F: BlockScalar> BlockEncoding<F> for Lbru {
    fn alphabet_size(&self) -> usize {
        self.prime
    }

    fn encode_into(&self, value: usize, output: &mut [Coefficient<F>]) -> Result<()> {
        ensure!(value < self.prime, "LBRU value {value} is out of range");
        ensure!(
            output.len() == self.prime - 1,
            "LBRU output has width {}, expected {}",
            output.len(),
            self.prime - 1
        );
        if value == 0 {
            output.fill(Coefficient::zero());
            return Ok(());
        }
        let logarithm = self.logarithms[value];
        output[0] = Coefficient::one();
        for (coordinate, coefficient) in output.iter_mut().enumerate().skip(1) {
            *coefficient = root_of_unity(self.prime - 1, (coordinate * logarithm) as i128);
        }
        Ok(())
    }

    fn interpolate(&self, values: &[Coefficient<F>]) -> Result<AffineFunction<F>> {
        check_values(values, self.prime)?;
        let fourier = |k: usize| {
            (0..self.prime - 1)
                .fold(Coefficient::zero(), |acc, logarithm| {
                    let value = pow_mod(self.generator, logarithm, self.prime);
                    acc + values[value] * root_of_unity(self.prime - 1, -((k * logarithm) as i128))
                })
                .div_usize(self.prime - 1)
        };
        let bias = values[0];
        let mut weights = Vec::with_capacity(self.prime - 1);
        weights.push(fourier(0) - bias);
        weights.extend((1..self.prime - 1).map(fourier));
        Ok(AffineFunction { bias, weights })
    }

    fn native_operation(&self) -> Option<NativeOperation> {
        Some(NativeOperation::MulMod)
    }

    fn native_product(&self, lhs: usize, rhs: usize) -> Option<usize> {
        (lhs < self.prime && rhs < self.prime).then_some(mul_mod(lhs, rhs, self.prime))
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
/// Walsh-Hadamard character encoding of fixed-width bit strings.
pub struct WalshHadamard {
    bits: usize,
    size: usize,
}

impl WalshHadamard {
    /// Constructs an encoding for bit strings of the requested width.
    pub fn new(bits: usize) -> Result<Self> {
        ensure!(bits > 0, "Walsh-Hadamard dimension must be positive");
        let shift = u32::try_from(bits)
            .map_err(|_| anyhow::anyhow!("Walsh-Hadamard alphabet is too large"))?;
        let size = 1usize
            .checked_shl(shift)
            .ok_or_else(|| anyhow::anyhow!("Walsh-Hadamard alphabet is too large"))?;
        Ok(Self { bits, size })
    }

    /// Returns the number of bits in each alphabet symbol.
    pub fn bits(&self) -> usize {
        self.bits
    }
}

impl<F: BlockScalar> BlockEncoding<F> for WalshHadamard {
    fn alphabet_size(&self) -> usize {
        self.size
    }

    fn encode_into(&self, value: usize, output: &mut [Coefficient<F>]) -> Result<()> {
        ensure!(
            value < self.size,
            "Walsh-Hadamard value {value} is out of range"
        );
        ensure!(
            output.len() == self.size - 1,
            "Walsh-Hadamard output has width {}, expected {}",
            output.len(),
            self.size - 1
        );
        for (coordinate, coefficient) in output.iter_mut().enumerate() {
            let mask = coordinate + 1;
            *coefficient = Coefficient::integer(if (mask & value).count_ones().is_multiple_of(2) {
                1
            } else {
                -1
            });
        }
        Ok(())
    }

    fn interpolate(&self, values: &[Coefficient<F>]) -> Result<AffineFunction<F>> {
        check_values(values, self.size)?;
        let coefficient = |mask: usize| {
            values
                .iter()
                .enumerate()
                .fold(Coefficient::zero(), |acc, (value, &output)| {
                    let sign = if (mask & value).count_ones().is_multiple_of(2) {
                        1
                    } else {
                        -1
                    };
                    acc + output.scale_integer(sign)
                })
                .div_usize(self.size)
        };
        Ok(AffineFunction {
            bias: coefficient(0),
            weights: (1..self.size).map(coefficient).collect(),
        })
    }

    fn native_operation(&self) -> Option<NativeOperation> {
        Some(NativeOperation::Xor)
    }

    fn native_product(&self, lhs: usize, rhs: usize) -> Option<usize> {
        (lhs < self.size && rhs < self.size).then_some(lhs ^ rhs)
    }

    fn cleaning_mode(&self) -> Option<CleaningMode> {
        Some(CleaningMode::Sign)
    }
}

fn is_prime(value: usize) -> bool {
    if value < 2 {
        return false;
    }
    if value.is_multiple_of(2) {
        return value == 2;
    }
    let mut divisor = 3;
    while divisor <= value / divisor {
        if value.is_multiple_of(divisor) {
            return false;
        }
        divisor += 2;
    }
    true
}

fn primitive_root(prime: usize) -> Result<usize> {
    ensure!(is_prime(prime), "LBRU modulus {prime} is not prime");
    if prime == 2 {
        return Ok(1);
    }
    let order = prime - 1;
    let factors = distinct_prime_factors(order);
    (2..prime)
        .find(|&candidate| {
            factors
                .iter()
                .all(|&factor| pow_mod(candidate, order / factor, prime) != 1)
        })
        .ok_or_else(|| anyhow::anyhow!("no primitive root found modulo {prime}"))
}

fn distinct_prime_factors(mut value: usize) -> Vec<usize> {
    let mut factors = Vec::new();
    let mut divisor = 2;
    while divisor <= value / divisor {
        if value.is_multiple_of(divisor) {
            factors.push(divisor);
            while value.is_multiple_of(divisor) {
                value /= divisor;
            }
        }
        divisor += usize::from(divisor != 2) + 1;
    }
    if value > 1 {
        factors.push(value);
    }
    factors
}

fn mul_mod(lhs: usize, rhs: usize, modulus: usize) -> usize {
    ((lhs as u128 * rhs as u128) % modulus as u128) as usize
}

fn pow_mod(mut base: usize, mut exponent: usize, modulus: usize) -> usize {
    let mut result = 1;
    while exponent != 0 {
        if exponent & 1 == 1 {
            result = mul_mod(result, base, modulus);
        }
        base = mul_mod(base, base, modulus);
        exponent >>= 1;
    }
    result
}
