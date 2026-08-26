use anyhow::{Result, ensure};

use super::{AffineFunction, Coefficient};
use crate::scalar::BlockScalar;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
/// Binary operation represented natively by coordinatewise block arithmetic.
pub enum NativeOperation {
    /// Addition modulo the alphabet size.
    AddMod,
    /// Multiplication modulo the alphabet size.
    MulMod,
    /// Bitwise exclusive-or.
    Xor,
    /// Minimum in a total order.
    Min,
    /// Meet in a finite lattice.
    Meet,
    /// Join in a finite lattice.
    Join,
    /// Equality of indicator-encoded symbols.
    EqualityGate,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
/// Cubic projection used to restore an encoding's coordinate set.
pub enum CleaningMode {
    /// Projects approximate coordinates toward zero or one.
    Binary,
    /// Projects approximate coordinates toward minus or plus one.
    Sign,
}

/// Maps a finite alphabet to a complex coordinate block with affine interpolation.
pub trait BlockEncoding<F: BlockScalar> {
    /// Returns the number of symbols in the encoded alphabet.
    fn alphabet_size(&self) -> usize;

    /// Returns the number of coordinates in one encoded block.
    fn block_size(&self) -> usize {
        self.alphabet_size() - 1
    }

    /// Encodes one symbol into a newly allocated coordinate vector.
    fn encode(&self, value: usize) -> Result<Vec<Coefficient<F>>> {
        let mut output = vec![Coefficient::zero(); self.block_size()];
        self.encode_into(value, &mut output)?;
        Ok(output)
    }

    /// Encodes one symbol into the supplied coordinate vector.
    fn encode_into(&self, value: usize, output: &mut [Coefficient<F>]) -> Result<()>;

    /// Interpolates an affine function over all symbols of the alphabet.
    fn interpolate(&self, values: &[Coefficient<F>]) -> Result<AffineFunction<F>>;

    /// Returns the operation implemented by coordinatewise multiplication, if any.
    fn native_operation(&self) -> Option<NativeOperation> {
        None
    }

    /// Computes the native product of two symbols, if one is defined.
    fn native_product(&self, _lhs: usize, _rhs: usize) -> Option<usize> {
        None
    }

    /// Returns the coordinate projection supported by this encoding, if any.
    fn cleaning_mode(&self) -> Option<CleaningMode> {
        None
    }

    /// Returns half the minimum infinity-norm distance between distinct codewords.
    fn decoding_radius(&self) -> Result<F>
    where
        Self: Sized,
    {
        let size = self.alphabet_size();
        ensure!(size >= 2, "an encoding needs at least two symbols");
        let codewords = (0..size)
            .map(|value| self.encode(value))
            .collect::<Result<Vec<_>>>()?;
        let mut distance = F::infinity();
        for lhs in 0..size {
            for rhs in lhs + 1..size {
                let pair_distance = codewords[lhs]
                    .iter()
                    .zip(&codewords[rhs])
                    .map(|(&a, &b)| (a.value() - b.value()).norm())
                    .fold(F::zero(), F::max);
                distance = distance.min(pair_distance);
            }
        }
        Ok(distance / F::from_u8(2).expect("BlockScalar must represent 2"))
    }
}
