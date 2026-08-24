use anyhow::{Result, ensure};

use super::{AffineFunction, Coefficient};
use crate::scalar::BlockScalar;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NativeOperation {
    AddMod,
    MulMod,
    Xor,
    Min,
    Meet,
    Join,
}

pub trait BlockEncoding<F: BlockScalar> {
    fn alphabet_size(&self) -> usize;

    fn block_size(&self) -> usize {
        self.alphabet_size() - 1
    }

    fn encode(&self, value: usize) -> Result<Vec<Coefficient<F>>>;

    fn interpolate(&self, values: &[Coefficient<F>]) -> Result<AffineFunction<F>>;

    fn native_operation(&self) -> Option<NativeOperation> {
        None
    }

    fn native_product(&self, _lhs: usize, _rhs: usize) -> Option<usize> {
        None
    }

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
