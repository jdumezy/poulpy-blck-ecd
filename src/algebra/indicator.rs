use anyhow::{Result, ensure};

use super::{AffineFunction, BlockEncoding, CleaningMode, Coefficient, NativeOperation};
use crate::scalar::BlockScalar;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
/// One-hot encoding with one redundant coordinate omitted.
pub struct Indicator {
    size: usize,
    omitted: usize,
}

impl Indicator {
    /// Constructs an indicator encoding and selects the omitted symbol.
    pub fn new(size: usize, omitted: usize) -> Result<Self> {
        ensure!(
            size >= 2,
            "indicator alphabet must contain at least two values"
        );
        ensure!(
            omitted < size,
            "omitted indicator {omitted} is out of range"
        );
        Ok(Self { size, omitted })
    }

    /// Returns the symbol whose indicator coordinate is omitted.
    pub fn omitted(&self) -> usize {
        self.omitted
    }
}

impl<F: BlockScalar> BlockEncoding<F> for Indicator {
    fn alphabet_size(&self) -> usize {
        self.size
    }

    fn encode_into(&self, value: usize, output: &mut [Coefficient<F>]) -> Result<()> {
        ensure!(value < self.size, "indicator value {value} is out of range");
        ensure!(
            output.len() == self.size - 1,
            "indicator output has width {}, expected {}",
            output.len(),
            self.size - 1
        );
        for (slot, coordinate) in (0..self.size)
            .filter(|&coordinate| coordinate != self.omitted)
            .enumerate()
        {
            output[slot] = Coefficient::integer(i128::from(coordinate == value));
        }
        Ok(())
    }

    fn interpolate(&self, values: &[Coefficient<F>]) -> Result<AffineFunction<F>> {
        ensure!(
            values.len() == self.size,
            "function has {} values, expected {}",
            values.len(),
            self.size
        );
        let bias = values[self.omitted];
        Ok(AffineFunction {
            bias,
            weights: (0..self.size)
                .filter(|&coordinate| coordinate != self.omitted)
                .map(|coordinate| values[coordinate] - bias)
                .collect(),
        })
    }

    fn cleaning_mode(&self) -> Option<CleaningMode> {
        Some(CleaningMode::Binary)
    }

    fn native_operation(&self) -> Option<NativeOperation> {
        Some(NativeOperation::EqualityGate)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
/// Cumulative binary encoding of a totally ordered alphabet.
pub struct Thermometer {
    size: usize,
}

impl Thermometer {
    /// Constructs a thermometer encoding for the requested alphabet size.
    pub fn new(size: usize) -> Result<Self> {
        ensure!(
            size >= 2,
            "thermometer alphabet must contain at least two values"
        );
        Ok(Self { size })
    }
}

impl<F: BlockScalar> BlockEncoding<F> for Thermometer {
    fn alphabet_size(&self) -> usize {
        self.size
    }

    fn encode_into(&self, value: usize, output: &mut [Coefficient<F>]) -> Result<()> {
        ensure!(
            value < self.size,
            "thermometer value {value} is out of range"
        );
        ensure!(
            output.len() == self.size - 1,
            "thermometer output has width {}, expected {}",
            output.len(),
            self.size - 1
        );
        for (coordinate, coefficient) in output.iter_mut().enumerate() {
            *coefficient = Coefficient::integer(i128::from(value > coordinate));
        }
        Ok(())
    }

    fn interpolate(&self, values: &[Coefficient<F>]) -> Result<AffineFunction<F>> {
        ensure!(
            values.len() == self.size,
            "function has {} values, expected {}",
            values.len(),
            self.size
        );
        Ok(AffineFunction {
            bias: values[0],
            weights: (1..self.size)
                .map(|value| values[value] - values[value - 1])
                .collect(),
        })
    }

    fn native_operation(&self) -> Option<NativeOperation> {
        Some(NativeOperation::Min)
    }

    fn native_product(&self, lhs: usize, rhs: usize) -> Option<usize> {
        (lhs < self.size && rhs < self.size).then_some(lhs.min(rhs))
    }

    fn cleaning_mode(&self) -> Option<CleaningMode> {
        Some(CleaningMode::Binary)
    }
}
