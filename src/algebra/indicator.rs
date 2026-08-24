use anyhow::{Result, ensure};

use super::{AffineFunction, BlockEncoding, Coefficient, NativeOperation};
use crate::scalar::BlockScalar;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Indicator {
    size: usize,
    omitted: usize,
}

impl Indicator {
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

    pub fn omitted(&self) -> usize {
        self.omitted
    }
}

impl<F: BlockScalar> BlockEncoding<F> for Indicator {
    fn alphabet_size(&self) -> usize {
        self.size
    }

    fn encode(&self, value: usize) -> Result<Vec<Coefficient<F>>> {
        ensure!(value < self.size, "indicator value {value} is out of range");
        Ok((0..self.size)
            .filter(|&coordinate| coordinate != self.omitted)
            .map(|coordinate| Coefficient::integer(i128::from(coordinate == value)))
            .collect())
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
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Thermometer {
    size: usize,
}

impl Thermometer {
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

    fn encode(&self, value: usize) -> Result<Vec<Coefficient<F>>> {
        ensure!(
            value < self.size,
            "thermometer value {value} is out of range"
        );
        Ok((1..self.size)
            .map(|threshold| Coefficient::integer(i128::from(value >= threshold)))
            .collect())
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
}
