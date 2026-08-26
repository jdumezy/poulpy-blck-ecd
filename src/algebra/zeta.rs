use anyhow::{Result, ensure};

use super::{
    AffineFunction, BlockEncoding, CleaningMode, Coefficient, FinitePoset, NativeOperation,
};
use crate::scalar::BlockScalar;

#[derive(Clone, Debug, PartialEq, Eq)]
/// Lower-zeta encoding whose native product is the meet operation.
pub struct MeetZeta {
    poset: FinitePoset,
    bottom: usize,
    mobius: Vec<i128>,
    product: Vec<usize>,
}

impl MeetZeta {
    /// Constructs a meet-zeta encoding for a poset with a bottom and all meets.
    pub fn new(poset: FinitePoset) -> Result<Self> {
        let bottom = poset
            .bottom()
            .ok_or_else(|| anyhow::anyhow!("meet-zeta poset has no bottom element"))?;
        let mobius = poset.mobius()?;
        let product = poset.meet_table()?;
        Ok(Self {
            poset,
            bottom,
            mobius,
            product,
        })
    }

    /// Constructs a meet-zeta encoding of a Boolean lattice.
    pub fn boolean(bits: usize) -> Result<Self> {
        Self::new(FinitePoset::boolean_lattice(bits)?)
    }

    /// Constructs a meet-zeta encoding of rooted-tree ancestry.
    pub fn ancestor(parents: &[Option<usize>]) -> Result<Self> {
        Self::new(FinitePoset::rooted_tree(parents)?)
    }

    /// Constructs a divisor meet-zeta encoding and its symbol-to-divisor mapping.
    pub fn divisors(modulus: usize) -> Result<(Self, Vec<usize>)> {
        let (poset, divisors) = FinitePoset::divisor_lattice(modulus)?;
        Ok((Self::new(poset)?, divisors))
    }

    /// Returns the underlying finite poset.
    pub fn poset(&self) -> &FinitePoset {
        &self.poset
    }
}

impl<F: BlockScalar> BlockEncoding<F> for MeetZeta {
    fn alphabet_size(&self) -> usize {
        self.poset.size()
    }

    fn encode_into(&self, value: usize, output: &mut [Coefficient<F>]) -> Result<()> {
        ensure!(
            value < self.poset.size(),
            "meet-zeta value {value} is out of range"
        );
        ensure!(
            output.len() == self.poset.size() - 1,
            "meet-zeta output has width {}, expected {}",
            output.len(),
            self.poset.size() - 1
        );
        for (slot, coordinate) in (0..self.poset.size())
            .filter(|&coordinate| coordinate != self.bottom)
            .enumerate()
        {
            output[slot] = Coefficient::integer(i128::from(self.poset.leq(coordinate, value)));
        }
        Ok(())
    }

    fn interpolate(&self, values: &[Coefficient<F>]) -> Result<AffineFunction<F>> {
        ensure!(
            values.len() == self.poset.size(),
            "function has {} values, expected {}",
            values.len(),
            self.poset.size()
        );
        let coefficient = |coordinate: usize| {
            (0..self.poset.size())
                .filter(|&value| self.poset.leq(value, coordinate))
                .fold(Coefficient::zero(), |acc, value| {
                    acc + values[value]
                        .scale_integer(self.mobius[value * self.poset.size() + coordinate])
                })
        };
        Ok(AffineFunction {
            bias: coefficient(self.bottom),
            weights: (0..self.poset.size())
                .filter(|&coordinate| coordinate != self.bottom)
                .map(coefficient)
                .collect(),
        })
    }

    fn native_operation(&self) -> Option<NativeOperation> {
        Some(NativeOperation::Meet)
    }

    fn native_product(&self, lhs: usize, rhs: usize) -> Option<usize> {
        (lhs < self.poset.size() && rhs < self.poset.size())
            .then_some(self.product[lhs * self.poset.size() + rhs])
    }

    fn cleaning_mode(&self) -> Option<CleaningMode> {
        Some(CleaningMode::Binary)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
/// Upper-zeta encoding whose native product is the join operation.
pub struct JoinZeta {
    poset: FinitePoset,
    top: usize,
    mobius: Vec<i128>,
    product: Vec<usize>,
}

impl JoinZeta {
    /// Constructs a join-zeta encoding for a poset with a top and all joins.
    pub fn new(poset: FinitePoset) -> Result<Self> {
        let top = poset
            .top()
            .ok_or_else(|| anyhow::anyhow!("join-zeta poset has no top element"))?;
        let mobius = poset.mobius()?;
        let product = poset.join_table()?;
        Ok(Self {
            poset,
            top,
            mobius,
            product,
        })
    }

    /// Constructs a join-zeta encoding of a Boolean lattice.
    pub fn boolean(bits: usize) -> Result<Self> {
        Self::new(FinitePoset::boolean_lattice(bits)?)
    }

    /// Constructs a divisor join-zeta encoding and its symbol-to-divisor mapping.
    pub fn divisors(modulus: usize) -> Result<(Self, Vec<usize>)> {
        let (poset, divisors) = FinitePoset::divisor_lattice(modulus)?;
        Ok((Self::new(poset)?, divisors))
    }

    /// Returns the underlying finite poset.
    pub fn poset(&self) -> &FinitePoset {
        &self.poset
    }
}

impl<F: BlockScalar> BlockEncoding<F> for JoinZeta {
    fn alphabet_size(&self) -> usize {
        self.poset.size()
    }

    fn encode_into(&self, value: usize, output: &mut [Coefficient<F>]) -> Result<()> {
        ensure!(
            value < self.poset.size(),
            "join-zeta value {value} is out of range"
        );
        ensure!(
            output.len() == self.poset.size() - 1,
            "join-zeta output has width {}, expected {}",
            output.len(),
            self.poset.size() - 1
        );
        for (slot, coordinate) in (0..self.poset.size())
            .filter(|&coordinate| coordinate != self.top)
            .enumerate()
        {
            output[slot] = Coefficient::integer(i128::from(self.poset.leq(value, coordinate)));
        }
        Ok(())
    }

    fn interpolate(&self, values: &[Coefficient<F>]) -> Result<AffineFunction<F>> {
        ensure!(
            values.len() == self.poset.size(),
            "function has {} values, expected {}",
            values.len(),
            self.poset.size()
        );
        let coefficient = |coordinate: usize| {
            (0..self.poset.size())
                .filter(|&value| self.poset.leq(coordinate, value))
                .fold(Coefficient::zero(), |acc, value| {
                    acc + values[value]
                        .scale_integer(self.mobius[coordinate * self.poset.size() + value])
                })
        };
        Ok(AffineFunction {
            bias: coefficient(self.top),
            weights: (0..self.poset.size())
                .filter(|&coordinate| coordinate != self.top)
                .map(coefficient)
                .collect(),
        })
    }

    fn native_operation(&self) -> Option<NativeOperation> {
        Some(NativeOperation::Join)
    }

    fn native_product(&self, lhs: usize, rhs: usize) -> Option<usize> {
        (lhs < self.poset.size() && rhs < self.poset.size())
            .then_some(self.product[lhs * self.poset.size() + rhs])
    }

    fn cleaning_mode(&self) -> Option<CleaningMode> {
        Some(CleaningMode::Binary)
    }
}
