use anyhow::{Result, ensure};
use num_complex::Complex;

use super::{codewords, nearest};
use crate::{algebra::BlockEncoding, scalar::BlockScalar};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PackedLayout {
    slots: usize,
    block_width: usize,
}

impl PackedLayout {
    pub fn new(slots: usize, block_width: usize) -> Result<Self> {
        ensure!(
            slots.is_power_of_two(),
            "packed slot count must be a power of two"
        );
        ensure!(block_width != 0, "packed block width must be non-zero");
        ensure!(
            block_width <= slots,
            "packed block width {block_width} exceeds {slots} slots"
        );
        Ok(Self { slots, block_width })
    }

    pub fn for_widths(slots: usize, input_width: usize, output_width: usize) -> Result<Self> {
        ensure!(
            input_width != 0 && output_width != 0,
            "block widths must be non-zero"
        );
        Self::new(slots, input_width.max(output_width))
    }

    pub fn slots(&self) -> usize {
        self.slots
    }

    pub fn block_width(&self) -> usize {
        self.block_width
    }

    pub fn block_count(&self) -> usize {
        self.slots / self.block_width
    }

    pub fn used_slots(&self) -> usize {
        self.block_count() * self.block_width
    }

    pub fn encode_slots<F, E>(&self, encoding: &E, values: &[usize]) -> Result<PackedSlots<F>>
    where
        F: BlockScalar,
        E: BlockEncoding<F> + ?Sized,
    {
        ensure!(
            encoding.block_size() <= self.block_width,
            "encoding width {} exceeds packed block width {}",
            encoding.block_size(),
            self.block_width
        );
        ensure!(
            values.len() <= self.block_count(),
            "{} values exceed packed capacity {}",
            values.len(),
            self.block_count()
        );
        let mut slots = PackedSlots {
            re: vec![F::zero(); self.slots],
            im: vec![F::zero(); self.slots],
        };
        self.encode_slots_into(encoding, values, &mut slots)?;
        Ok(slots)
    }

    pub fn encode_slots_into<F, E>(
        &self,
        encoding: &E,
        values: &[usize],
        slots: &mut PackedSlots<F>,
    ) -> Result<()>
    where
        F: BlockScalar,
        E: BlockEncoding<F> + ?Sized,
    {
        ensure!(
            slots.re.len() == self.slots && slots.im.len() == self.slots,
            "packed slot vector length does not match layout"
        );
        ensure!(
            encoding.block_size() <= self.block_width,
            "encoding width {} exceeds packed block width {}",
            encoding.block_size(),
            self.block_width
        );
        ensure!(
            values.len() <= self.block_count(),
            "{} values exceed packed capacity {}",
            values.len(),
            self.block_count()
        );
        slots.re.fill(F::zero());
        slots.im.fill(F::zero());
        let mut word = vec![crate::algebra::Coefficient::zero(); encoding.block_size()];
        for (block, &value) in values.iter().enumerate() {
            encoding.encode_into(value, &mut word)?;
            for (coordinate, &coefficient) in word.iter().enumerate() {
                let slot = block * self.block_width + coordinate;
                let coefficient = coefficient.value();
                slots.re[slot] = coefficient.re;
                slots.im[slot] = coefficient.im;
            }
        }
        Ok(())
    }

    pub fn decode_slots<F, E>(
        &self,
        encoding: &E,
        slots: &PackedSlots<F>,
        count: usize,
    ) -> Result<Vec<usize>>
    where
        F: BlockScalar,
        E: BlockEncoding<F> + ?Sized,
    {
        ensure!(
            slots.re.len() == self.slots && slots.im.len() == self.slots,
            "packed slot vector length does not match layout"
        );
        ensure!(
            count <= self.block_count(),
            "decode count exceeds packed capacity"
        );
        ensure!(
            encoding.block_size() <= self.block_width,
            "encoding width {} exceeds packed block width {}",
            encoding.block_size(),
            self.block_width
        );
        let codewords = codewords(encoding)?;
        (0..count)
            .map(|block| {
                let start = block * self.block_width;
                let value = (0..encoding.block_size())
                    .map(|coordinate| {
                        Complex::new(slots.re[start + coordinate], slots.im[start + coordinate])
                    })
                    .collect::<Vec<_>>();
                nearest(&value, &codewords)
            })
            .collect()
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct PackedSlots<F> {
    pub re: Vec<F>,
    pub im: Vec<F>,
}
