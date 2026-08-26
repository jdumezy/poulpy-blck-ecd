use anyhow::{Result, ensure};
use num_complex::Complex;

use super::{codewords, nearest};
use crate::{algebra::BlockEncoding, scalar::BlockScalar};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
/// Placement of several complete encoded blocks within one CKKS ciphertext.
pub struct PackedLayout {
    slots: usize,
    block_width: usize,
    lane_width: usize,
    interleaved: bool,
}

impl PackedLayout {
    /// Constructs the default interleaved layout.
    pub fn new(slots: usize, block_width: usize) -> Result<Self> {
        Self::interleaved(slots, block_width)
    }

    /// Constructs a coordinate-major layout with power-of-two lane padding.
    pub fn interleaved(slots: usize, block_width: usize) -> Result<Self> {
        ensure!(
            slots.is_power_of_two(),
            "packed slot count must be a power of two"
        );
        ensure!(block_width != 0, "packed block width must be non-zero");
        ensure!(
            block_width <= slots,
            "packed block width {block_width} exceeds {slots} slots"
        );
        let lane_width = block_width.next_power_of_two();
        Ok(Self {
            slots,
            block_width,
            lane_width,
            interleaved: true,
        })
    }

    /// Block-major packing without power-of-two padding.
    pub fn contiguous(slots: usize, block_width: usize) -> Result<Self> {
        ensure!(
            slots.is_power_of_two(),
            "packed slot count must be a power of two"
        );
        ensure!(block_width != 0, "packed block width must be non-zero");
        ensure!(
            block_width <= slots,
            "packed block width {block_width} exceeds {slots} slots"
        );
        Ok(Self {
            slots,
            block_width,
            lane_width: block_width,
            interleaved: false,
        })
    }

    /// Constructs a layout wide enough for both input and output blocks.
    pub fn for_widths(slots: usize, input_width: usize, output_width: usize) -> Result<Self> {
        ensure!(
            input_width != 0 && output_width != 0,
            "block widths must be non-zero"
        );
        Self::new(slots, input_width.max(output_width))
    }

    /// Returns the total number of complex CKKS slots.
    pub fn slots(&self) -> usize {
        self.slots
    }

    /// Returns the number of active coordinates reserved per block.
    pub fn block_width(&self) -> usize {
        self.block_width
    }

    /// Power-of-two coordinate lane count used by interleaved layouts.
    pub fn lane_width(&self) -> usize {
        self.lane_width
    }

    /// Reports whether coordinates are stored in separate interleaved lanes.
    pub fn is_interleaved(&self) -> bool {
        self.interleaved
    }

    /// Returns the number of complete blocks that fit in the layout.
    pub fn block_count(&self) -> usize {
        self.slots
            / if self.interleaved {
                self.lane_width
            } else {
                self.block_width
            }
    }

    /// Returns the number of slots occupied by active block coordinates.
    pub fn used_slots(&self) -> usize {
        self.block_count() * self.block_width
    }

    /// Returns the slot index for one block coordinate.
    pub fn slot(&self, block: usize, coordinate: usize) -> usize {
        assert!(
            block < self.block_count(),
            "packed block index out of range"
        );
        assert!(
            coordinate < self.block_width,
            "packed coordinate out of range"
        );
        if self.interleaved {
            coordinate * self.block_count() + block
        } else {
            block * self.block_width + coordinate
        }
    }

    /// Encodes symbols into newly allocated real and imaginary slot vectors.
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

    /// Encodes symbols into existing real and imaginary slot vectors.
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
                let slot = self.slot(block, coordinate);
                let coefficient = coefficient.value();
                slots.re[slot] = coefficient.re;
                slots.im[slot] = coefficient.im;
            }
        }
        Ok(())
    }

    /// Decodes the requested number of blocks by nearest codeword.
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
                let value = (0..encoding.block_size())
                    .map(|coordinate| {
                        let slot = self.slot(block, coordinate);
                        Complex::new(slots.re[slot], slots.im[slot])
                    })
                    .collect::<Vec<_>>();
                nearest(&value, &codewords)
            })
            .collect()
    }
}

#[derive(Clone, Debug, PartialEq)]
/// Real and imaginary CKKS slot vectors for a packed layout.
pub struct PackedSlots<F> {
    /// Real slot components.
    pub re: Vec<F>,
    /// Imaginary slot components.
    pub im: Vec<F>,
}
