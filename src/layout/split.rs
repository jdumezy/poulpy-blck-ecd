use anyhow::{Result, ensure};
use num_complex::Complex;

use super::{codewords, nearest};
use crate::{algebra::BlockEncoding, scalar::BlockScalar};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SplitLayout {
    slots: usize,
}

impl SplitLayout {
    pub fn new(slots: usize) -> Result<Self> {
        ensure!(
            slots.is_power_of_two(),
            "split slot count must be a power of two"
        );
        Ok(Self { slots })
    }

    pub fn slots(&self) -> usize {
        self.slots
    }

    pub fn encode_slots<F, E>(&self, encoding: &E, values: &[usize]) -> Result<SplitSlots<F>>
    where
        F: BlockScalar,
        E: BlockEncoding<F> + ?Sized,
    {
        ensure!(
            values.len() <= self.slots,
            "{} values exceed split capacity {}",
            values.len(),
            self.slots
        );
        let mut slots = SplitSlots {
            re: vec![vec![F::zero(); self.slots]; encoding.block_size()],
            im: vec![vec![F::zero(); self.slots]; encoding.block_size()],
        };
        self.encode_slots_into(encoding, values, &mut slots)?;
        Ok(slots)
    }

    pub fn encode_slots_into<F, E>(
        &self,
        encoding: &E,
        values: &[usize],
        slots: &mut SplitSlots<F>,
    ) -> Result<()>
    where
        F: BlockScalar,
        E: BlockEncoding<F> + ?Sized,
    {
        ensure!(
            values.len() <= self.slots,
            "{} values exceed split capacity {}",
            values.len(),
            self.slots
        );
        ensure!(
            slots.re.len() == encoding.block_size() && slots.im.len() == encoding.block_size(),
            "split coordinate count does not match encoding width"
        );
        ensure!(
            slots
                .re
                .iter()
                .chain(&slots.im)
                .all(|coordinate| coordinate.len() == self.slots),
            "split slot vector length does not match layout"
        );
        for coordinate in slots.re.iter_mut().chain(&mut slots.im) {
            coordinate.fill(F::zero());
        }
        let mut word = vec![crate::algebra::Coefficient::zero(); encoding.block_size()];
        for (slot, &value) in values.iter().enumerate() {
            encoding.encode_into(value, &mut word)?;
            for (coordinate, &coefficient) in word.iter().enumerate() {
                let coefficient = coefficient.value();
                slots.re[coordinate][slot] = coefficient.re;
                slots.im[coordinate][slot] = coefficient.im;
            }
        }
        Ok(())
    }

    pub fn decode_slots<F, E>(
        &self,
        encoding: &E,
        slots: &SplitSlots<F>,
        count: usize,
    ) -> Result<Vec<usize>>
    where
        F: BlockScalar,
        E: BlockEncoding<F> + ?Sized,
    {
        ensure!(count <= self.slots, "decode count exceeds split capacity");
        ensure!(
            slots.re.len() == encoding.block_size() && slots.im.len() == encoding.block_size(),
            "split coordinate count does not match encoding width"
        );
        ensure!(
            slots
                .re
                .iter()
                .chain(&slots.im)
                .all(|coordinate| coordinate.len() == self.slots),
            "split slot vector length does not match layout"
        );
        let codewords = codewords(encoding)?;
        (0..count)
            .map(|slot| {
                let value = (0..encoding.block_size())
                    .map(|coordinate| {
                        Complex::new(slots.re[coordinate][slot], slots.im[coordinate][slot])
                    })
                    .collect::<Vec<_>>();
                nearest(&value, &codewords)
            })
            .collect()
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct SplitSlots<F> {
    pub re: Vec<Vec<F>>,
    pub im: Vec<Vec<F>>,
}
