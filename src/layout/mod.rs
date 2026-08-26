//! Host-side layouts for placing encoded blocks in CKKS slots.
//!
//! Packed layouts place every coordinate in one ciphertext, whereas split layouts use one
//! ciphertext per coordinate; both provide matching encode and nearest-codeword decode helpers.

mod packed;
mod split;

use anyhow::{Result, ensure};
use num_complex::Complex;

use crate::{algebra::BlockEncoding, scalar::BlockScalar};

pub use packed::{PackedLayout, PackedSlots};
pub use split::{SplitLayout, SplitSlots};

fn codewords<F, E>(encoding: &E) -> Result<Vec<Vec<Complex<F>>>>
where
    F: BlockScalar,
    E: BlockEncoding<F> + ?Sized,
{
    (0..encoding.alphabet_size())
        .map(|value| {
            encoding.encode(value).map(|word| {
                word.into_iter()
                    .map(|coefficient| coefficient.value())
                    .collect()
            })
        })
        .collect()
}

fn nearest<F: BlockScalar>(value: &[Complex<F>], codewords: &[Vec<Complex<F>>]) -> Result<usize> {
    ensure!(
        !codewords.is_empty(),
        "cannot decode with an empty codebook"
    );
    ensure!(
        codewords.iter().all(|word| word.len() == value.len()),
        "codeword width does not match layout width"
    );
    Ok(codewords
        .iter()
        .enumerate()
        .map(|(symbol, word)| {
            let distance = value
                .iter()
                .zip(word)
                .map(|(&lhs, &rhs)| (lhs - rhs).norm())
                .fold(F::zero(), F::max);
            (symbol, distance)
        })
        .min_by(|lhs, rhs| {
            lhs.1
                .partial_cmp(&rhs.1)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .expect("codebook is non-empty")
        .0)
}
