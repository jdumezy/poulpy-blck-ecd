//! Algebraic descriptions of finite-alphabet block encodings and lookup tables.
//!
//! Encodings map symbols to short complex vectors, while affine and tensor maps compile
//! univariate and multivariate tables into forms consumed by the CKKS circuit plans.

mod affine;
mod character;
mod coefficient;
mod encoding;
mod indicator;
mod poset;
mod tensor;
mod zeta;

#[cfg(test)]
mod tests;

pub use affine::{AffineFunction, AffineMap, compile_coordinates, compile_lut};
pub use character::{Bru, Lbru, WalshHadamard};
pub use coefficient::Coefficient;
pub use encoding::{BlockEncoding, CleaningMode, NativeOperation};
pub use indicator::{Indicator, Thermometer};
pub use poset::FinitePoset;
pub use tensor::{TensorMap, compile_multivariate_coordinates, compile_multivariate_lut};
pub use zeta::{JoinZeta, MeetZeta};
