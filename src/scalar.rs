//! Scalar bounds shared by block encodings and compiled coefficient maps.

use std::fmt::Debug;

use num_traits::{Float, FloatConst, FromPrimitive, ToPrimitive};

/// Floating-point scalar supported by block encodings and their coefficient algebra.
pub trait BlockScalar:
    Float + FloatConst + FromPrimitive + ToPrimitive + Debug + Send + Sync + 'static
{
}

impl<T> BlockScalar for T where
    T: Float + FloatConst + FromPrimitive + ToPrimitive + Debug + Send + Sync + 'static
{
}
