use std::fmt::Debug;

use num_traits::{Float, FloatConst, FromPrimitive, ToPrimitive};

pub trait BlockScalar:
    Float + FloatConst + FromPrimitive + ToPrimitive + Debug + Send + Sync + 'static
{
}

impl<T> BlockScalar for T where
    T: Float + FloatConst + FromPrimitive + ToPrimitive + Debug + Send + Sync + 'static
{
}
