pub mod algebra;
pub mod scalar;

pub mod prelude {
    pub use crate::algebra::{
        AffineFunction, AffineMap, BlockEncoding, Bru, Coefficient, FinitePoset, Indicator,
        JoinZeta, Lbru, MeetZeta, NativeOperation, Thermometer, WalshHadamard, compile_lut,
    };
    pub use crate::scalar::BlockScalar;
}
