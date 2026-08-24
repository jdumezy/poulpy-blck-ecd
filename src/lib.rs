pub mod algebra;
pub mod layout;
pub mod scalar;
pub mod transform;

#[cfg(any(test, feature = "test-utils"))]
pub mod test_suite;

#[cfg(test)]
mod backend_tests;

pub mod prelude {
    pub use crate::algebra::{
        AffineFunction, AffineMap, BlockEncoding, Bru, CleaningMode, Coefficient, FinitePoset,
        Indicator, JoinZeta, Lbru, MeetZeta, NativeOperation, TensorMap, Thermometer,
        WalshHadamard, compile_lut, compile_multivariate_lut,
    };
    pub use crate::layout::{PackedLayout, PackedSlots, SplitLayout, SplitSlots};
    pub use crate::scalar::BlockScalar;
    pub use crate::transform::{
        CKKSBlockCleaningOps, CKKSBlockMulOps, CKKSCleaningCircuitOps, CKKSMultivariateOps,
        CKKSPackedAffineOps, CKKSSplitAffineOps, PackedAffinePlan, PackedAffineWorkspace,
        PackedCleaningPlan, PackedCleaningWorkspace, PackedMultivariatePlan,
        PackedMultivariateWorkspace, SplitAffinePlan, SplitCleaningPlan, SplitCleaningWorkspace,
        SplitMultivariatePlan, TransformStrategy,
    };
}
