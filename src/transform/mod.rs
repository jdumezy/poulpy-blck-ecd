//! Prepared CKKS circuits for evaluating maps and native operations on encoded blocks.
//!
//! Plans precompute backend data and expose reusable workspaces, while operation traits extend
//! Poulpy modules with packed and split evaluation routines.

mod clean;
mod clean_plan;
mod multivariate;
mod native;
mod packed;
mod split;

use anyhow::{Result, ensure};
use poulpy_core::layouts::LinearTransformationStrategy;
use poulpy_core::optimal_bsgs_giant_step;

pub use multivariate::{
    CKKSMultivariateOps, PackedMultivariatePlan, PackedMultivariateWorkspace, SplitMultivariatePlan,
};
pub use native::CKKSBlockMulOps;
pub use packed::{CKKSPackedAffineOps, PackedAffinePlan, PackedAffineWorkspace};
pub use split::{CKKSSplitAffineOps, SplitAffinePlan};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
/// Scheduling strategy for packed linear transformations.
pub enum TransformStrategy {
    /// Selects a strategy and baby-step/giant-step split automatically.
    #[default]
    Auto,
    /// Evaluates each nonzero diagonal directly.
    Direct,
    /// Uses an explicit baby-step/giant-step decomposition.
    Bsgs {
        /// Number of diagonals grouped into each giant step.
        giant_step: usize,
    },
}

impl TransformStrategy {
    fn resolve(self, indexes: &[i64], slots: usize) -> Result<LinearTransformationStrategy> {
        ensure!(
            !indexes.is_empty(),
            "cannot schedule an empty linear transform"
        );
        Ok(match self {
            Self::Auto => LinearTransformationStrategy::Bsgs {
                giant_step: optimal_bsgs_giant_step(indexes.iter().copied(), slots),
            },
            Self::Direct => LinearTransformationStrategy::Direct,
            Self::Bsgs { giant_step } => {
                ensure!(giant_step != 0, "BSGS giant step must be non-zero");
                LinearTransformationStrategy::Bsgs { giant_step }
            }
        })
    }
}
pub use clean::CKKSBlockCleaningOps;
pub use clean_plan::{
    CKKSCleaningCircuitOps, PackedCleaningPlan, PackedCleaningWorkspace, SplitCleaningPlan,
    SplitCleaningWorkspace,
};
