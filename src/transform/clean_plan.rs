use std::collections::BTreeSet;

use anyhow::{Result, ensure};
use poulpy_ckks::{
    CKKSAtkBounds, CKKSCtBounds, CoeffsMeta, SetCKKSInfos,
    api::{CKKSCopyOps, CKKSEncodingOps, CKKSEncodingScalar, CKKSLinearTransformationOps},
    layouts::{CKKSCiphertextOwned, CKKSModuleAlloc},
};
use poulpy_core::layouts::{
    GGLWEInfos, GLWEAutomorphismKeyHelper, GLWEInfos, GLWEToBackendMut, GLWEToBackendRef,
    prepared::{GGLWEPreparedToBackendRef, GLWETensorKeyPreparedToBackendRef},
};
use poulpy_hal::{
    api::{CnvPVecAlloc, ModuleN},
    layouts::{Backend, Module, ScratchArena},
};

use super::{
    CKKSBlockCleaningOps, CKKSPackedAffineOps, CKKSSplitAffineOps, PackedAffinePlan,
    PackedAffineWorkspace, SplitAffinePlan, TransformStrategy,
};
use crate::{
    algebra::{AffineMap, BlockEncoding, CleaningMode, Coefficient, Indicator, compile_lut},
    layout::PackedLayout,
    scalar::BlockScalar,
};

/// Prepared packed-ciphertext circuit for cleaning and optionally changing an encoding.
pub struct PackedCleaningPlan<BE: Backend> {
    mode: CleaningMode,
    clean_width: usize,
    output_width: usize,
    pre: Option<PackedAffinePlan<BE>>,
    post: Option<PackedAffinePlan<BE>>,
    galois_elements: Vec<i64>,
}

impl<BE: Backend> PackedCleaningPlan<BE> {
    #[allow(clippy::too_many_arguments)]
    /// Compiles pre- and post-transforms around the encoding's cleaning polynomial.
    pub fn compile<F>(
        module: &Module<BE>,
        layout: PackedLayout,
        input: &dyn BlockEncoding<F>,
        output: &dyn BlockEncoding<F>,
        base2k: poulpy_core::layouts::Base2K,
        coeffs_meta: CoeffsMeta,
        strategy: TransformStrategy,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<Self>
    where
        F: BlockScalar + CKKSEncodingScalar,
        Module<BE>: CKKSModuleAlloc<BE>
            + CKKSEncodingOps<BE, F>
            + CKKSLinearTransformationOps<BE>
            + CnvPVecAlloc<BE>
            + ModuleN,
    {
        let maps = cleaning_maps(input, output)?;
        ensure!(
            maps.clean_width <= layout.block_width() && output.block_size() <= layout.block_width(),
            "cleaning widths exceed packed block width {}",
            layout.block_width()
        );
        let pre = maps
            .pre
            .as_ref()
            .map(|map| {
                PackedAffinePlan::compile(
                    module,
                    layout,
                    map,
                    base2k,
                    coeffs_meta,
                    strategy,
                    scratch,
                )
            })
            .transpose()?;
        let post = maps
            .post
            .as_ref()
            .map(|map| {
                PackedAffinePlan::compile(
                    module,
                    layout,
                    map,
                    base2k,
                    coeffs_meta,
                    strategy,
                    scratch,
                )
            })
            .transpose()?;
        let galois_elements = pre
            .iter()
            .flat_map(PackedAffinePlan::galois_elements)
            .chain(post.iter().flat_map(PackedAffinePlan::galois_elements))
            .copied()
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect();
        Ok(Self {
            mode: maps.mode,
            clean_width: maps.clean_width,
            output_width: output.block_size(),
            pre,
            post,
            galois_elements,
        })
    }

    /// Returns the cubic projection selected by the input encoding.
    pub fn cleaning_mode(&self) -> CleaningMode {
        self.mode
    }

    /// Returns the number of coordinates processed by the cleaning stage.
    pub fn cleaning_width(&self) -> usize {
        self.clean_width
    }

    /// Returns the number of output block coordinates produced by the plan.
    pub fn output_width(&self) -> usize {
        self.output_width
    }

    /// Returns the Galois elements required by the pre- and post-transforms.
    pub fn galois_elements(&self) -> &[i64] {
        &self.galois_elements
    }

    /// Allocates reusable intermediate ciphertexts and affine workspaces.
    pub fn alloc_workspace<C>(&self, module: &Module<BE>, input: &C) -> PackedCleaningWorkspace<BE>
    where
        Module<BE>: CKKSModuleAlloc<BE> + CnvPVecAlloc<BE>,
        C: GLWEInfos + poulpy_ckks::CKKSInfos,
    {
        PackedCleaningWorkspace {
            pre: self
                .pre
                .as_ref()
                .map(|plan| plan.alloc_workspace(module, input)),
            post: self
                .post
                .as_ref()
                .map(|plan| plan.alloc_workspace(module, input)),
            stage: self
                .pre
                .as_ref()
                .map(|_| module.ckks_ciphertext_alloc_from_infos(input)),
            cleaned: self
                .post
                .as_ref()
                .map(|_| module.ckks_ciphertext_alloc_from_infos(input)),
        }
    }
}

/// Reusable storage for evaluating a packed cleaning plan.
pub struct PackedCleaningWorkspace<BE: Backend> {
    pre: Option<PackedAffineWorkspace<BE>>,
    post: Option<PackedAffineWorkspace<BE>>,
    stage: Option<CKKSCiphertextOwned<BE>>,
    cleaned: Option<CKKSCiphertextOwned<BE>>,
}

/// Prepared split-ciphertext circuit for cleaning and optionally changing an encoding.
pub struct SplitCleaningPlan<BE: Backend> {
    mode: CleaningMode,
    clean_width: usize,
    output_width: usize,
    pre: Option<SplitAffinePlan<BE>>,
    post: Option<SplitAffinePlan<BE>>,
}

impl<BE: Backend> SplitCleaningPlan<BE> {
    /// Compiles split pre- and post-transforms around the cleaning polynomial.
    pub fn compile<F>(
        module: &Module<BE>,
        input: &dyn BlockEncoding<F>,
        output: &dyn BlockEncoding<F>,
        base2k: poulpy_core::layouts::Base2K,
        coeffs_meta: CoeffsMeta,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<Self>
    where
        F: BlockScalar + CKKSEncodingScalar,
        Module<BE>: CKKSModuleAlloc<BE> + CKKSEncodingOps<BE, F> + ModuleN,
    {
        let maps = cleaning_maps(input, output)?;
        Ok(Self {
            mode: maps.mode,
            clean_width: maps.clean_width,
            output_width: output.block_size(),
            pre: maps
                .pre
                .as_ref()
                .map(|map| SplitAffinePlan::compile(module, map, base2k, coeffs_meta, scratch))
                .transpose()?,
            post: maps
                .post
                .as_ref()
                .map(|map| SplitAffinePlan::compile(module, map, base2k, coeffs_meta, scratch))
                .transpose()?,
        })
    }

    /// Returns the cubic projection selected by the input encoding.
    pub fn cleaning_mode(&self) -> CleaningMode {
        self.mode
    }

    /// Returns the number of coordinates processed by the cleaning stage.
    pub fn cleaning_width(&self) -> usize {
        self.clean_width
    }

    /// Returns the number of output coordinate ciphertexts produced by the plan.
    pub fn output_width(&self) -> usize {
        self.output_width
    }

    /// Allocates reusable intermediate coordinate ciphertexts.
    pub fn alloc_workspace<C>(&self, module: &Module<BE>, input: &C) -> SplitCleaningWorkspace<BE>
    where
        Module<BE>: CKKSModuleAlloc<BE>,
        C: GLWEInfos + poulpy_ckks::CKKSInfos,
    {
        SplitCleaningWorkspace {
            stage: self.pre.as_ref().map(|_| {
                (0..self.clean_width)
                    .map(|_| module.ckks_ciphertext_alloc_from_infos(input))
                    .collect()
            }),
            cleaned: self.post.as_ref().map(|_| {
                (0..self.clean_width)
                    .map(|_| module.ckks_ciphertext_alloc_from_infos(input))
                    .collect()
            }),
        }
    }
}

/// Reusable storage for evaluating a split cleaning plan.
pub struct SplitCleaningWorkspace<BE: Backend> {
    stage: Option<Vec<CKKSCiphertextOwned<BE>>>,
    cleaned: Option<Vec<CKKSCiphertextOwned<BE>>>,
}

/// Planned cleaning-circuit operations implemented by compatible Poulpy modules.
pub trait CKKSCleaningCircuitOps<BE: Backend> {
    /// Returns the scratch-memory requirement for a packed cleaning circuit.
    fn ckks_packed_cleaning_tmp_bytes<C, K, T>(
        &self,
        input: &C,
        plan: &PackedCleaningPlan<BE>,
        automorphism_key: &K,
        tensor_key: &T,
    ) -> usize
    where
        C: CKKSCtBounds,
        K: GGLWEInfos,
        T: GGLWEInfos;

    #[allow(clippy::too_many_arguments)]
    /// Evaluates a packed cleaning plan into one output ciphertext.
    fn ckks_packed_cleaning_into<Dst, Src, H, K, T>(
        &self,
        output: &mut Dst,
        input: &Src,
        plan: &PackedCleaningPlan<BE>,
        workspace: &mut PackedCleaningWorkspace<BE>,
        automorphism_keys: &H,
        tensor_key: &T,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
        Src: GLWEToBackendRef<BE> + CKKSCtBounds,
        K: CKKSAtkBounds<BE>,
        H: GLWEAutomorphismKeyHelper<K, BE>,
        T: GGLWEInfos + GLWETensorKeyPreparedToBackendRef<BE> + GGLWEPreparedToBackendRef<BE>;

    /// Evaluates a split cleaning plan into output coordinate ciphertexts.
    fn ckks_split_cleaning_into<Dst, Src, T>(
        &self,
        outputs: &mut [Dst],
        inputs: &[Src],
        plan: &SplitCleaningPlan<BE>,
        workspace: &mut SplitCleaningWorkspace<BE>,
        tensor_key: &T,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
        Src: GLWEToBackendRef<BE> + CKKSCtBounds,
        T: GGLWEInfos + GLWETensorKeyPreparedToBackendRef<BE> + GGLWEPreparedToBackendRef<BE>;

    /// Returns the scratch-memory requirement for a split cleaning circuit.
    fn ckks_split_cleaning_tmp_bytes<C, T>(
        &self,
        input: &C,
        plan: &SplitCleaningPlan<BE>,
        tensor_key: &T,
    ) -> usize
    where
        C: CKKSCtBounds,
        T: GGLWEInfos;
}

impl<BE: Backend> CKKSCleaningCircuitOps<BE> for Module<BE>
where
    Module<BE>: CKKSBlockCleaningOps<BE>
        + CKKSCopyOps<BE>
        + CKKSPackedAffineOps<BE>
        + CKKSSplitAffineOps<BE>,
{
    fn ckks_packed_cleaning_tmp_bytes<C, K, T>(
        &self,
        input: &C,
        plan: &PackedCleaningPlan<BE>,
        automorphism_key: &K,
        tensor_key: &T,
    ) -> usize
    where
        C: CKKSCtBounds,
        K: GGLWEInfos,
        T: GGLWEInfos,
    {
        self.ckks_clean_tmp_bytes(input, input, tensor_key).max(
            plan.pre
                .iter()
                .map(|affine| self.ckks_packed_affine_tmp_bytes(affine, input, automorphism_key))
                .chain(plan.post.iter().map(|affine| {
                    self.ckks_packed_affine_tmp_bytes(affine, input, automorphism_key)
                }))
                .max()
                .unwrap_or(0),
        )
    }

    fn ckks_packed_cleaning_into<Dst, Src, H, K, T>(
        &self,
        output: &mut Dst,
        input: &Src,
        plan: &PackedCleaningPlan<BE>,
        workspace: &mut PackedCleaningWorkspace<BE>,
        automorphism_keys: &H,
        tensor_key: &T,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
        Src: GLWEToBackendRef<BE> + CKKSCtBounds,
        K: CKKSAtkBounds<BE>,
        H: GLWEAutomorphismKeyHelper<K, BE>,
        T: GGLWEInfos + GLWETensorKeyPreparedToBackendRef<BE> + GGLWEPreparedToBackendRef<BE>,
    {
        match (&plan.pre, &plan.post) {
            (Some(pre), Some(post)) => {
                let stage = workspace
                    .stage
                    .as_mut()
                    .expect("cleaning stage is allocated");
                self.ckks_packed_affine_into(
                    stage,
                    input,
                    pre,
                    workspace.pre.as_mut().unwrap(),
                    automorphism_keys,
                    scratch,
                )?;
                let cleaned = workspace
                    .cleaned
                    .as_mut()
                    .expect("cleaning output stage is allocated");
                self.ckks_clean_into(cleaned, stage, plan.mode, tensor_key, scratch)?;
                self.ckks_packed_affine_into(
                    output,
                    cleaned,
                    post,
                    workspace.post.as_mut().unwrap(),
                    automorphism_keys,
                    scratch,
                )
            }
            (Some(pre), None) => {
                let stage = workspace
                    .stage
                    .as_mut()
                    .expect("cleaning stage is allocated");
                self.ckks_packed_affine_into(
                    stage,
                    input,
                    pre,
                    workspace.pre.as_mut().unwrap(),
                    automorphism_keys,
                    scratch,
                )?;
                self.ckks_clean_into(output, stage, plan.mode, tensor_key, scratch)
            }
            (None, Some(post)) => {
                let cleaned = workspace
                    .cleaned
                    .as_mut()
                    .expect("cleaning output stage is allocated");
                self.ckks_clean_into(cleaned, input, plan.mode, tensor_key, scratch)?;
                self.ckks_packed_affine_into(
                    output,
                    cleaned,
                    post,
                    workspace.post.as_mut().unwrap(),
                    automorphism_keys,
                    scratch,
                )
            }
            (None, None) => self.ckks_clean_into(output, input, plan.mode, tensor_key, scratch),
        }
    }

    fn ckks_split_cleaning_into<Dst, Src, T>(
        &self,
        outputs: &mut [Dst],
        inputs: &[Src],
        plan: &SplitCleaningPlan<BE>,
        workspace: &mut SplitCleaningWorkspace<BE>,
        tensor_key: &T,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
        Src: GLWEToBackendRef<BE> + CKKSCtBounds,
        T: GGLWEInfos + GLWETensorKeyPreparedToBackendRef<BE> + GGLWEPreparedToBackendRef<BE>,
    {
        ensure!(
            outputs.len() == plan.output_width,
            "split cleaning output width does not match plan"
        );
        match (&plan.pre, &plan.post) {
            (Some(pre), Some(post)) => {
                let stage = workspace
                    .stage
                    .as_mut()
                    .expect("cleaning stage is allocated");
                self.ckks_split_affine_into(stage, inputs, pre, scratch)?;
                let cleaned = workspace
                    .cleaned
                    .as_mut()
                    .expect("cleaning output stage is allocated");
                self.ckks_clean_split_into(cleaned, stage, plan.mode, tensor_key, scratch)?;
                self.ckks_split_affine_into(outputs, cleaned, post, scratch)
            }
            (Some(pre), None) => {
                let stage = workspace
                    .stage
                    .as_mut()
                    .expect("cleaning stage is allocated");
                self.ckks_split_affine_into(stage, inputs, pre, scratch)?;
                self.ckks_clean_split_into(outputs, stage, plan.mode, tensor_key, scratch)
            }
            (None, Some(post)) => {
                let cleaned = workspace
                    .cleaned
                    .as_mut()
                    .expect("cleaning output stage is allocated");
                self.ckks_clean_split_into(cleaned, inputs, plan.mode, tensor_key, scratch)?;
                self.ckks_split_affine_into(outputs, cleaned, post, scratch)
            }
            (None, None) => {
                self.ckks_clean_split_into(outputs, inputs, plan.mode, tensor_key, scratch)
            }
        }
    }

    fn ckks_split_cleaning_tmp_bytes<C, T>(
        &self,
        input: &C,
        plan: &SplitCleaningPlan<BE>,
        tensor_key: &T,
    ) -> usize
    where
        C: CKKSCtBounds,
        T: GGLWEInfos,
    {
        self.ckks_clean_tmp_bytes(input, input, tensor_key).max(
            plan.pre
                .iter()
                .map(|affine| self.ckks_split_affine_tmp_bytes(affine, input, input))
                .chain(
                    plan.post
                        .iter()
                        .map(|affine| self.ckks_split_affine_tmp_bytes(affine, input, input)),
                )
                .max()
                .unwrap_or(0),
        )
    }
}

struct CleaningMaps<F> {
    mode: CleaningMode,
    clean_width: usize,
    pre: Option<AffineMap<F>>,
    post: Option<AffineMap<F>>,
}

fn cleaning_maps<F: BlockScalar>(
    input: &dyn BlockEncoding<F>,
    output: &dyn BlockEncoding<F>,
) -> Result<CleaningMaps<F>> {
    ensure!(
        input.alphabet_size() == output.alphabet_size(),
        "cleaning cannot change alphabet size"
    );
    let table = (0..input.alphabet_size()).collect::<Vec<_>>();
    if let Some(mode) = input.cleaning_mode() {
        return Ok(CleaningMaps {
            mode,
            clean_width: input.block_size(),
            pre: None,
            post: (!same_encoding(input, output)?)
                .then(|| compile_lut(input, output, &table))
                .transpose()?,
        });
    }
    let indicator = Indicator::new(input.alphabet_size(), 0)?;
    Ok(CleaningMaps {
        mode: CleaningMode::Binary,
        clean_width: input.alphabet_size() - 1,
        pre: Some(compile_lut(input, &indicator, &table)?),
        post: (!same_encoding(&indicator, output)?)
            .then(|| compile_lut(&indicator, output, &table))
            .transpose()?,
    })
}

fn same_encoding<F: BlockScalar>(
    lhs: &dyn BlockEncoding<F>,
    rhs: &dyn BlockEncoding<F>,
) -> Result<bool> {
    if lhs.alphabet_size() != rhs.alphabet_size() || lhs.block_size() != rhs.block_size() {
        return Ok(false);
    }
    let mut lhs_word = vec![Coefficient::zero(); lhs.block_size()];
    let mut rhs_word = vec![Coefficient::zero(); rhs.block_size()];
    for value in 0..lhs.alphabet_size() {
        lhs.encode_into(value, &mut lhs_word)?;
        rhs.encode_into(value, &mut rhs_word)?;
        if lhs_word != rhs_word {
            return Ok(false);
        }
    }
    Ok(true)
}
