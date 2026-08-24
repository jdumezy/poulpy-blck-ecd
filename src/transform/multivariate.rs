use std::collections::BTreeSet;

use anyhow::{Result, ensure};
use poulpy_ckks::{
    CKKSAtkBounds, CKKSCtBounds, CoeffsMeta, SetCKKSInfos,
    api::{
        CKKSCopyOps, CKKSEncodingOps, CKKSEncodingScalar, CKKSLinearTransformationOps, CKKSMulOps,
    },
    layouts::{CKKSCiphertextOwned, CKKSModuleAlloc, ScratchArenaTakeCKKS},
};
use poulpy_core::{
    GLWEBytesOf,
    layouts::{
        GGLWEInfos, GLWEAutomorphismKeyHelper, GLWEInfos, GLWEToBackendMut, GLWEToBackendRef,
        prepared::{GGLWEPreparedToBackendRef, GLWETensorKeyPreparedToBackendRef},
    },
};
use poulpy_hal::{
    api::{CnvPVecAlloc, ModuleN},
    layouts::{Backend, Module, ScratchArena},
};

use super::{
    CKKSPackedAffineOps, CKKSSplitAffineOps, PackedAffinePlan, PackedAffineWorkspace,
    SplitAffinePlan, TransformStrategy,
};
use crate::{
    algebra::{AffineMap, Coefficient, TensorMap},
    layout::PackedLayout,
    scalar::BlockScalar,
};

pub struct PackedMultivariatePlan<BE: Backend> {
    layout: PackedLayout,
    input_widths: Vec<usize>,
    alignments: Vec<PackedAffinePlan<BE>>,
    output: PackedAffinePlan<BE>,
    galois_elements: Vec<i64>,
}

impl<BE: Backend> PackedMultivariatePlan<BE> {
    #[allow(clippy::too_many_arguments)]
    pub fn compile<F>(
        module: &Module<BE>,
        layout: PackedLayout,
        tensor: &TensorMap<F>,
        base2k: poulpy_core::layouts::Base2K,
        alignment_meta: CoeffsMeta,
        output_meta: CoeffsMeta,
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
        let feature_width = tensor.feature_width();
        ensure!(
            feature_width <= layout.block_width(),
            "tensor basis width {feature_width} exceeds packed block width {}",
            layout.block_width()
        );
        let input_widths = tensor.input_widths().collect::<Vec<_>>();
        let mut alignments = Vec::with_capacity(input_widths.len());
        for variable in 0..input_widths.len() {
            let alignment = alignment_map(tensor, variable)?;
            alignments.push(PackedAffinePlan::compile(
                module,
                layout,
                &alignment,
                base2k,
                alignment_meta,
                strategy,
                scratch,
            )?);
        }
        let output = PackedAffinePlan::compile(
            module,
            layout,
            &tensor.as_affine(),
            base2k,
            output_meta,
            strategy,
            scratch,
        )?;
        let galois_elements = alignments
            .iter()
            .flat_map(PackedAffinePlan::galois_elements)
            .chain(output.galois_elements())
            .copied()
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect();
        Ok(Self {
            layout,
            input_widths,
            alignments,
            output,
            galois_elements,
        })
    }

    pub fn layout(&self) -> PackedLayout {
        self.layout
    }

    pub fn input_widths(&self) -> &[usize] {
        &self.input_widths
    }

    pub fn output_width(&self) -> usize {
        self.output.output_width()
    }

    pub fn galois_elements(&self) -> &[i64] {
        &self.galois_elements
    }

    pub fn alloc_workspace<C>(
        &self,
        module: &Module<BE>,
        input: &C,
    ) -> PackedMultivariateWorkspace<BE>
    where
        Module<BE>: CKKSModuleAlloc<BE> + CnvPVecAlloc<BE>,
        C: GLWEInfos + poulpy_ckks::CKKSInfos,
    {
        let alignment = self
            .alignments
            .iter()
            .map(|plan| plan.alloc_workspace(module, input))
            .collect();
        let mut width = self.alignments.len();
        let mut layers = Vec::new();
        while width != 0 {
            layers.push(
                (0..width)
                    .map(|_| module.ckks_ciphertext_alloc_from_infos(input))
                    .collect(),
            );
            if width == 1 {
                break;
            }
            width = width.div_ceil(2);
        }
        let output = self.output.alloc_workspace(module, input);
        PackedMultivariateWorkspace {
            alignment,
            layers,
            output,
        }
    }
}

pub struct PackedMultivariateWorkspace<BE: Backend> {
    alignment: Vec<PackedAffineWorkspace<BE>>,
    layers: Vec<Vec<CKKSCiphertextOwned<BE>>>,
    output: PackedAffineWorkspace<BE>,
}

#[derive(Clone, Copy)]
enum FeatureRecipe {
    Input { variable: usize, coordinate: usize },
    Product { lhs: usize, rhs: usize },
}

pub struct SplitMultivariatePlan<BE: Backend> {
    input_widths: Vec<usize>,
    recipes: Vec<FeatureRecipe>,
    output: SplitAffinePlan<BE>,
}

impl<BE: Backend> SplitMultivariatePlan<BE> {
    pub fn compile<F>(
        module: &Module<BE>,
        tensor: &TensorMap<F>,
        base2k: poulpy_core::layouts::Base2K,
        coeffs_meta: CoeffsMeta,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<Self>
    where
        F: BlockScalar + CKKSEncodingScalar,
        Module<BE>: CKKSModuleAlloc<BE> + CKKSEncodingOps<BE, F> + ModuleN,
    {
        Ok(Self {
            input_widths: tensor.input_widths().collect(),
            recipes: feature_recipes(tensor.input_sizes())?,
            output: SplitAffinePlan::compile(
                module,
                &tensor.as_affine(),
                base2k,
                coeffs_meta,
                scratch,
            )?,
        })
    }

    pub fn input_widths(&self) -> &[usize] {
        &self.input_widths
    }

    pub fn feature_width(&self) -> usize {
        self.recipes.len()
    }

    pub fn output_width(&self) -> usize {
        self.output.output_width()
    }
}

pub trait CKKSMultivariateOps<BE: Backend> {
    fn ckks_packed_multivariate_tmp_bytes<C, K, T>(
        &self,
        plan: &PackedMultivariatePlan<BE>,
        input: &C,
        automorphism_key: &K,
        tensor_key: &T,
    ) -> usize
    where
        C: CKKSCtBounds,
        K: GGLWEInfos,
        T: GGLWEInfos;

    #[allow(clippy::too_many_arguments)]
    fn ckks_packed_multivariate_into<Dst, Src, H, K, T>(
        &self,
        output: &mut Dst,
        inputs: &[Src],
        plan: &PackedMultivariatePlan<BE>,
        workspace: &mut PackedMultivariateWorkspace<BE>,
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

    fn ckks_split_multivariate_tmp_bytes<R, A, T>(
        &self,
        plan: &SplitMultivariatePlan<BE>,
        output: &R,
        input: &A,
        tensor_key: &T,
    ) -> usize
    where
        R: CKKSCtBounds,
        A: CKKSCtBounds,
        T: GGLWEInfos;

    fn ckks_split_multivariate_into<Dst, Src, T>(
        &self,
        outputs: &mut [Dst],
        inputs: &[&[Src]],
        plan: &SplitMultivariatePlan<BE>,
        tensor_key: &T,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
        Src: GLWEToBackendRef<BE> + CKKSCtBounds,
        T: GGLWEInfos + GLWETensorKeyPreparedToBackendRef<BE> + GGLWEPreparedToBackendRef<BE>;
}

impl<BE: Backend> CKKSMultivariateOps<BE> for Module<BE>
where
    Module<BE>: CKKSPackedAffineOps<BE>
        + CKKSSplitAffineOps<BE>
        + CKKSCopyOps<BE>
        + CKKSMulOps<BE>
        + GLWEBytesOf<BE>,
{
    fn ckks_packed_multivariate_tmp_bytes<C, K, T>(
        &self,
        plan: &PackedMultivariatePlan<BE>,
        input: &C,
        automorphism_key: &K,
        tensor_key: &T,
    ) -> usize
    where
        C: CKKSCtBounds,
        K: GGLWEInfos,
        T: GGLWEInfos,
    {
        plan.alignments
            .iter()
            .map(|alignment| self.ckks_packed_affine_tmp_bytes(alignment, input, automorphism_key))
            .chain(std::iter::once(self.ckks_packed_affine_tmp_bytes(
                &plan.output,
                input,
                automorphism_key,
            )))
            .max()
            .unwrap_or(0)
            .max(self.ckks_mul_tmp_bytes(input, input, input, tensor_key))
            .max(self.ckks_copy_tmp_bytes())
    }

    fn ckks_packed_multivariate_into<Dst, Src, H, K, T>(
        &self,
        output: &mut Dst,
        inputs: &[Src],
        plan: &PackedMultivariatePlan<BE>,
        workspace: &mut PackedMultivariateWorkspace<BE>,
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
        ensure!(
            inputs.len() == plan.alignments.len(),
            "packed multivariate input count does not match plan"
        );
        ensure!(
            workspace.alignment.len() == plan.alignments.len()
                && workspace.layers.first().map(Vec::len) == Some(inputs.len()),
            "packed multivariate workspace does not match plan"
        );
        for (((aligned, input), alignment), alignment_workspace) in workspace.layers[0]
            .iter_mut()
            .zip(inputs)
            .zip(&plan.alignments)
            .zip(&mut workspace.alignment)
        {
            self.ckks_packed_affine_into(
                aligned,
                input,
                alignment,
                alignment_workspace,
                automorphism_keys,
                scratch,
            )?;
        }
        for level in 0..workspace.layers.len() - 1 {
            let (previous, following) = workspace.layers.split_at_mut(level + 1);
            let previous = &previous[level];
            let next = &mut following[0];
            for (index, pair) in previous.chunks_exact(2).enumerate() {
                self.ckks_mul_into(&mut next[index], &pair[0], &pair[1], tensor_key, scratch)?;
            }
            if previous.len() % 2 == 1 {
                self.ckks_copy(
                    next.last_mut().expect("next layer is non-empty"),
                    previous.last().unwrap(),
                    scratch,
                )?;
            }
        }
        let tensor = &workspace.layers.last().expect("plan has inputs")[0];
        self.ckks_packed_affine_into(
            output,
            tensor,
            &plan.output,
            &mut workspace.output,
            automorphism_keys,
            scratch,
        )
    }

    fn ckks_split_multivariate_tmp_bytes<R, A, T>(
        &self,
        plan: &SplitMultivariatePlan<BE>,
        output: &R,
        input: &A,
        tensor_key: &T,
    ) -> usize
    where
        R: CKKSCtBounds,
        A: CKKSCtBounds,
        T: GGLWEInfos,
    {
        plan.feature_width() * self.glwe_bytes_of_from_infos(input)
            + self
                .ckks_copy_tmp_bytes()
                .max(self.ckks_mul_tmp_bytes(input, input, input, tensor_key))
                .max(self.ckks_split_affine_tmp_bytes(&plan.output, output, input))
    }

    fn ckks_split_multivariate_into<Dst, Src, T>(
        &self,
        outputs: &mut [Dst],
        inputs: &[&[Src]],
        plan: &SplitMultivariatePlan<BE>,
        tensor_key: &T,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
        Src: GLWEToBackendRef<BE> + CKKSCtBounds,
        T: GGLWEInfos + GLWETensorKeyPreparedToBackendRef<BE> + GGLWEPreparedToBackendRef<BE>,
    {
        ensure!(
            inputs.len() == plan.input_widths.len(),
            "split multivariate input count does not match plan"
        );
        ensure!(
            inputs
                .iter()
                .zip(&plan.input_widths)
                .all(|(input, &width)| input.len() == width),
            "split multivariate input width does not match plan"
        );
        ensure!(
            !inputs.is_empty() && !inputs[0].is_empty(),
            "empty multivariate input"
        );
        scratch.scope(|local| {
            let (mut features, mut local) = local.take_ckks_ciphertext_slice_scratch(
                plan.feature_width(),
                &inputs[0][0],
                inputs[0][0].meta(),
            );
            for (index, &recipe) in plan.recipes.iter().enumerate() {
                let (previous, current) = features.split_at_mut(index);
                match recipe {
                    FeatureRecipe::Input {
                        variable,
                        coordinate,
                    } => {
                        self.ckks_copy(&mut current[0], &inputs[variable][coordinate], &mut local)?
                    }
                    FeatureRecipe::Product { lhs, rhs } => self.ckks_mul_into(
                        &mut current[0],
                        &previous[lhs],
                        &previous[rhs],
                        tensor_key,
                        &mut local,
                    )?,
                }
            }
            self.ckks_split_affine_into(outputs, &features, &plan.output, &mut local)
        })
    }
}

fn alignment_map<F: BlockScalar>(tensor: &TensorMap<F>, variable: usize) -> Result<AffineMap<F>> {
    let feature_width = tensor.feature_width();
    let input_width = tensor.input_sizes()[variable] - 1;
    let stride = tensor.input_sizes()[variable + 1..]
        .iter()
        .product::<usize>();
    let mut matrix = vec![Coefficient::zero(); feature_width * input_width];
    let mut bias = vec![Coefficient::zero(); feature_width];
    for flat in 1..=feature_width {
        let digit = (flat / stride) % tensor.input_sizes()[variable];
        if digit == 0 {
            bias[flat - 1] = Coefficient::one();
        } else {
            matrix[(flat - 1) * input_width + digit - 1] = Coefficient::one();
        }
    }
    AffineMap::new(feature_width, input_width, matrix, bias)
}

fn feature_recipes(input_sizes: &[usize]) -> Result<Vec<FeatureRecipe>> {
    let total = input_sizes.iter().try_fold(1usize, |product, &size| {
        product
            .checked_mul(size)
            .ok_or_else(|| anyhow::anyhow!("tensor basis is too large"))
    })?;
    let mut strides = vec![1usize; input_sizes.len()];
    for index in (0..input_sizes.len() - 1).rev() {
        strides[index] = strides[index + 1] * input_sizes[index + 1];
    }
    let mut recipes = Vec::with_capacity(total - 1);
    for flat in 1..total {
        let terms = input_sizes
            .iter()
            .zip(&strides)
            .enumerate()
            .filter_map(|(variable, (&size, &stride))| {
                let digit = (flat / stride) % size;
                (digit != 0).then(|| (variable, digit - 1, digit * stride))
            })
            .collect::<Vec<_>>();
        if let [(variable, coordinate, _)] = terms.as_slice() {
            recipes.push(FeatureRecipe::Input {
                variable: *variable,
                coordinate: *coordinate,
            });
        } else {
            let middle = terms.len() / 2;
            let lhs = terms[..middle].iter().map(|term| term.2).sum::<usize>() - 1;
            let rhs = terms[middle..].iter().map(|term| term.2).sum::<usize>() - 1;
            ensure!(
                lhs < flat - 1 && rhs < flat - 1,
                "invalid tensor feature schedule"
            );
            recipes.push(FeatureRecipe::Product { lhs, rhs });
        }
    }
    Ok(recipes)
}
