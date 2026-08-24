use anyhow::{Result, ensure};
use poulpy_ckks::{
    CKKSAtkBounds, CKKSCtBounds, CoeffsMeta, SetCKKSInfos,
    api::{
        CKKSAddOps, CKKSCopyOps, CKKSEncodingHostOps, CKKSEncodingOps, CKKSEncodingScalar,
        CKKSLinearTransformationOps, CKKSSubOps, LinearTransformationBabySteps,
        LinearTransformationPrepared,
    },
    default::ckks_encode_linear_transformation_from_diagonals,
    layouts::{CKKSModuleAlloc, CKKSPlaintextOwned, ComplexDiagonals},
};
use poulpy_core::{
    GLWEBytesOf,
    layouts::{
        Diagonals, GGLWEInfos, GLWEAutomorphismKeyHelper, GLWEInfos, GLWEToBackendMut,
        GLWEToBackendRef,
    },
};
use poulpy_hal::{
    api::{CnvPVecAlloc, ModuleN},
    layouts::{Backend, Module, ScratchArena},
};

use super::TransformStrategy;
use crate::{algebra::AffineMap, layout::PackedLayout, scalar::BlockScalar};

enum PackedLinear<BE: Backend> {
    Zero,
    Identity,
    Transform(LinearTransformationPrepared<BE>),
}

pub struct PackedAffinePlan<BE: Backend> {
    layout: PackedLayout,
    input_width: usize,
    output_width: usize,
    linear: PackedLinear<BE>,
    bias: Option<CKKSPlaintextOwned<BE>>,
    galois_elements: Vec<i64>,
}

impl<BE: Backend> PackedAffinePlan<BE> {
    #[allow(clippy::too_many_arguments)]
    pub fn compile<F>(
        module: &Module<BE>,
        layout: PackedLayout,
        map: &AffineMap<F>,
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
        validate_map(map)?;
        ensure!(
            layout.slots() == module.n() / 2,
            "packed layout has {} slots, backend module has {}",
            layout.slots(),
            module.n() / 2
        );
        ensure!(
            map.cols <= layout.block_width() && map.rows <= layout.block_width(),
            "affine map shape {}x{} exceeds packed block width {}",
            map.rows,
            map.cols,
            layout.block_width()
        );

        let matrix_meta = effective_meta(&map.matrix, coeffs_meta)?;
        let bias_meta = effective_meta(&map.bias, coeffs_meta)?;
        let all_zero = map.matrix.iter().all(is_zero);
        let identity = is_identity(map);
        let (linear, galois_elements) = if all_zero {
            (PackedLinear::Zero, Vec::new())
        } else if identity {
            (PackedLinear::Identity, Vec::new())
        } else {
            let diagonals = build_diagonals(layout, map);
            let strategy = strategy.resolve(&diagonals.indexes(), layout.slots())?;
            let encoded = ckks_encode_linear_transformation_from_diagonals(
                module,
                base2k,
                matrix_meta,
                &diagonals,
                strategy,
                false,
                scratch,
            )?;
            let galois_elements = encoded.galois_elements(module.n() as i64 * 2);
            let first = encoded
                .first_diagonal_plaintext()
                .expect("non-zero diagonal map produced an empty transform");
            let mut prepared = LinearTransformationPrepared::alloc_prepared_from_index(
                module,
                &encoded.index(),
                first,
            );
            module.ckks_prepare_linear_transformation_rhs(&mut prepared, &encoded, scratch);
            (PackedLinear::Transform(prepared), galois_elements)
        };

        let bias = encode_bias(module, layout, map, base2k, bias_meta, scratch)?;
        Ok(Self {
            layout,
            input_width: map.cols,
            output_width: map.rows,
            linear,
            bias,
            galois_elements,
        })
    }

    pub fn layout(&self) -> PackedLayout {
        self.layout
    }

    pub fn input_width(&self) -> usize {
        self.input_width
    }

    pub fn output_width(&self) -> usize {
        self.output_width
    }

    pub fn galois_elements(&self) -> &[i64] {
        &self.galois_elements
    }

    pub fn alloc_workspace<C>(&self, module: &Module<BE>, input: &C) -> PackedAffineWorkspace<BE>
    where
        Module<BE>: CnvPVecAlloc<BE>,
        C: GLWEInfos,
    {
        let babies = match &self.linear {
            PackedLinear::Transform(transform) => Some(LinearTransformationBabySteps::alloc(
                module,
                transform.baby_steps(),
                input,
            )),
            PackedLinear::Zero | PackedLinear::Identity => None,
        };
        PackedAffineWorkspace { babies }
    }
}

pub struct PackedAffineWorkspace<BE: Backend> {
    babies: Option<LinearTransformationBabySteps<BE>>,
}

pub trait CKKSPackedAffineOps<BE: Backend> {
    fn ckks_packed_affine_tmp_bytes<C, K>(
        &self,
        plan: &PackedAffinePlan<BE>,
        input: &C,
        key: &K,
    ) -> usize
    where
        C: CKKSCtBounds,
        K: GGLWEInfos;

    fn ckks_packed_affine_into<Dst, Src, H, K>(
        &self,
        dst: &mut Dst,
        input: &Src,
        plan: &PackedAffinePlan<BE>,
        workspace: &mut PackedAffineWorkspace<BE>,
        keys: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
        Src: GLWEToBackendRef<BE> + CKKSCtBounds,
        K: CKKSAtkBounds<BE>,
        H: GLWEAutomorphismKeyHelper<K, BE>;
}

impl<BE: Backend> CKKSPackedAffineOps<BE> for Module<BE>
where
    Module<BE>: CKKSLinearTransformationOps<BE>
        + poulpy_ckks::api::CKKSCopyOps<BE>
        + poulpy_ckks::api::CKKSSubOps<BE>
        + poulpy_ckks::api::CKKSAddOps<BE>
        + CnvPVecAlloc<BE>
        + GLWEBytesOf<BE>,
{
    fn ckks_packed_affine_tmp_bytes<C, K>(
        &self,
        plan: &PackedAffinePlan<BE>,
        input: &C,
        key: &K,
    ) -> usize
    where
        C: CKKSCtBounds,
        K: GGLWEInfos,
    {
        let linear = match &plan.linear {
            PackedLinear::Transform(_) => self
                .ckks_prepare_linear_transformation_baby_steps_tmp_bytes(input, key)
                .max(self.ckks_eval_linear_transformation_tmp_bytes(input, key)),
            PackedLinear::Identity => self.ckks_copy_tmp_bytes(),
            PackedLinear::Zero => self.ckks_sub_tmp_bytes(),
        };
        if plan.bias.is_some() {
            linear.max(self.ckks_add_pt_vec_tmp_bytes())
        } else {
            linear
        }
    }

    fn ckks_packed_affine_into<Dst, Src, H, K>(
        &self,
        dst: &mut Dst,
        input: &Src,
        plan: &PackedAffinePlan<BE>,
        workspace: &mut PackedAffineWorkspace<BE>,
        keys: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
        Src: GLWEToBackendRef<BE> + CKKSCtBounds,
        K: CKKSAtkBounds<BE>,
        H: GLWEAutomorphismKeyHelper<K, BE>,
    {
        ensure!(
            input.n().as_usize() / 2 == plan.layout.slots(),
            "packed ciphertext and plan use different slot counts"
        );
        match &plan.linear {
            PackedLinear::Zero => self.ckks_sub_into(dst, input, input, scratch)?,
            PackedLinear::Identity => self.ckks_copy(dst, input, scratch)?,
            PackedLinear::Transform(transform) => {
                let babies = workspace.babies.as_mut().ok_or_else(|| {
                    anyhow::anyhow!("packed affine workspace has no baby-step cache")
                })?;
                if babies.size() != input.size() || babies.cols() != input.rank().as_usize() + 1 {
                    *babies =
                        LinearTransformationBabySteps::alloc(self, transform.baby_steps(), input);
                }
                self.ckks_prepare_linear_transformation_baby_steps(babies, input, keys, scratch)?;
                self.ckks_eval_linear_transformation_into(
                    dst, input, babies, transform, keys, scratch,
                )?;
            }
        }
        if let Some(bias) = &plan.bias {
            self.ckks_add_pt_vec_assign(dst, bias, scratch)?;
        }
        Ok(())
    }
}

fn validate_map<F: BlockScalar>(map: &AffineMap<F>) -> Result<()> {
    ensure!(
        map.rows != 0 && map.cols != 0,
        "affine map dimensions must be non-zero"
    );
    let matrix_len = map
        .rows
        .checked_mul(map.cols)
        .ok_or_else(|| anyhow::anyhow!("affine map dimensions overflow usize"))?;
    ensure!(
        map.matrix.len() == matrix_len,
        "affine matrix storage has the wrong length"
    );
    ensure!(
        map.bias.len() == map.rows,
        "affine bias storage has the wrong length"
    );
    Ok(())
}

fn is_zero<F: BlockScalar>(coefficient: &crate::algebra::Coefficient<F>) -> bool {
    let value = (*coefficient).value();
    value.re == F::zero() && value.im == F::zero()
}

fn is_one<F: BlockScalar>(coefficient: &crate::algebra::Coefficient<F>) -> bool {
    coefficient.exact_parts() == Some((1, 0, 0))
}

fn is_identity<F: BlockScalar>(map: &AffineMap<F>) -> bool {
    map.rows == map.cols
        && (0..map.rows).all(|row| {
            (0..map.cols).all(|col| {
                let coefficient = &map.matrix[row * map.cols + col];
                if row == col {
                    is_one(coefficient)
                } else {
                    is_zero(coefficient)
                }
            })
        })
}

fn effective_meta<F: BlockScalar>(
    coefficients: &[crate::algebra::Coefficient<F>],
    requested: CoeffsMeta,
) -> Result<CoeffsMeta> {
    let mut exact_log_delta = 0usize;
    for coefficient in coefficients {
        if let Some((_, _, log_den)) = coefficient.exact_parts() {
            exact_log_delta = exact_log_delta.max(log_den as usize);
        }
        let value = coefficient.value();
        ensure!(
            value.re.is_finite() && value.im.is_finite(),
            "affine coefficient is not finite"
        );
    }
    ensure!(
        requested.log_delta() >= exact_log_delta,
        "coefficient precision {} cannot represent an exact dyadic denominator 2^{exact_log_delta}",
        requested.log_delta()
    );
    let mut meta = requested.meta;
    meta.log_sparsity = 0;
    meta.slots = poulpy_ckks::SlotsKind::Complex;
    Ok(CoeffsMeta {
        k: (requested.log_delta() + requested.log_budget()).into(),
        meta,
    })
}

fn build_diagonals<F: BlockScalar>(
    layout: PackedLayout,
    map: &AffineMap<F>,
) -> ComplexDiagonals<F> {
    let mut real = Diagonals::new(layout.slots());
    let mut imag = Diagonals::new(layout.slots());
    for diagonal in -(map.rows as i64 - 1)..map.cols as i64 {
        let mut re = vec![F::zero(); layout.slots()];
        let mut im = vec![F::zero(); layout.slots()];
        let mut has_re = false;
        let mut has_im = false;
        for block in 0..layout.block_count() {
            for row in 0..map.rows {
                let col = row as i64 + diagonal;
                if !(0..map.cols as i64).contains(&col) {
                    continue;
                }
                let value = map.matrix[row * map.cols + col as usize].value();
                let slot = block * layout.block_width() + row;
                re[slot] = value.re;
                im[slot] = value.im;
                has_re |= value.re != F::zero();
                has_im |= value.im != F::zero();
            }
        }
        if has_re {
            real.set(diagonal, re);
        }
        if has_im {
            imag.set(diagonal, im);
        }
    }
    ComplexDiagonals::new(real, imag)
}

fn encode_bias<BE, F>(
    module: &Module<BE>,
    layout: PackedLayout,
    map: &AffineMap<F>,
    base2k: poulpy_core::layouts::Base2K,
    coeffs_meta: CoeffsMeta,
    scratch: &mut ScratchArena<'_, BE>,
) -> Result<Option<CKKSPlaintextOwned<BE>>>
where
    BE: Backend,
    F: BlockScalar + CKKSEncodingScalar,
    Module<BE>: CKKSModuleAlloc<BE> + CKKSEncodingOps<BE, F> + ModuleN,
{
    if map.bias.iter().all(is_zero) {
        return Ok(None);
    }
    let mut re = vec![F::zero(); layout.slots()];
    let mut im = vec![F::zero(); layout.slots()];
    for block in 0..layout.block_count() {
        for (row, coefficient) in map.bias.iter().enumerate() {
            let value = coefficient.value();
            let slot = block * layout.block_width() + row;
            re[slot] = value.re;
            im[slot] = value.im;
        }
    }
    let mut bias = module.ckks_pt_vec_alloc(base2k, coeffs_meta.k);
    bias.set_meta_checked(coeffs_meta.meta)?;
    module.ckks_encode_reim_into(&mut bias, &re, &im, scratch)?;
    Ok(Some(bias))
}
