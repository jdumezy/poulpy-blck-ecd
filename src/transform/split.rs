use anyhow::{Result, ensure};
use poulpy_ckks::{
    CKKSCtBounds, CoeffsMeta, SetCKKSInfos, SlotsKind,
    api::{
        CKKSAddOps, CKKSCopyOps, CKKSEncodingHostOps, CKKSEncodingOps, CKKSEncodingScalar,
        CKKSImagOps, CKKSMulAddOps, CKKSMulOps, CKKSNegOps, CKKSPow2Ops, CKKSSubOps,
    },
    layouts::{CKKSModuleAlloc, CKKSPlaintextOwned, ScratchArenaTakeCKKS},
};
use poulpy_core::{
    GLWEBytesOf,
    layouts::{GLWEToBackendMut, GLWEToBackendRef},
};
use poulpy_hal::{
    api::ModuleN,
    layouts::{Backend, Module, ScratchArena},
};

use crate::{
    algebra::{AffineMap, Coefficient},
    scalar::BlockScalar,
};

#[derive(Clone, Copy)]
struct ScalarIndex(usize);

#[derive(Clone, Copy)]
enum SplitTerm {
    Exact {
        input: usize,
        quarter_turns: u8,
        shift: i64,
    },
    Scalar {
        input: usize,
        re: Option<ScalarIndex>,
        im: Option<ScalarIndex>,
    },
}

#[derive(Clone, Copy, Default)]
struct SplitBias {
    re: Option<ScalarIndex>,
    im: Option<ScalarIndex>,
}

struct SplitRow {
    terms: Vec<SplitTerm>,
    bias: SplitBias,
}

pub struct SplitAffinePlan<BE: Backend> {
    input_width: usize,
    rows: Vec<SplitRow>,
    scalar_banks: Vec<CKKSPlaintextOwned<BE>>,
    bank_width: usize,
}

impl<BE: Backend> SplitAffinePlan<BE> {
    pub fn compile<F>(
        module: &Module<BE>,
        map: &AffineMap<F>,
        base2k: poulpy_core::layouts::Base2K,
        coeffs_meta: CoeffsMeta,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<Self>
    where
        F: BlockScalar + CKKSEncodingScalar,
        Module<BE>: CKKSModuleAlloc<BE> + CKKSEncodingOps<BE, F> + ModuleN,
    {
        validate_map(map)?;
        let scalar_meta = effective_scalar_meta(map, coeffs_meta)?;
        let mut scalars = Vec::new();
        let mut rows = Vec::with_capacity(map.rows);
        for row in 0..map.rows {
            let mut terms = Vec::new();
            for input in 0..map.cols {
                let coefficient = map.matrix[row * map.cols + input];
                if coefficient_is_zero(coefficient) {
                    continue;
                }
                if let Some(scales) = exact_scales(coefficient) {
                    terms.extend(scales.into_iter().map(|(quarter_turns, shift)| {
                        SplitTerm::Exact {
                            input,
                            quarter_turns,
                            shift,
                        }
                    }));
                } else {
                    validate_precision(coefficient, scalar_meta)?;
                    let (re, im) = scalar_components(coefficient, &mut scalars)?;
                    terms.push(SplitTerm::Scalar { input, re, im });
                }
            }
            validate_precision(map.bias[row], scalar_meta)?;
            let (re, im) = scalar_components(map.bias[row], &mut scalars)?;
            rows.push(SplitRow {
                terms,
                bias: SplitBias { re, im },
            });
        }

        let bank_width = module.n();
        let mut scalar_banks = Vec::with_capacity(scalars.len().div_ceil(bank_width));
        let mut meta = scalar_meta.meta;
        meta.slots = SlotsKind::Real;
        for values in scalars.chunks(bank_width) {
            let mut plaintext = module.ckks_pt_coeffs_alloc(values.len(), base2k, scalar_meta.k);
            plaintext.set_meta_checked(meta)?;
            module.ckks_encode_coeffs_host_into(&mut plaintext, values, scratch)?;
            scalar_banks.push(plaintext);
        }
        Ok(Self {
            input_width: map.cols,
            rows,
            scalar_banks,
            bank_width,
        })
    }

    pub fn input_width(&self) -> usize {
        self.input_width
    }

    pub fn output_width(&self) -> usize {
        self.rows.len()
    }

    fn scalar(&self, index: ScalarIndex) -> (&CKKSPlaintextOwned<BE>, usize) {
        (
            &self.scalar_banks[index.0 / self.bank_width],
            index.0 % self.bank_width,
        )
    }
}

pub trait CKKSSplitAffineOps<BE: Backend> {
    fn ckks_split_affine_tmp_bytes<R, A>(
        &self,
        plan: &SplitAffinePlan<BE>,
        output: &R,
        input: &A,
    ) -> usize
    where
        R: CKKSCtBounds,
        A: CKKSCtBounds;

    fn ckks_split_affine_into<Dst, Src>(
        &self,
        outputs: &mut [Dst],
        inputs: &[Src],
        plan: &SplitAffinePlan<BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
        Src: GLWEToBackendRef<BE> + CKKSCtBounds;
}

impl<BE: Backend> CKKSSplitAffineOps<BE> for Module<BE>
where
    Module<BE>: CKKSAddOps<BE>
        + CKKSCopyOps<BE>
        + CKKSImagOps<BE>
        + CKKSMulAddOps<BE>
        + CKKSMulOps<BE>
        + CKKSNegOps<BE>
        + CKKSPow2Ops<BE>
        + CKKSSubOps<BE>
        + GLWEBytesOf<BE>,
{
    fn ckks_split_affine_tmp_bytes<R, A>(
        &self,
        plan: &SplitAffinePlan<BE>,
        output: &R,
        input: &A,
    ) -> usize
    where
        R: CKKSCtBounds,
        A: CKKSCtBounds,
    {
        let mut operations = self
            .ckks_copy_tmp_bytes()
            .max(self.ckks_add_tmp_bytes())
            .max(self.ckks_add_pt_const_tmp_bytes())
            .max(self.ckks_mul_i_tmp_bytes())
            .max(self.ckks_div_i_tmp_bytes())
            .max(self.ckks_mul_pow2_tmp_bytes())
            .max(self.ckks_div_pow2_tmp_bytes())
            .max(self.ckks_sub_tmp_bytes());
        if let Some(bank) = plan.scalar_banks.first() {
            operations = operations
                .max(self.ckks_mul_pt_const_tmp_bytes(output, input, bank))
                .max(self.ckks_mul_add_pt_const_tmp_bytes(output, input, bank));
        }
        2 * self.glwe_bytes_of_from_infos(output) + operations
    }

    fn ckks_split_affine_into<Dst, Src>(
        &self,
        outputs: &mut [Dst],
        inputs: &[Src],
        plan: &SplitAffinePlan<BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
        Src: GLWEToBackendRef<BE> + CKKSCtBounds,
    {
        ensure!(
            inputs.len() == plan.input_width,
            "split input width does not match plan"
        );
        ensure!(
            outputs.len() == plan.rows.len(),
            "split output width does not match plan"
        );
        for (output, row) in outputs.iter_mut().zip(&plan.rows) {
            scratch.scope(|local| {
                let (mut temporaries, mut local) =
                    local.take_ckks_ciphertext_slice_scratch(2, output, output.meta());
                let (term, auxiliary) = temporaries.split_at_mut(1);
                let term = &mut term[0];
                let auxiliary = &mut auxiliary[0];

                if let Some((&first, rest)) = row.terms.split_first() {
                    apply_term(self, output, auxiliary, inputs, plan, first, &mut local)?;
                    for &next in rest {
                        accumulate_term(
                            self, output, term, auxiliary, inputs, plan, next, &mut local,
                        )?;
                    }
                } else {
                    self.ckks_sub_into(output, &inputs[0], &inputs[0], &mut local)?;
                }
                add_bias(self, output, plan, row.bias, &mut local)
            })?;
        }
        Ok(())
    }
}

#[allow(clippy::too_many_arguments)]
fn accumulate_term<BE, M, Dst, Tmp, Aux, Src>(
    module: &M,
    output: &mut Dst,
    term: &mut Tmp,
    auxiliary: &mut Aux,
    inputs: &[Src],
    plan: &SplitAffinePlan<BE>,
    next: SplitTerm,
    scratch: &mut ScratchArena<'_, BE>,
) -> Result<()>
where
    BE: Backend,
    M: CKKSAddOps<BE>
        + CKKSCopyOps<BE>
        + CKKSImagOps<BE>
        + CKKSMulAddOps<BE>
        + CKKSMulOps<BE>
        + CKKSNegOps<BE>
        + CKKSPow2Ops<BE>,
    Dst: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
    Tmp: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
    Aux: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
    Src: GLWEToBackendRef<BE> + CKKSCtBounds,
{
    match next {
        SplitTerm::Scalar {
            input,
            re: Some(re),
            im: None,
        } => {
            let (bank, coefficient) = plan.scalar(re);
            module.ckks_mul_add_pt_const_into(
                output,
                &inputs[input],
                bank,
                coefficient,
                scratch,
            )?;
        }
        SplitTerm::Scalar {
            input,
            re: Some(re),
            im: Some(im),
        } => {
            let (bank, coefficient) = plan.scalar(re);
            module.ckks_mul_add_pt_const_into(
                output,
                &inputs[input],
                bank,
                coefficient,
                scratch,
            )?;
            let (bank, coefficient) = plan.scalar(im);
            module.ckks_mul_pt_const_into(auxiliary, &inputs[input], bank, coefficient, scratch)?;
            module.ckks_mul_i_assign(auxiliary, scratch)?;
            module.ckks_add_assign(output, auxiliary, scratch)?;
        }
        _ => {
            apply_term(module, term, auxiliary, inputs, plan, next, scratch)?;
            module.ckks_add_assign(output, term, scratch)?;
        }
    }
    Ok(())
}

fn apply_term<BE, M, Dst, Aux, Src>(
    module: &M,
    dst: &mut Dst,
    auxiliary: &mut Aux,
    inputs: &[Src],
    plan: &SplitAffinePlan<BE>,
    term: SplitTerm,
    scratch: &mut ScratchArena<'_, BE>,
) -> Result<()>
where
    BE: Backend,
    M: CKKSAddOps<BE>
        + CKKSCopyOps<BE>
        + CKKSImagOps<BE>
        + CKKSMulOps<BE>
        + CKKSNegOps<BE>
        + CKKSPow2Ops<BE>,
    Dst: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
    Aux: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
    Src: GLWEToBackendRef<BE> + CKKSCtBounds,
{
    match term {
        SplitTerm::Exact {
            input,
            quarter_turns,
            shift,
        } => {
            module.ckks_copy(dst, &inputs[input], scratch)?;
            if shift > 0 {
                module.ckks_mul_pow2_assign(dst, shift as usize, scratch)?;
            } else if shift < 0 {
                module.ckks_div_pow2_assign(dst, shift.unsigned_abs() as usize)?;
            }
            match quarter_turns {
                0 => {}
                1 => module.ckks_mul_i_assign(dst, scratch)?,
                2 => module.ckks_neg_assign(dst)?,
                3 => module.ckks_div_i_assign(dst, scratch)?,
                _ => unreachable!("quarter turns are reduced modulo four"),
            }
        }
        SplitTerm::Scalar { input, re, im } => match (re, im) {
            (Some(re), None) => {
                let (bank, coefficient) = plan.scalar(re);
                module.ckks_mul_pt_const_into(dst, &inputs[input], bank, coefficient, scratch)?;
            }
            (None, Some(im)) => {
                let (bank, coefficient) = plan.scalar(im);
                module.ckks_mul_pt_const_into(dst, &inputs[input], bank, coefficient, scratch)?;
                module.ckks_mul_i_assign(dst, scratch)?;
            }
            (Some(re), Some(im)) => {
                let (real_bank, real_coefficient) = plan.scalar(re);
                module.ckks_mul_pt_const_into(
                    dst,
                    &inputs[input],
                    real_bank,
                    real_coefficient,
                    scratch,
                )?;
                let (imag_bank, imag_coefficient) = plan.scalar(im);
                module.ckks_mul_pt_const_into(
                    auxiliary,
                    &inputs[input],
                    imag_bank,
                    imag_coefficient,
                    scratch,
                )?;
                module.ckks_mul_i_assign(auxiliary, scratch)?;
                module.ckks_add_assign(dst, auxiliary, scratch)?;
            }
            (None, None) => unreachable!("zero coefficients are omitted from split plans"),
        },
    }
    Ok(())
}

fn add_bias<BE, M, Dst>(
    module: &M,
    dst: &mut Dst,
    plan: &SplitAffinePlan<BE>,
    bias: SplitBias,
    scratch: &mut ScratchArena<'_, BE>,
) -> Result<()>
where
    BE: Backend,
    M: CKKSAddOps<BE>,
    Dst: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
{
    if let Some(re) = bias.re {
        let (bank, coefficient) = plan.scalar(re);
        module.ckks_add_pt_const_assign(dst, 0, bank, coefficient, scratch)?;
    }
    if let Some(im) = bias.im {
        let (bank, coefficient) = plan.scalar(im);
        module.ckks_add_pt_const_assign(dst, dst.n().as_usize() / 2, bank, coefficient, scratch)?;
    }
    Ok(())
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

fn coefficient_is_zero<F: BlockScalar>(coefficient: Coefficient<F>) -> bool {
    let value = coefficient.value();
    value.re == F::zero() && value.im == F::zero()
}

fn exact_scales<F: BlockScalar>(coefficient: Coefficient<F>) -> Option<Vec<(u8, i64)>> {
    let (re, im, log_den) = coefficient.exact_parts()?;
    let mut terms = Vec::new();
    for (value, positive_turn, negative_turn) in [(re, 0, 2), (im, 1, 3)] {
        let mut magnitude = value.unsigned_abs();
        while magnitude != 0 {
            let bit = magnitude.trailing_zeros();
            terms.push((
                if value > 0 {
                    positive_turn
                } else {
                    negative_turn
                },
                i64::from(bit) - i64::from(log_den),
            ));
            if terms.len() > 4 {
                return None;
            }
            magnitude &= magnitude - 1;
        }
    }
    Some(terms)
}

fn validate_precision<F: BlockScalar>(
    coefficient: Coefficient<F>,
    coeffs_meta: CoeffsMeta,
) -> Result<()> {
    if let Some((_, _, log_den)) = coefficient.exact_parts() {
        ensure!(
            coeffs_meta.log_delta() >= log_den as usize,
            "coefficient precision {} cannot represent an exact dyadic denominator 2^{log_den}",
            coeffs_meta.log_delta()
        );
    }
    let value = coefficient.value();
    ensure!(
        value.re.is_finite() && value.im.is_finite(),
        "affine coefficient is not finite"
    );
    Ok(())
}

fn effective_scalar_meta<F: BlockScalar>(
    map: &AffineMap<F>,
    requested: CoeffsMeta,
) -> Result<CoeffsMeta> {
    let mut all_exact = true;
    let mut exact_log_delta = 0usize;
    for coefficient in map
        .matrix
        .iter()
        .copied()
        .filter(|coefficient| {
            !coefficient_is_zero(*coefficient) && exact_scales(*coefficient).is_none()
        })
        .chain(
            map.bias
                .iter()
                .copied()
                .filter(|coefficient| !coefficient_is_zero(*coefficient)),
        )
    {
        match coefficient.exact_parts() {
            Some((_, _, log_den)) => exact_log_delta = exact_log_delta.max(log_den as usize),
            None => all_exact = false,
        }
    }
    let log_delta = if all_exact {
        exact_log_delta
    } else {
        ensure!(
            requested.log_delta() >= exact_log_delta,
            "coefficient precision {} cannot represent an exact dyadic denominator 2^{exact_log_delta}",
            requested.log_delta()
        );
        requested.log_delta()
    };
    let mut meta = requested.meta;
    meta.log_delta = log_delta;
    meta.log_sparsity = 0;
    meta.slots = SlotsKind::Real;
    Ok(CoeffsMeta {
        k: (log_delta + requested.log_budget()).into(),
        meta,
    })
}

fn scalar_components<F: BlockScalar>(
    coefficient: Coefficient<F>,
    scalars: &mut Vec<F>,
) -> Result<(Option<ScalarIndex>, Option<ScalarIndex>)> {
    let value = coefficient.value();
    ensure!(
        value.re.is_finite() && value.im.is_finite(),
        "affine coefficient is not finite"
    );
    let mut push = |value: F| {
        if value == F::zero() {
            None
        } else {
            let index = ScalarIndex(
                scalars
                    .iter()
                    .take(64)
                    .position(|&stored| stored == value)
                    .unwrap_or_else(|| {
                        scalars.push(value);
                        scalars.len() - 1
                    }),
            );
            Some(index)
        }
    };
    Ok((push(value.re), push(value.im)))
}
