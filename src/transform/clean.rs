use anyhow::{Result, ensure};
use poulpy_ckks::{
    CKKSCtBounds, SetCKKSInfos,
    api::{CKKSAddOps, CKKSCopyOps, CKKSMulOps, CKKSNegOps, CKKSPow2Ops},
    layouts::ScratchArenaTakeCKKS,
};
use poulpy_core::{
    GLWEBytesOf,
    layouts::{
        GGLWEInfos, GLWEToBackendMut, GLWEToBackendRef,
        prepared::{GGLWEPreparedToBackendRef, GLWETensorKeyPreparedToBackendRef},
    },
};
use poulpy_hal::layouts::{Backend, Module, ScratchArena};

use crate::algebra::CleaningMode;

pub trait CKKSBlockCleaningOps<BE: Backend> {
    fn ckks_clean_tmp_bytes<R, A, T>(&self, output: &R, input: &A, tensor_key: &T) -> usize
    where
        R: CKKSCtBounds,
        A: CKKSCtBounds,
        T: GGLWEInfos;

    fn ckks_clean_into<Dst, Src, T>(
        &self,
        output: &mut Dst,
        input: &Src,
        mode: CleaningMode,
        tensor_key: &T,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
        Src: GLWEToBackendRef<BE> + CKKSCtBounds,
        T: GGLWEInfos + GLWETensorKeyPreparedToBackendRef<BE> + GGLWEPreparedToBackendRef<BE>;

    fn ckks_clean_split_into<Dst, Src, T>(
        &self,
        outputs: &mut [Dst],
        inputs: &[Src],
        mode: CleaningMode,
        tensor_key: &T,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
        Src: GLWEToBackendRef<BE> + CKKSCtBounds,
        T: GGLWEInfos + GLWETensorKeyPreparedToBackendRef<BE> + GGLWEPreparedToBackendRef<BE>;
}

impl<BE: Backend> CKKSBlockCleaningOps<BE> for Module<BE>
where
    Module<BE>: CKKSAddOps<BE>
        + CKKSCopyOps<BE>
        + CKKSMulOps<BE>
        + CKKSNegOps<BE>
        + CKKSPow2Ops<BE>
        + GLWEBytesOf<BE>,
{
    fn ckks_clean_tmp_bytes<R, A, T>(&self, output: &R, input: &A, tensor_key: &T) -> usize
    where
        R: CKKSCtBounds,
        A: CKKSCtBounds,
        T: GGLWEInfos,
    {
        self.glwe_bytes_of_from_infos(input)
            + self
                .ckks_square_tmp_bytes(output, input, tensor_key)
                .max(self.ckks_mul_tmp_bytes(output, input, input, tensor_key))
                .max(self.ckks_copy_tmp_bytes())
                .max(self.ckks_add_tmp_bytes())
                .max(self.ckks_neg_tmp_bytes())
                .max(self.ckks_mul_pow2_tmp_bytes())
                .max(self.ckks_div_pow2_tmp_bytes())
    }

    fn ckks_clean_into<Dst, Src, T>(
        &self,
        output: &mut Dst,
        input: &Src,
        mode: CleaningMode,
        tensor_key: &T,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
        Src: GLWEToBackendRef<BE> + CKKSCtBounds,
        T: GGLWEInfos + GLWETensorKeyPreparedToBackendRef<BE> + GGLWEPreparedToBackendRef<BE>,
    {
        scratch.scope(|local| {
            let (mut square, mut local) = local.take_ckks_ciphertext_scratch(input, input.meta());
            self.ckks_square_into(&mut square, input, tensor_key, &mut local)?;
            self.ckks_mul_into(output, &square, input, tensor_key, &mut local)?;
            self.ckks_neg_assign(output)?;
            match mode {
                CleaningMode::Binary => {
                    self.ckks_mul_pow2_assign(output, 1, &mut local)?;
                    self.ckks_add_assign(output, &square, &mut local)?;
                    self.ckks_mul_pow2_assign(&mut square, 1, &mut local)?;
                    self.ckks_add_assign(output, &square, &mut local)?;
                }
                CleaningMode::Sign => {
                    self.ckks_add_assign(output, input, &mut local)?;
                    self.ckks_copy(&mut square, input, &mut local)?;
                    self.ckks_mul_pow2_assign(&mut square, 1, &mut local)?;
                    self.ckks_add_assign(output, &square, &mut local)?;
                    self.ckks_div_pow2_assign(output, 1)?;
                }
            }
            Ok(())
        })
    }

    fn ckks_clean_split_into<Dst, Src, T>(
        &self,
        outputs: &mut [Dst],
        inputs: &[Src],
        mode: CleaningMode,
        tensor_key: &T,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
        Src: GLWEToBackendRef<BE> + CKKSCtBounds,
        T: GGLWEInfos + GLWETensorKeyPreparedToBackendRef<BE> + GGLWEPreparedToBackendRef<BE>,
    {
        ensure!(outputs.len() == inputs.len(), "cleaning widths differ");
        for (output, input) in outputs.iter_mut().zip(inputs) {
            self.ckks_clean_into(output, input, mode, tensor_key, scratch)?;
        }
        Ok(())
    }
}
