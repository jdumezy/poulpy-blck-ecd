use anyhow::{Result, ensure};
use poulpy_ckks::{
    CKKSAtkBounds, CKKSCtBounds, SetCKKSInfos,
    api::{CKKSAddOps, CKKSConjugateOps, CKKSMulOps, CKKSNegOps},
    layouts::ScratchArenaTakeCKKS,
};
use poulpy_core::GLWEBytesOf;
use poulpy_core::layouts::{
    GGLWEInfos, GLWEToBackendMut, GLWEToBackendRef, LWEInfos,
    prepared::{GGLWEPreparedToBackendRef, GLWETensorKeyPreparedToBackendRef},
};
use poulpy_hal::layouts::{Backend, Module, ScratchArena};

use super::precision::multiplication_output_k;

/// Native block products, extrema, and conjugation operations for supported encodings.
pub trait CKKSBlockMulOps<BE: Backend> {
    /// Returns the scratch-memory requirement for a packed native product.
    fn ckks_packed_native_mul_tmp_bytes<R, A, B, T>(
        &self,
        output: &R,
        lhs: &A,
        rhs: &B,
        tensor_key: &T,
    ) -> usize
    where
        R: CKKSCtBounds,
        A: CKKSCtBounds,
        B: CKKSCtBounds,
        T: GGLWEInfos;

    /// Multiplies two packed encoded values coordinatewise at the natural product width.
    fn ckks_packed_native_mul_into<Dst, A, B, T>(
        &self,
        output: &mut Dst,
        lhs: &A,
        rhs: &B,
        tensor_key: &T,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
        A: GLWEToBackendRef<BE> + CKKSCtBounds,
        B: GLWEToBackendRef<BE> + CKKSCtBounds,
        T: GGLWEInfos + GLWETensorKeyPreparedToBackendRef<BE> + GGLWEPreparedToBackendRef<BE>;

    /// Returns the scratch-memory requirement for a split native product.
    fn ckks_split_native_mul_tmp_bytes<R, A, B, T>(
        &self,
        output: &R,
        lhs: &A,
        rhs: &B,
        tensor_key: &T,
    ) -> usize
    where
        R: CKKSCtBounds,
        A: CKKSCtBounds,
        B: CKKSCtBounds,
        T: GGLWEInfos;

    /// Multiplies two split encoded values coordinatewise at the natural product width.
    fn ckks_split_native_mul_into<Dst, A, B, T>(
        &self,
        outputs: &mut [Dst],
        lhs: &[A],
        rhs: &[B],
        tensor_key: &T,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
        A: GLWEToBackendRef<BE> + CKKSCtBounds,
        B: GLWEToBackendRef<BE> + CKKSCtBounds,
        T: GGLWEInfos + GLWETensorKeyPreparedToBackendRef<BE> + GGLWEPreparedToBackendRef<BE>;

    /// Evaluates the coordinatewise product used by packed indicator equality.
    fn ckks_packed_equality_gate_into<Dst, A, B, T>(
        &self,
        output: &mut Dst,
        lhs: &A,
        rhs: &B,
        tensor_key: &T,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
        A: GLWEToBackendRef<BE> + CKKSCtBounds,
        B: GLWEToBackendRef<BE> + CKKSCtBounds,
        T: GGLWEInfos + GLWETensorKeyPreparedToBackendRef<BE> + GGLWEPreparedToBackendRef<BE>,
    {
        self.ckks_packed_native_mul_into(output, lhs, rhs, tensor_key, scratch)
    }

    /// Evaluates the coordinatewise product used by split indicator equality.
    fn ckks_split_equality_gate_into<Dst, A, B, T>(
        &self,
        outputs: &mut [Dst],
        lhs: &[A],
        rhs: &[B],
        tensor_key: &T,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
        A: GLWEToBackendRef<BE> + CKKSCtBounds,
        B: GLWEToBackendRef<BE> + CKKSCtBounds,
        T: GGLWEInfos + GLWETensorKeyPreparedToBackendRef<BE> + GGLWEPreparedToBackendRef<BE>,
    {
        self.ckks_split_native_mul_into(outputs, lhs, rhs, tensor_key, scratch)
    }

    /// Returns the scratch-memory requirement for a native maximum operation.
    fn ckks_native_max_tmp_bytes<R, A, B, T>(
        &self,
        output: &R,
        lhs: &A,
        rhs: &B,
        tensor_key: &T,
    ) -> usize
    where
        R: CKKSCtBounds,
        A: CKKSCtBounds,
        B: CKKSCtBounds,
        T: GGLWEInfos;

    /// Evaluates a native maximum on packed encoded values at the natural product width.
    fn ckks_packed_native_max_into<Dst, A, B, T>(
        &self,
        output: &mut Dst,
        lhs: &A,
        rhs: &B,
        tensor_key: &T,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
        A: GLWEToBackendRef<BE> + CKKSCtBounds,
        B: GLWEToBackendRef<BE> + CKKSCtBounds,
        T: GGLWEInfos + GLWETensorKeyPreparedToBackendRef<BE> + GGLWEPreparedToBackendRef<BE>;

    /// Evaluates a native maximum on split encoded values at the natural product width.
    fn ckks_split_native_max_into<Dst, A, B, T>(
        &self,
        outputs: &mut [Dst],
        lhs: &[A],
        rhs: &[B],
        tensor_key: &T,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
        A: GLWEToBackendRef<BE> + CKKSCtBounds,
        B: GLWEToBackendRef<BE> + CKKSCtBounds,
        T: GGLWEInfos + GLWETensorKeyPreparedToBackendRef<BE> + GGLWEPreparedToBackendRef<BE>;

    /// Returns the scratch-memory requirement for blockwise complex conjugation.
    fn ckks_block_conjugate_tmp_bytes<C, K>(&self, input: &C, key: &K) -> usize
    where
        C: CKKSCtBounds,
        K: GGLWEInfos;

    /// Conjugates every encoded block in a packed ciphertext.
    fn ckks_packed_block_conjugate_into<Dst, Src, K>(
        &self,
        output: &mut Dst,
        input: &Src,
        key: &K,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
        Src: GLWEToBackendRef<BE> + CKKSCtBounds,
        K: CKKSAtkBounds<BE>;

    /// Conjugates every coordinate ciphertext of a split encoding.
    fn ckks_split_block_conjugate_into<Dst, Src, K>(
        &self,
        outputs: &mut [Dst],
        inputs: &[Src],
        key: &K,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
        Src: GLWEToBackendRef<BE> + CKKSCtBounds,
        K: CKKSAtkBounds<BE>;

    /// Returns the scratch-memory requirement for a conjugated native product.
    fn ckks_conjugate_product_tmp_bytes<R, A, B, K, T>(
        &self,
        output: &R,
        lhs: &A,
        rhs: &B,
        conjugation_key: &K,
        tensor_key: &T,
    ) -> usize
    where
        R: CKKSCtBounds,
        A: CKKSCtBounds,
        B: CKKSCtBounds,
        K: GGLWEInfos,
        T: GGLWEInfos;

    #[allow(clippy::too_many_arguments)]
    /// Multiplies a packed encoded value by the blockwise conjugate of another at the natural product width.
    fn ckks_packed_conjugate_product_into<Dst, A, B, K, T>(
        &self,
        output: &mut Dst,
        lhs: &A,
        rhs: &B,
        conjugation_key: &K,
        tensor_key: &T,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
        A: GLWEToBackendRef<BE> + CKKSCtBounds,
        B: GLWEToBackendRef<BE> + CKKSCtBounds,
        K: CKKSAtkBounds<BE>,
        T: GGLWEInfos + GLWETensorKeyPreparedToBackendRef<BE> + GGLWEPreparedToBackendRef<BE>;

    #[allow(clippy::too_many_arguments)]
    /// Multiplies split encoded values after blockwise conjugating the right input at the natural product width.
    fn ckks_split_conjugate_product_into<Dst, A, B, K, T>(
        &self,
        outputs: &mut [Dst],
        lhs: &[A],
        rhs: &[B],
        conjugation_key: &K,
        tensor_key: &T,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
        A: GLWEToBackendRef<BE> + CKKSCtBounds,
        B: GLWEToBackendRef<BE> + CKKSCtBounds,
        K: CKKSAtkBounds<BE>,
        T: GGLWEInfos + GLWETensorKeyPreparedToBackendRef<BE> + GGLWEPreparedToBackendRef<BE>;
}

impl<BE: Backend> CKKSBlockMulOps<BE> for Module<BE>
where
    Module<BE>:
        CKKSAddOps<BE> + CKKSConjugateOps<BE> + CKKSMulOps<BE> + CKKSNegOps<BE> + GLWEBytesOf<BE>,
{
    fn ckks_packed_native_mul_tmp_bytes<R, A, B, T>(
        &self,
        output: &R,
        lhs: &A,
        rhs: &B,
        tensor_key: &T,
    ) -> usize
    where
        R: CKKSCtBounds,
        A: CKKSCtBounds,
        B: CKKSCtBounds,
        T: GGLWEInfos,
    {
        self.ckks_mul_tmp_bytes(output, lhs, rhs, tensor_key)
    }

    fn ckks_packed_native_mul_into<Dst, A, B, T>(
        &self,
        output: &mut Dst,
        lhs: &A,
        rhs: &B,
        tensor_key: &T,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
        A: GLWEToBackendRef<BE> + CKKSCtBounds,
        B: GLWEToBackendRef<BE> + CKKSCtBounds,
        T: GGLWEInfos + GLWETensorKeyPreparedToBackendRef<BE> + GGLWEPreparedToBackendRef<BE>,
    {
        output.set_k(
            output
                .k()
                .as_usize()
                .min(multiplication_output_k(
                    lhs.k().as_usize(),
                    lhs.log_delta(),
                    rhs.k().as_usize(),
                    rhs.log_delta(),
                ))
                .into(),
        );
        self.ckks_mul_into(output, lhs, rhs, tensor_key, scratch)?;
        Ok(())
    }

    fn ckks_split_native_mul_tmp_bytes<R, A, B, T>(
        &self,
        output: &R,
        lhs: &A,
        rhs: &B,
        tensor_key: &T,
    ) -> usize
    where
        R: CKKSCtBounds,
        A: CKKSCtBounds,
        B: CKKSCtBounds,
        T: GGLWEInfos,
    {
        self.ckks_mul_tmp_bytes(output, lhs, rhs, tensor_key)
    }

    fn ckks_split_native_mul_into<Dst, A, B, T>(
        &self,
        outputs: &mut [Dst],
        lhs: &[A],
        rhs: &[B],
        tensor_key: &T,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
        A: GLWEToBackendRef<BE> + CKKSCtBounds,
        B: GLWEToBackendRef<BE> + CKKSCtBounds,
        T: GGLWEInfos + GLWETensorKeyPreparedToBackendRef<BE> + GGLWEPreparedToBackendRef<BE>,
    {
        ensure!(
            lhs.len() == rhs.len(),
            "split native-product inputs have different widths"
        );
        ensure!(
            outputs.len() == lhs.len(),
            "split native-product output has the wrong width"
        );
        for ((output, lhs), rhs) in outputs.iter_mut().zip(lhs).zip(rhs) {
            self.ckks_packed_native_mul_into(output, lhs, rhs, tensor_key, scratch)?;
        }
        Ok(())
    }

    fn ckks_native_max_tmp_bytes<R, A, B, T>(
        &self,
        output: &R,
        lhs: &A,
        rhs: &B,
        tensor_key: &T,
    ) -> usize
    where
        R: CKKSCtBounds,
        A: CKKSCtBounds,
        B: CKKSCtBounds,
        T: GGLWEInfos,
    {
        self.ckks_mul_tmp_bytes(output, lhs, rhs, tensor_key)
            .max(self.ckks_add_tmp_bytes())
            .max(self.ckks_neg_tmp_bytes())
    }

    fn ckks_packed_native_max_into<Dst, A, B, T>(
        &self,
        output: &mut Dst,
        lhs: &A,
        rhs: &B,
        tensor_key: &T,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
        A: GLWEToBackendRef<BE> + CKKSCtBounds,
        B: GLWEToBackendRef<BE> + CKKSCtBounds,
        T: GGLWEInfos + GLWETensorKeyPreparedToBackendRef<BE> + GGLWEPreparedToBackendRef<BE>,
    {
        output.set_k(
            output
                .k()
                .as_usize()
                .min(multiplication_output_k(
                    lhs.k().as_usize(),
                    lhs.log_delta(),
                    rhs.k().as_usize(),
                    rhs.log_delta(),
                ))
                .into(),
        );
        self.ckks_mul_into(output, lhs, rhs, tensor_key, scratch)?;
        self.ckks_neg_assign(output)?;
        self.ckks_add_assign(output, lhs, scratch)?;
        self.ckks_add_assign(output, rhs, scratch)?;
        Ok(())
    }

    fn ckks_split_native_max_into<Dst, A, B, T>(
        &self,
        outputs: &mut [Dst],
        lhs: &[A],
        rhs: &[B],
        tensor_key: &T,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
        A: GLWEToBackendRef<BE> + CKKSCtBounds,
        B: GLWEToBackendRef<BE> + CKKSCtBounds,
        T: GGLWEInfos + GLWETensorKeyPreparedToBackendRef<BE> + GGLWEPreparedToBackendRef<BE>,
    {
        ensure!(
            lhs.len() == rhs.len(),
            "split max inputs have different widths"
        );
        ensure!(
            outputs.len() == lhs.len(),
            "split max output has the wrong width"
        );
        for ((output, lhs), rhs) in outputs.iter_mut().zip(lhs).zip(rhs) {
            self.ckks_packed_native_max_into(output, lhs, rhs, tensor_key, scratch)?;
        }
        Ok(())
    }

    fn ckks_block_conjugate_tmp_bytes<C, K>(&self, input: &C, key: &K) -> usize
    where
        C: CKKSCtBounds,
        K: GGLWEInfos,
    {
        self.ckks_conjugate_tmp_bytes(input, key)
    }

    fn ckks_packed_block_conjugate_into<Dst, Src, K>(
        &self,
        output: &mut Dst,
        input: &Src,
        key: &K,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
        Src: GLWEToBackendRef<BE> + CKKSCtBounds,
        K: CKKSAtkBounds<BE>,
    {
        output.set_k(output.k().as_usize().min(input.k().as_usize()).into());
        self.ckks_conjugate_into(output, input, key, scratch)?;
        Ok(())
    }

    fn ckks_split_block_conjugate_into<Dst, Src, K>(
        &self,
        outputs: &mut [Dst],
        inputs: &[Src],
        key: &K,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
        Src: GLWEToBackendRef<BE> + CKKSCtBounds,
        K: CKKSAtkBounds<BE>,
    {
        ensure!(outputs.len() == inputs.len(), "conjugation widths differ");
        for (output, input) in outputs.iter_mut().zip(inputs) {
            self.ckks_conjugate_into(output, input, key, scratch)?;
        }
        Ok(())
    }

    fn ckks_conjugate_product_tmp_bytes<R, A, B, K, T>(
        &self,
        output: &R,
        lhs: &A,
        rhs: &B,
        conjugation_key: &K,
        tensor_key: &T,
    ) -> usize
    where
        R: CKKSCtBounds,
        A: CKKSCtBounds,
        B: CKKSCtBounds,
        K: GGLWEInfos,
        T: GGLWEInfos,
    {
        self.glwe_bytes_of_from_infos(rhs)
            + self
                .ckks_conjugate_tmp_bytes(rhs, conjugation_key)
                .max(self.ckks_mul_tmp_bytes(output, lhs, rhs, tensor_key))
    }

    fn ckks_packed_conjugate_product_into<Dst, A, B, K, T>(
        &self,
        output: &mut Dst,
        lhs: &A,
        rhs: &B,
        conjugation_key: &K,
        tensor_key: &T,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
        A: GLWEToBackendRef<BE> + CKKSCtBounds,
        B: GLWEToBackendRef<BE> + CKKSCtBounds,
        K: CKKSAtkBounds<BE>,
        T: GGLWEInfos + GLWETensorKeyPreparedToBackendRef<BE> + GGLWEPreparedToBackendRef<BE>,
    {
        output.set_k(
            output
                .k()
                .as_usize()
                .min(multiplication_output_k(
                    lhs.k().as_usize(),
                    lhs.log_delta(),
                    rhs.k().as_usize(),
                    rhs.log_delta(),
                ))
                .into(),
        );
        scratch.scope(|local| {
            let (mut conjugate, mut local) = local.take_ckks_ciphertext_like_scratch(rhs);
            self.ckks_conjugate_into(&mut conjugate, rhs, conjugation_key, &mut local)?;
            self.ckks_mul_into(output, lhs, &conjugate, tensor_key, &mut local)
        })?;
        Ok(())
    }

    fn ckks_split_conjugate_product_into<Dst, A, B, K, T>(
        &self,
        outputs: &mut [Dst],
        lhs: &[A],
        rhs: &[B],
        conjugation_key: &K,
        tensor_key: &T,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
        A: GLWEToBackendRef<BE> + CKKSCtBounds,
        B: GLWEToBackendRef<BE> + CKKSCtBounds,
        K: CKKSAtkBounds<BE>,
        T: GGLWEInfos + GLWETensorKeyPreparedToBackendRef<BE> + GGLWEPreparedToBackendRef<BE>,
    {
        ensure!(
            lhs.len() == rhs.len(),
            "conjugate-product inputs have different widths"
        );
        ensure!(
            outputs.len() == lhs.len(),
            "conjugate-product output has the wrong width"
        );
        for ((output, lhs), rhs) in outputs.iter_mut().zip(lhs).zip(rhs) {
            self.ckks_packed_conjugate_product_into(
                output,
                lhs,
                rhs,
                conjugation_key,
                tensor_key,
                scratch,
            )?;
        }
        Ok(())
    }
}
