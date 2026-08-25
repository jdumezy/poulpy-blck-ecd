use std::collections::HashMap;

use poulpy_ckks::{
    CoeffsMeta,
    api::{CKKSEncodingOps, CKKSLinearTransformationOps},
    test_suite::{
        CKKSTestParams,
        helpers::{
            TestContextBackend, TestContextHostModule, TestContextModule, TestScalar, alloc_ct,
            alloc_scratch, assert_decrypt_precision_at_log_delta, ckks_encrypt, gen_atk,
            gen_sk_with_raw, gen_tsk,
        },
        reference_encoder::ReferenceEncoder,
    },
};
use poulpy_hal::{
    api::{
        CnvPVecAlloc, ModuleN, NegacyclicFFT, NegacyclicFFTNew, ScratchOwnedAlloc,
        ScratchOwnedBorrow,
    },
    layouts::{HostBytesBackend, HostDataMut, HostDataRef, Module, ScratchOwned},
};

use crate::{
    algebra::{
        AffineMap, Bru, CleaningMode, Indicator, Thermometer, compile_lut, compile_multivariate_lut,
    },
    layout::{PackedLayout, SplitLayout},
    scalar::BlockScalar,
    transform::{
        CKKSBlockCleaningOps, CKKSBlockMulOps, CKKSCleaningCircuitOps, CKKSMultivariateOps,
        CKKSPackedAffineOps, CKKSSplitAffineOps, PackedAffinePlan, PackedCleaningPlan,
        PackedMultivariatePlan, SplitAffinePlan, SplitCleaningPlan, SplitMultivariatePlan,
        TransformStrategy,
    },
};

pub fn test_layouts<BE, F, E>(
    params: CKKSTestParams,
    module: &Module<BE>,
    host_module: &Module<HostBytesBackend>,
) where
    BE: TestContextBackend,
    Module<BE>: TestContextModule<BE>
        + CKKSEncodingOps<BE, F>
        + CKKSLinearTransformationOps<BE>
        + CnvPVecAlloc<BE>
        + ModuleN,
    Module<HostBytesBackend>: TestContextHostModule,
    F: TestScalar + BlockScalar,
    E: NegacyclicFFT<F> + NegacyclicFFTNew<F>,
    for<'a> BE::BufRef<'a>: HostDataRef,
    for<'a> BE::BufMut<'a>: HostDataMut,
{
    let slots = params.n / 2;
    let encoder = ReferenceEncoder::<E>::new::<F>(slots).unwrap();
    let (sk_raw, sk) = gen_sk_with_raw(&params, module, host_module, [17u8; 32]);
    let mut scratch = alloc_scratch(&params, module);
    let tensor_key = gen_tsk(&params, module, &sk_raw, &mut scratch.borrow());

    test_packed(
        &params,
        module,
        host_module,
        &encoder,
        &sk_raw,
        &sk,
        &tensor_key,
        &mut scratch,
    );
    test_split(
        &params,
        module,
        host_module,
        &encoder,
        &sk,
        &tensor_key,
        &mut scratch,
    );
    test_multivariate_and_cleaning(
        &params,
        module,
        host_module,
        &encoder,
        &sk_raw,
        &sk,
        &tensor_key,
        &mut scratch,
    );
}

#[allow(clippy::too_many_arguments)]
fn test_packed<BE, F, E, T>(
    params: &CKKSTestParams,
    module: &Module<BE>,
    host_module: &Module<HostBytesBackend>,
    encoder: &ReferenceEncoder<E>,
    sk_raw: &poulpy_core::layouts::BackendGLWESecret<BE>,
    sk: &poulpy_core::layouts::prepared::GLWESecretPrepared<BE::OwnedBuf, BE>,
    tensor_key: &T,
    scratch: &mut poulpy_hal::layouts::ScratchOwned<BE>,
) where
    BE: TestContextBackend,
    Module<BE>: TestContextModule<BE>
        + CKKSEncodingOps<BE, F>
        + CKKSLinearTransformationOps<BE>
        + CnvPVecAlloc<BE>
        + ModuleN,
    Module<HostBytesBackend>: TestContextHostModule,
    F: TestScalar + BlockScalar,
    E: NegacyclicFFT<F>,
    T: poulpy_core::layouts::GGLWEInfos
        + poulpy_core::layouts::prepared::GLWETensorKeyPreparedToBackendRef<BE>
        + poulpy_core::layouts::prepared::GGLWEPreparedToBackendRef<BE>,
    for<'a> BE::BufRef<'a>: HostDataRef,
    for<'a> BE::BufMut<'a>: HostDataMut,
{
    let input = Bru::new(4).unwrap();
    let output = Indicator::new(4, 0).unwrap();
    let table = [1, 2, 3, 0];
    let map: AffineMap<F> = compile_lut(&input, &output, &table).unwrap();
    let layout = PackedLayout::for_widths(params.n / 2, map.cols(), map.rows()).unwrap();
    let values = (0..layout.block_count())
        .map(|value| value % 4)
        .collect::<Vec<_>>();
    let input_slots = layout.encode_slots::<F, _>(&input, &values).unwrap();
    let input_ct = ckks_encrypt(
        params,
        module,
        host_module,
        encoder,
        sk,
        params.k,
        &input_slots.re,
        &input_slots.im,
        &mut scratch.borrow(),
    );
    let plan = PackedAffinePlan::compile(
        module,
        layout,
        &map,
        params.base2k.into(),
        CoeffsMeta::from_delta_budget(24, 4),
        TransformStrategy::Auto,
        &mut scratch.borrow(),
    )
    .unwrap();
    let mut keys = HashMap::new();
    for &galois_element in plan.galois_elements() {
        keys.insert(
            galois_element,
            gen_atk(
                params,
                module,
                galois_element,
                sk_raw,
                &mut scratch.borrow(),
            ),
        );
    }
    let mut workspace = plan.alloc_workspace(module, &input_ct);
    let mut transformed = alloc_ct(params, module, params.k);
    module
        .ckks_packed_affine_into(
            &mut transformed,
            &input_ct,
            &plan,
            &mut workspace,
            &keys,
            &mut scratch.borrow(),
        )
        .unwrap();
    let mapped = values.iter().map(|&value| table[value]).collect::<Vec<_>>();
    let want = layout.encode_slots::<F, _>(&output, &mapped).unwrap();
    assert_decrypt_precision_at_log_delta(
        "packed_affine",
        params,
        module,
        encoder,
        &transformed,
        sk,
        &want.re,
        &want.im,
        20,
        &mut scratch.borrow(),
    );

    let rhs_values = values
        .iter()
        .map(|value| (3 * value + 1) % 4)
        .collect::<Vec<_>>();
    let rhs_slots = layout.encode_slots::<F, _>(&input, &rhs_values).unwrap();
    let rhs_ct = ckks_encrypt(
        params,
        module,
        host_module,
        encoder,
        sk,
        params.k,
        &rhs_slots.re,
        &rhs_slots.im,
        &mut scratch.borrow(),
    );
    let mut product = alloc_ct(params, module, params.k);
    module
        .ckks_packed_native_mul_into(
            &mut product,
            &input_ct,
            &rhs_ct,
            tensor_key,
            &mut scratch.borrow(),
        )
        .unwrap();
    let product_values = values
        .iter()
        .zip(&rhs_values)
        .map(|(&lhs, &rhs)| (lhs + rhs) % 4)
        .collect::<Vec<_>>();
    let want = layout
        .encode_slots::<F, _>(&input, &product_values)
        .unwrap();
    assert_decrypt_precision_at_log_delta(
        "packed_native_product",
        params,
        module,
        encoder,
        &product,
        sk,
        &want.re,
        &want.im,
        30,
        &mut scratch.borrow(),
    );

    let conjugation_key = gen_atk(
        params,
        module,
        (2 * module.n() - 1) as i64,
        sk_raw,
        &mut scratch.borrow(),
    );
    let mut conjugated = alloc_ct(params, module, params.k);
    module
        .ckks_packed_block_conjugate_into(
            &mut conjugated,
            &input_ct,
            &conjugation_key,
            &mut scratch.borrow(),
        )
        .unwrap();
    let conjugated_values = values
        .iter()
        .map(|value| (4 - value) % 4)
        .collect::<Vec<_>>();
    let want = layout
        .encode_slots::<F, _>(&input, &conjugated_values)
        .unwrap();
    assert_decrypt_precision_at_log_delta(
        "packed_conjugate",
        params,
        module,
        encoder,
        &conjugated,
        sk,
        &want.re,
        &want.im,
        35,
        &mut scratch.borrow(),
    );

    let mut difference = alloc_ct(params, module, params.k);
    module
        .ckks_packed_conjugate_product_into(
            &mut difference,
            &input_ct,
            &rhs_ct,
            &conjugation_key,
            tensor_key,
            &mut scratch.borrow(),
        )
        .unwrap();
    let difference_values = values
        .iter()
        .zip(&rhs_values)
        .map(|(&lhs, &rhs)| (lhs + 4 - rhs) % 4)
        .collect::<Vec<_>>();
    let want = layout
        .encode_slots::<F, _>(&input, &difference_values)
        .unwrap();
    assert_decrypt_precision_at_log_delta(
        "packed_conjugate_product",
        params,
        module,
        encoder,
        &difference,
        sk,
        &want.re,
        &want.im,
        30,
        &mut scratch.borrow(),
    );
}

#[allow(clippy::too_many_arguments)]
fn test_split<BE, F, E, T>(
    params: &CKKSTestParams,
    module: &Module<BE>,
    host_module: &Module<HostBytesBackend>,
    encoder: &ReferenceEncoder<E>,
    sk: &poulpy_core::layouts::prepared::GLWESecretPrepared<BE::OwnedBuf, BE>,
    tensor_key: &T,
    scratch: &mut poulpy_hal::layouts::ScratchOwned<BE>,
) where
    BE: TestContextBackend,
    Module<BE>: TestContextModule<BE> + CKKSEncodingOps<BE, F> + ModuleN,
    Module<HostBytesBackend>: TestContextHostModule,
    F: TestScalar + BlockScalar,
    E: NegacyclicFFT<F>,
    T: poulpy_core::layouts::GGLWEInfos
        + poulpy_core::layouts::prepared::GLWETensorKeyPreparedToBackendRef<BE>
        + poulpy_core::layouts::prepared::GGLWEPreparedToBackendRef<BE>,
    for<'a> BE::BufRef<'a>: HostDataRef,
    for<'a> BE::BufMut<'a>: HostDataMut,
{
    let input = Thermometer::new(4).unwrap();
    let output = Indicator::new(4, 0).unwrap();
    let table = [3, 2, 1, 0];
    let map: AffineMap<F> = compile_lut(&input, &output, &table).unwrap();
    let input_width = map.cols();
    let output_width = map.rows();
    let layout = SplitLayout::new(params.n / 2).unwrap();
    let values = (0..layout.slots())
        .map(|value| value % 4)
        .collect::<Vec<_>>();
    let input_slots = layout.encode_slots::<F, _>(&input, &values).unwrap();
    let mut inputs = Vec::with_capacity(input_width);
    for coordinate in 0..input_width {
        inputs.push(ckks_encrypt(
            params,
            module,
            host_module,
            encoder,
            sk,
            params.k,
            &input_slots.re[coordinate],
            &input_slots.im[coordinate],
            &mut scratch.borrow(),
        ));
    }
    let plan = SplitAffinePlan::compile(
        module,
        &map,
        params.base2k.into(),
        CoeffsMeta::from_delta_budget(16, 4),
        &mut scratch.borrow(),
    )
    .unwrap();
    let mut transformed = (0..output_width)
        .map(|_| alloc_ct(params, module, params.k))
        .collect::<Vec<_>>();
    module
        .ckks_split_affine_into(&mut transformed, &inputs, &plan, &mut scratch.borrow())
        .unwrap();
    let mapped = values.iter().map(|&value| table[value]).collect::<Vec<_>>();
    let want = layout.encode_slots::<F, _>(&output, &mapped).unwrap();
    for (coordinate, ciphertext) in transformed.iter().enumerate() {
        assert_decrypt_precision_at_log_delta(
            "split_affine",
            params,
            module,
            encoder,
            ciphertext,
            sk,
            &want.re[coordinate],
            &want.im[coordinate],
            35,
            &mut scratch.borrow(),
        );
    }

    let rhs_values = values.iter().map(|value| 3 - value).collect::<Vec<_>>();
    let rhs_slots = layout.encode_slots::<F, _>(&input, &rhs_values).unwrap();
    let mut rhs = Vec::with_capacity(input_width);
    for coordinate in 0..input_width {
        rhs.push(ckks_encrypt(
            params,
            module,
            host_module,
            encoder,
            sk,
            params.k,
            &rhs_slots.re[coordinate],
            &rhs_slots.im[coordinate],
            &mut scratch.borrow(),
        ));
    }
    let mut product = (0..input_width)
        .map(|_| alloc_ct(params, module, params.k))
        .collect::<Vec<_>>();
    module
        .ckks_split_native_mul_into(
            &mut product,
            &inputs,
            &rhs,
            tensor_key,
            &mut scratch.borrow(),
        )
        .unwrap();
    let product_values = values
        .iter()
        .zip(&rhs_values)
        .map(|(&lhs, &rhs)| lhs.min(rhs))
        .collect::<Vec<_>>();
    let want = layout
        .encode_slots::<F, _>(&input, &product_values)
        .unwrap();
    for (coordinate, ciphertext) in product.iter().enumerate() {
        assert_decrypt_precision_at_log_delta(
            "split_native_product",
            params,
            module,
            encoder,
            ciphertext,
            sk,
            &want.re[coordinate],
            &want.im[coordinate],
            30,
            &mut scratch.borrow(),
        );
    }

    let mut maximum = (0..input_width)
        .map(|_| alloc_ct(params, module, params.k))
        .collect::<Vec<_>>();
    module
        .ckks_split_native_max_into(
            &mut maximum,
            &inputs,
            &rhs,
            tensor_key,
            &mut scratch.borrow(),
        )
        .unwrap();
    let maximum_values = values
        .iter()
        .zip(&rhs_values)
        .map(|(&lhs, &rhs)| lhs.max(rhs))
        .collect::<Vec<_>>();
    let want = layout
        .encode_slots::<F, _>(&input, &maximum_values)
        .unwrap();
    for (coordinate, ciphertext) in maximum.iter().enumerate() {
        assert_decrypt_precision_at_log_delta(
            "split_native_max",
            params,
            module,
            encoder,
            ciphertext,
            sk,
            &want.re[coordinate],
            &want.im[coordinate],
            30,
            &mut scratch.borrow(),
        );
    }
}

#[allow(clippy::too_many_arguments)]
fn test_multivariate_and_cleaning<BE, F, E, T>(
    params: &CKKSTestParams,
    module: &Module<BE>,
    host_module: &Module<HostBytesBackend>,
    encoder: &ReferenceEncoder<E>,
    sk_raw: &poulpy_core::layouts::BackendGLWESecret<BE>,
    sk: &poulpy_core::layouts::prepared::GLWESecretPrepared<BE::OwnedBuf, BE>,
    tensor_key: &T,
    scratch: &mut poulpy_hal::layouts::ScratchOwned<BE>,
) where
    BE: TestContextBackend,
    Module<BE>: TestContextModule<BE>
        + CKKSEncodingOps<BE, F>
        + CKKSLinearTransformationOps<BE>
        + CnvPVecAlloc<BE>
        + ModuleN,
    Module<HostBytesBackend>: TestContextHostModule,
    F: TestScalar + BlockScalar,
    E: NegacyclicFFT<F>,
    T: poulpy_core::layouts::GGLWEInfos
        + poulpy_core::layouts::prepared::GLWETensorKeyPreparedToBackendRef<BE>
        + poulpy_core::layouts::prepared::GGLWEPreparedToBackendRef<BE>,
    for<'a> BE::BufRef<'a>: HostDataRef,
    for<'a> BE::BufMut<'a>: HostDataMut,
{
    let input = Bru::new(4).unwrap();
    let output = Indicator::new(4, 0).unwrap();
    let table = (0..4)
        .flat_map(|lhs| (0..4).map(move |rhs| (lhs + 2 * rhs) % 4))
        .collect::<Vec<_>>();
    let tensor = compile_multivariate_lut::<F, _>(&[&input, &input], &output, &table).unwrap();

    let packed_layout = PackedLayout::new(params.n / 2, tensor.feature_width()).unwrap();
    let lhs_values = (0..packed_layout.block_count())
        .map(|value| value % 4)
        .collect::<Vec<_>>();
    let rhs_values = lhs_values
        .iter()
        .map(|value| (3 * value + 1) % 4)
        .collect::<Vec<_>>();
    let lhs_slots = packed_layout
        .encode_slots::<F, _>(&input, &lhs_values)
        .unwrap();
    let rhs_slots = packed_layout
        .encode_slots::<F, _>(&input, &rhs_values)
        .unwrap();
    let lhs = ckks_encrypt(
        params,
        module,
        host_module,
        encoder,
        sk,
        params.k,
        &lhs_slots.re,
        &lhs_slots.im,
        &mut scratch.borrow(),
    );
    let rhs = ckks_encrypt(
        params,
        module,
        host_module,
        encoder,
        sk,
        params.k,
        &rhs_slots.re,
        &rhs_slots.im,
        &mut scratch.borrow(),
    );
    let plan = PackedMultivariatePlan::compile(
        module,
        packed_layout,
        &tensor,
        params.base2k.into(),
        CoeffsMeta::from_delta_budget(24, 4),
        CoeffsMeta::from_delta_budget(24, 4),
        TransformStrategy::Auto,
        &mut scratch.borrow(),
    )
    .unwrap();
    let mut keys = HashMap::new();
    for &galois_element in plan.galois_elements() {
        keys.insert(
            galois_element,
            gen_atk(
                params,
                module,
                galois_element,
                sk_raw,
                &mut scratch.borrow(),
            ),
        );
    }
    let mut workspace = plan.alloc_workspace(module, &lhs);
    let mut result = alloc_ct(params, module, params.k);
    module
        .ckks_packed_multivariate_into(
            &mut result,
            &[lhs, rhs],
            &plan,
            &mut workspace,
            &keys,
            tensor_key,
            &mut scratch.borrow(),
        )
        .unwrap();
    let mapped = lhs_values
        .iter()
        .zip(&rhs_values)
        .map(|(&lhs, &rhs)| (lhs + 2 * rhs) % 4)
        .collect::<Vec<_>>();
    let want = packed_layout
        .encode_slots::<F, _>(&output, &mapped)
        .unwrap();
    assert_decrypt_precision_at_log_delta(
        "packed_multivariate",
        params,
        module,
        encoder,
        &result,
        sk,
        &want.re,
        &want.im,
        12,
        &mut scratch.borrow(),
    );

    let clean_layout = PackedLayout::new(params.n / 2, 3).unwrap();
    let clean_values = (0..clean_layout.block_count())
        .map(|value| value % 4)
        .collect::<Vec<_>>();
    let clean_slots = clean_layout
        .encode_slots::<F, _>(&output, &clean_values)
        .unwrap();
    let mut noisy_slots = clean_slots.clone();
    let noise = F::from_f64(0.02).unwrap();
    for block in 0..clean_layout.block_count() {
        for coordinate in 0..3 {
            let slot = clean_layout.slot(block, coordinate);
            noisy_slots.re[slot] = if noisy_slots.re[slot] == F::zero() {
                noise
            } else {
                noisy_slots.re[slot] - noise
            };
        }
    }
    let clean_input = ckks_encrypt(
        params,
        module,
        host_module,
        encoder,
        sk,
        params.k,
        &noisy_slots.re,
        &noisy_slots.im,
        &mut scratch.borrow(),
    );
    let mut cleaned = alloc_ct(params, module, params.k);
    module
        .ckks_clean_into(
            &mut cleaned,
            &clean_input,
            CleaningMode::Binary,
            tensor_key,
            &mut scratch.borrow(),
        )
        .unwrap();
    assert_decrypt_precision_at_log_delta(
        "packed_cleaning",
        params,
        module,
        encoder,
        &cleaned,
        sk,
        &clean_slots.re,
        &clean_slots.im,
        18,
        &mut scratch.borrow(),
    );

    let bru_slots = clean_layout
        .encode_slots::<F, _>(&input, &clean_values)
        .unwrap();
    let bru_input = ckks_encrypt(
        params,
        module,
        host_module,
        encoder,
        sk,
        params.k,
        &bru_slots.re,
        &bru_slots.im,
        &mut scratch.borrow(),
    );
    let plan = PackedCleaningPlan::compile(
        module,
        clean_layout,
        &input,
        &input,
        params.base2k.into(),
        CoeffsMeta::from_delta_budget(24, 4),
        TransformStrategy::Auto,
        &mut scratch.borrow(),
    )
    .unwrap();
    let mut keys = HashMap::new();
    for &galois_element in plan.galois_elements() {
        keys.insert(
            galois_element,
            gen_atk(
                params,
                module,
                galois_element,
                sk_raw,
                &mut scratch.borrow(),
            ),
        );
    }
    let mut workspace = plan.alloc_workspace(module, &bru_input);
    let mut cleaned = alloc_ct(params, module, params.k);
    module
        .ckks_packed_cleaning_into(
            &mut cleaned,
            &bru_input,
            &plan,
            &mut workspace,
            &keys,
            tensor_key,
            &mut scratch.borrow(),
        )
        .unwrap();
    assert_decrypt_precision_at_log_delta(
        "packed_general_cleaning",
        params,
        module,
        encoder,
        &cleaned,
        sk,
        &bru_slots.re,
        &bru_slots.im,
        8,
        &mut scratch.borrow(),
    );

    let split_layout = SplitLayout::new(params.n / 2).unwrap();
    let lhs_values = (0..split_layout.slots())
        .map(|value| value % 4)
        .collect::<Vec<_>>();
    let rhs_values = lhs_values
        .iter()
        .map(|value| (value + 1) % 4)
        .collect::<Vec<_>>();
    let lhs_slots = split_layout
        .encode_slots::<F, _>(&input, &lhs_values)
        .unwrap();
    let rhs_slots = split_layout
        .encode_slots::<F, _>(&input, &rhs_values)
        .unwrap();
    let mut lhs = Vec::new();
    let mut rhs = Vec::new();
    for coordinate in 0..3 {
        lhs.push(ckks_encrypt(
            params,
            module,
            host_module,
            encoder,
            sk,
            params.k,
            &lhs_slots.re[coordinate],
            &lhs_slots.im[coordinate],
            &mut scratch.borrow(),
        ));
        rhs.push(ckks_encrypt(
            params,
            module,
            host_module,
            encoder,
            sk,
            params.k,
            &rhs_slots.re[coordinate],
            &rhs_slots.im[coordinate],
            &mut scratch.borrow(),
        ));
    }
    let plan = SplitMultivariatePlan::compile(
        module,
        &tensor,
        params.base2k.into(),
        CoeffsMeta::from_delta_budget(24, 4),
        &mut scratch.borrow(),
    )
    .unwrap();
    let mut result = (0..plan.output_width())
        .map(|_| alloc_ct(params, module, params.k))
        .collect::<Vec<_>>();
    let operation_bytes =
        module.ckks_split_multivariate_tmp_bytes(&plan, &result[0], &lhs[0], tensor_key);
    let mut operation_scratch = ScratchOwned::<BE>::alloc(operation_bytes);
    module
        .ckks_split_multivariate_into(
            &mut result,
            &[&lhs, &rhs],
            &plan,
            tensor_key,
            &mut operation_scratch.borrow(),
        )
        .unwrap();
    let mapped = lhs_values
        .iter()
        .zip(&rhs_values)
        .map(|(&lhs, &rhs)| (lhs + 2 * rhs) % 4)
        .collect::<Vec<_>>();
    let want = split_layout.encode_slots::<F, _>(&output, &mapped).unwrap();
    for (coordinate, ciphertext) in result.iter().enumerate() {
        assert_decrypt_precision_at_log_delta(
            "split_multivariate",
            params,
            module,
            encoder,
            ciphertext,
            sk,
            &want.re[coordinate],
            &want.im[coordinate],
            18,
            &mut scratch.borrow(),
        );
    }

    let cleaning_plan = SplitCleaningPlan::compile(
        module,
        &input,
        &input,
        params.base2k.into(),
        CoeffsMeta::from_delta_budget(24, 4),
        &mut scratch.borrow(),
    )
    .unwrap();
    let mut cleaning_workspace = cleaning_plan.alloc_workspace(module, &lhs[0]);
    let mut cleaned = (0..cleaning_plan.output_width())
        .map(|_| alloc_ct(params, module, params.k))
        .collect::<Vec<_>>();
    let operation_bytes = module.ckks_split_cleaning_tmp_bytes(&lhs[0], &cleaning_plan, tensor_key);
    let mut operation_scratch = ScratchOwned::<BE>::alloc(operation_bytes);
    module
        .ckks_split_cleaning_into(
            &mut cleaned,
            &lhs,
            &cleaning_plan,
            &mut cleaning_workspace,
            tensor_key,
            &mut operation_scratch.borrow(),
        )
        .unwrap();
    for (coordinate, ciphertext) in cleaned.iter().enumerate() {
        assert_decrypt_precision_at_log_delta(
            "split_general_cleaning",
            params,
            module,
            encoder,
            ciphertext,
            sk,
            &lhs_slots.re[coordinate],
            &lhs_slots.im[coordinate],
            8,
            &mut scratch.borrow(),
        );
    }

    let bit = Bru::new(2).unwrap();
    let bit_output = Indicator::new(2, 0).unwrap();
    let table = (0usize..16)
        .map(|value| value.count_ones() as usize % 2)
        .collect::<Vec<_>>();
    let tensor =
        compile_multivariate_lut::<F, _>(&[&bit, &bit, &bit, &bit], &bit_output, &table).unwrap();
    let input_values = (0..4)
        .map(|variable| {
            (0..split_layout.slots())
                .map(|slot| (slot >> variable) & 1)
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let mut inputs = Vec::new();
    for values in &input_values {
        let slots = split_layout.encode_slots::<F, _>(&bit, values).unwrap();
        inputs.push(vec![ckks_encrypt(
            params,
            module,
            host_module,
            encoder,
            sk,
            params.k,
            &slots.re[0],
            &slots.im[0],
            &mut scratch.borrow(),
        )]);
    }
    let plan = SplitMultivariatePlan::compile(
        module,
        &tensor,
        params.base2k.into(),
        CoeffsMeta::from_delta_budget(8, 4),
        &mut scratch.borrow(),
    )
    .unwrap();
    let mut result = vec![alloc_ct(params, module, params.k)];
    let input_refs = inputs.iter().map(Vec::as_slice).collect::<Vec<_>>();
    let operation_bytes =
        module.ckks_split_multivariate_tmp_bytes(&plan, &result[0], &inputs[0][0], tensor_key);
    let mut operation_scratch = ScratchOwned::<BE>::alloc(operation_bytes);
    module
        .ckks_split_multivariate_into(
            &mut result,
            &input_refs,
            &plan,
            tensor_key,
            &mut operation_scratch.borrow(),
        )
        .unwrap();
    let mapped = (0..split_layout.slots())
        .map(|slot| {
            input_values
                .iter()
                .fold(0, |acc, values| acc ^ values[slot])
        })
        .collect::<Vec<_>>();
    let want = split_layout
        .encode_slots::<F, _>(&bit_output, &mapped)
        .unwrap();
    assert_decrypt_precision_at_log_delta(
        "split_four_variate",
        params,
        module,
        encoder,
        &result[0],
        sk,
        &want.re[0],
        &want.im[0],
        20,
        &mut scratch.borrow(),
    );
}
