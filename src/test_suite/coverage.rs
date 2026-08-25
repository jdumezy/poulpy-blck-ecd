use std::collections::{BTreeSet, HashMap};

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
        BlockEncoding, Bru, CleaningMode, Indicator, JoinZeta, Lbru, MeetZeta, WalshHadamard,
        compile_lut, compile_multivariate_lut,
    },
    layout::{PackedLayout, SplitLayout},
    scalar::BlockScalar,
    transform::{
        CKKSBlockCleaningOps, CKKSBlockMulOps, CKKSMultivariateOps, CKKSPackedAffineOps,
        PackedAffinePlan, PackedMultivariatePlan, SplitMultivariatePlan, TransformStrategy,
    },
};

pub fn test_characters<BE, F, E>(
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
    let encoder = ReferenceEncoder::<E>::new::<F>(params.n / 2).unwrap();
    let (sk_raw, sk) = gen_sk_with_raw(&params, module, host_module, [0x31; 32]);
    let mut scratch = alloc_scratch(&params, module);
    let tensor_key = gen_tsk(&params, module, &sk_raw, &mut scratch.borrow());

    let lbru = Lbru::new(5).unwrap();
    let lbru_width = <Lbru as BlockEncoding<F>>::block_size(&lbru);
    let layout = PackedLayout::new(params.n / 2, lbru_width).unwrap();
    let lhs_values = (0..layout.block_count())
        .map(|value| value % 5)
        .collect::<Vec<_>>();
    let rhs_values = lhs_values
        .iter()
        .map(|value| (2 * value + 1) % 5)
        .collect::<Vec<_>>();
    let lhs_slots = layout.encode_slots::<F, _>(&lbru, &lhs_values).unwrap();
    let rhs_slots = layout.encode_slots::<F, _>(&lbru, &rhs_values).unwrap();
    let lhs = ckks_encrypt(
        &params,
        module,
        host_module,
        &encoder,
        &sk,
        params.k,
        &lhs_slots.re,
        &lhs_slots.im,
        &mut scratch.borrow(),
    );
    let rhs = ckks_encrypt(
        &params,
        module,
        host_module,
        &encoder,
        &sk,
        params.k,
        &rhs_slots.re,
        &rhs_slots.im,
        &mut scratch.borrow(),
    );
    let mut product = alloc_ct(&params, module, params.k);
    module
        .ckks_packed_native_mul_into(&mut product, &lhs, &rhs, &tensor_key, &mut scratch.borrow())
        .unwrap();
    let product_values = lhs_values
        .iter()
        .zip(&rhs_values)
        .map(|(&lhs, &rhs)| lhs * rhs % 5)
        .collect::<Vec<_>>();
    let want = layout.encode_slots::<F, _>(&lbru, &product_values).unwrap();
    assert_decrypt_precision_at_log_delta(
        "lbru_native_product",
        &params,
        module,
        &encoder,
        &product,
        &sk,
        &want.re,
        &want.im,
        30,
        &mut scratch.borrow(),
    );

    let split_layout = SplitLayout::new(params.n / 2).unwrap();
    let split_values = (0..split_layout.slots())
        .map(|value| value % 5)
        .collect::<Vec<_>>();
    let split_slots = split_layout
        .encode_slots::<F, _>(&lbru, &split_values)
        .unwrap();
    let split_inputs = (0..lbru_width)
        .map(|coordinate| {
            ckks_encrypt(
                &params,
                module,
                host_module,
                &encoder,
                &sk,
                params.k,
                &split_slots.re[coordinate],
                &split_slots.im[coordinate],
                &mut scratch.borrow(),
            )
        })
        .collect::<Vec<_>>();
    let conjugation_key = gen_atk(
        &params,
        module,
        (2 * module.n() - 1) as i64,
        &sk_raw,
        &mut scratch.borrow(),
    );
    let mut conjugated = (0..lbru_width)
        .map(|_| alloc_ct(&params, module, params.k))
        .collect::<Vec<_>>();
    module
        .ckks_split_block_conjugate_into(
            &mut conjugated,
            &split_inputs,
            &conjugation_key,
            &mut scratch.borrow(),
        )
        .unwrap();
    let inverses = split_values
        .iter()
        .map(|&value| {
            if value == 0 {
                0
            } else {
                (1..5).find(|candidate| candidate * value % 5 == 1).unwrap()
            }
        })
        .collect::<Vec<_>>();
    let want = split_layout.encode_slots::<F, _>(&lbru, &inverses).unwrap();
    for (coordinate, ciphertext) in conjugated.iter().enumerate() {
        assert_decrypt_precision_at_log_delta(
            "lbru_split_conjugate",
            &params,
            module,
            &encoder,
            ciphertext,
            &sk,
            &want.re[coordinate],
            &want.im[coordinate],
            35,
            &mut scratch.borrow(),
        );
    }

    let walsh = WalshHadamard::new(2).unwrap();
    let walsh_width = <WalshHadamard as BlockEncoding<F>>::block_size(&walsh);
    let layout = PackedLayout::new(params.n / 2, walsh_width).unwrap();
    let lhs_values = (0..layout.block_count())
        .map(|value| value % 4)
        .collect::<Vec<_>>();
    let rhs_values = lhs_values
        .iter()
        .map(|value| (value + 1) % 4)
        .collect::<Vec<_>>();
    let lhs_slots = layout.encode_slots::<F, _>(&walsh, &lhs_values).unwrap();
    let rhs_slots = layout.encode_slots::<F, _>(&walsh, &rhs_values).unwrap();
    let lhs = ckks_encrypt(
        &params,
        module,
        host_module,
        &encoder,
        &sk,
        params.k,
        &lhs_slots.re,
        &lhs_slots.im,
        &mut scratch.borrow(),
    );
    let rhs = ckks_encrypt(
        &params,
        module,
        host_module,
        &encoder,
        &sk,
        params.k,
        &rhs_slots.re,
        &rhs_slots.im,
        &mut scratch.borrow(),
    );
    let mut product = alloc_ct(&params, module, params.k);
    module
        .ckks_packed_native_mul_into(&mut product, &lhs, &rhs, &tensor_key, &mut scratch.borrow())
        .unwrap();
    let xor_values = lhs_values
        .iter()
        .zip(&rhs_values)
        .map(|(&lhs, &rhs)| lhs ^ rhs)
        .collect::<Vec<_>>();
    let want = layout.encode_slots::<F, _>(&walsh, &xor_values).unwrap();
    assert_decrypt_precision_at_log_delta(
        "walsh_native_xor",
        &params,
        module,
        &encoder,
        &product,
        &sk,
        &want.re,
        &want.im,
        30,
        &mut scratch.borrow(),
    );

    let mut noisy = lhs_slots.clone();
    let noise = F::from_f64(0.02).unwrap();
    for block in 0..layout.block_count() {
        for coordinate in 0..walsh_width {
            let slot = layout.slot(block, coordinate);
            noisy.re[slot] = if noisy.re[slot] > F::zero() {
                noisy.re[slot] - noise
            } else {
                noisy.re[slot] + noise
            };
        }
    }
    let noisy = ckks_encrypt(
        &params,
        module,
        host_module,
        &encoder,
        &sk,
        params.k,
        &noisy.re,
        &noisy.im,
        &mut scratch.borrow(),
    );
    let mut cleaned = alloc_ct(&params, module, params.k);
    module
        .ckks_clean_into(
            &mut cleaned,
            &noisy,
            CleaningMode::Sign,
            &tensor_key,
            &mut scratch.borrow(),
        )
        .unwrap();
    assert_decrypt_precision_at_log_delta(
        "walsh_sign_cleaning",
        &params,
        module,
        &encoder,
        &cleaned,
        &sk,
        &lhs_slots.re,
        &lhs_slots.im,
        18,
        &mut scratch.borrow(),
    );
}

pub fn test_zeta_and_indicator<BE, F, E>(
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
    let encoder = ReferenceEncoder::<E>::new::<F>(params.n / 2).unwrap();
    let (sk_raw, sk) = gen_sk_with_raw(&params, module, host_module, [0x42; 32]);
    let mut scratch = alloc_scratch(&params, module);
    let tensor_key = gen_tsk(&params, module, &sk_raw, &mut scratch.borrow());
    let meet = MeetZeta::boolean(2).unwrap();
    let zeta_width = <MeetZeta as BlockEncoding<F>>::block_size(&meet);
    let layout = PackedLayout::new(params.n / 2, zeta_width).unwrap();
    let lhs_values = (0..layout.block_count())
        .map(|value| value % 4)
        .collect::<Vec<_>>();
    let rhs_values = lhs_values
        .iter()
        .map(|value| (3 * value + 1) % 4)
        .collect::<Vec<_>>();
    let lhs_slots = layout.encode_slots::<F, _>(&meet, &lhs_values).unwrap();
    let rhs_slots = layout.encode_slots::<F, _>(&meet, &rhs_values).unwrap();
    let lhs = ckks_encrypt(
        &params,
        module,
        host_module,
        &encoder,
        &sk,
        params.k,
        &lhs_slots.re,
        &lhs_slots.im,
        &mut scratch.borrow(),
    );
    let rhs = ckks_encrypt(
        &params,
        module,
        host_module,
        &encoder,
        &sk,
        params.k,
        &rhs_slots.re,
        &rhs_slots.im,
        &mut scratch.borrow(),
    );
    let mut product = alloc_ct(&params, module, params.k);
    module
        .ckks_packed_native_mul_into(&mut product, &lhs, &rhs, &tensor_key, &mut scratch.borrow())
        .unwrap();
    let meet_values = lhs_values
        .iter()
        .zip(&rhs_values)
        .map(|(&lhs, &rhs)| lhs & rhs)
        .collect::<Vec<_>>();
    let want = layout.encode_slots::<F, _>(&meet, &meet_values).unwrap();
    assert_decrypt_precision_at_log_delta(
        "meet_zeta_native",
        &params,
        module,
        &encoder,
        &product,
        &sk,
        &want.re,
        &want.im,
        30,
        &mut scratch.borrow(),
    );

    let indicator = Indicator::new(4, 0).unwrap();
    let lhs_slots = layout
        .encode_slots::<F, _>(&indicator, &lhs_values)
        .unwrap();
    let rhs_slots = layout
        .encode_slots::<F, _>(&indicator, &rhs_values)
        .unwrap();
    let lhs = ckks_encrypt(
        &params,
        module,
        host_module,
        &encoder,
        &sk,
        params.k,
        &lhs_slots.re,
        &lhs_slots.im,
        &mut scratch.borrow(),
    );
    let rhs = ckks_encrypt(
        &params,
        module,
        host_module,
        &encoder,
        &sk,
        params.k,
        &rhs_slots.re,
        &rhs_slots.im,
        &mut scratch.borrow(),
    );
    let mut equality = alloc_ct(&params, module, params.k);
    module
        .ckks_packed_equality_gate_into(
            &mut equality,
            &lhs,
            &rhs,
            &tensor_key,
            &mut scratch.borrow(),
        )
        .unwrap();
    let equality_values = lhs_values
        .iter()
        .zip(&rhs_values)
        .map(|(&lhs, &rhs)| if lhs == rhs { lhs } else { 0 })
        .collect::<Vec<_>>();
    let want = layout
        .encode_slots::<F, _>(&indicator, &equality_values)
        .unwrap();
    assert_decrypt_precision_at_log_delta(
        "indicator_equality_gate",
        &params,
        module,
        &encoder,
        &equality,
        &sk,
        &want.re,
        &want.im,
        30,
        &mut scratch.borrow(),
    );

    let join = JoinZeta::boolean(2).unwrap();
    let join_width = <JoinZeta as BlockEncoding<F>>::block_size(&join);
    let split_layout = SplitLayout::new(params.n / 2).unwrap();
    let lhs_values = (0..split_layout.slots())
        .map(|value| value % 4)
        .collect::<Vec<_>>();
    let rhs_values = lhs_values
        .iter()
        .map(|value| (value + 2) % 4)
        .collect::<Vec<_>>();
    let lhs_slots = split_layout
        .encode_slots::<F, _>(&join, &lhs_values)
        .unwrap();
    let rhs_slots = split_layout
        .encode_slots::<F, _>(&join, &rhs_values)
        .unwrap();
    let lhs = (0..join_width)
        .map(|coordinate| {
            ckks_encrypt(
                &params,
                module,
                host_module,
                &encoder,
                &sk,
                params.k,
                &lhs_slots.re[coordinate],
                &lhs_slots.im[coordinate],
                &mut scratch.borrow(),
            )
        })
        .collect::<Vec<_>>();
    let rhs = (0..join_width)
        .map(|coordinate| {
            ckks_encrypt(
                &params,
                module,
                host_module,
                &encoder,
                &sk,
                params.k,
                &rhs_slots.re[coordinate],
                &rhs_slots.im[coordinate],
                &mut scratch.borrow(),
            )
        })
        .collect::<Vec<_>>();
    let mut product = (0..join_width)
        .map(|_| alloc_ct(&params, module, params.k))
        .collect::<Vec<_>>();
    module
        .ckks_split_native_mul_into(&mut product, &lhs, &rhs, &tensor_key, &mut scratch.borrow())
        .unwrap();
    let join_values = lhs_values
        .iter()
        .zip(&rhs_values)
        .map(|(&lhs, &rhs)| lhs | rhs)
        .collect::<Vec<_>>();
    let want = split_layout
        .encode_slots::<F, _>(&join, &join_values)
        .unwrap();
    for (coordinate, ciphertext) in product.iter().enumerate() {
        assert_decrypt_precision_at_log_delta(
            "join_zeta_native",
            &params,
            module,
            &encoder,
            ciphertext,
            &sk,
            &want.re[coordinate],
            &want.im[coordinate],
            30,
            &mut scratch.borrow(),
        );
    }
}

pub fn test_transform_strategies<BE, F, E>(
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
    let encoder = ReferenceEncoder::<E>::new::<F>(params.n / 2).unwrap();
    let (sk_raw, sk) = gen_sk_with_raw(&params, module, host_module, [0x53; 32]);
    let mut scratch = alloc_scratch(&params, module);
    let input = Bru::new(5).unwrap();
    let output = Indicator::new(5, 0).unwrap();
    let table = [4, 2, 0, 3, 1];
    let map = compile_lut::<F, _, _>(&input, &output, &table).unwrap();
    let layout = PackedLayout::new(params.n / 2, 5).unwrap();
    let values = (0..layout.block_count())
        .map(|value| value % 5)
        .collect::<Vec<_>>();
    let slots = layout.encode_slots::<F, _>(&input, &values).unwrap();
    let ciphertext = ckks_encrypt(
        &params,
        module,
        host_module,
        &encoder,
        &sk,
        params.k,
        &slots.re,
        &slots.im,
        &mut scratch.borrow(),
    );
    let direct = PackedAffinePlan::compile(
        module,
        layout,
        &map,
        params.base2k.into(),
        CoeffsMeta::from_delta_budget(24, 4),
        TransformStrategy::Direct,
        &mut scratch.borrow(),
    )
    .unwrap();
    let bsgs = PackedAffinePlan::compile(
        module,
        layout,
        &map,
        params.base2k.into(),
        CoeffsMeta::from_delta_budget(24, 4),
        TransformStrategy::Bsgs { giant_step: 2 },
        &mut scratch.borrow(),
    )
    .unwrap();
    let required_keys = direct
        .galois_elements()
        .iter()
        .chain(bsgs.galois_elements())
        .copied()
        .collect::<BTreeSet<_>>();
    let mut keys = HashMap::new();
    for galois_element in required_keys {
        keys.insert(
            galois_element,
            gen_atk(
                &params,
                module,
                galois_element,
                &sk_raw,
                &mut scratch.borrow(),
            ),
        );
    }
    let mapped = values.iter().map(|&value| table[value]).collect::<Vec<_>>();
    let want = layout.encode_slots::<F, _>(&output, &mapped).unwrap();
    for (name, plan) in [("packed_direct", &direct), ("packed_bsgs", &bsgs)] {
        let mut workspace = plan.alloc_workspace(module, &ciphertext);
        let mut result = alloc_ct(&params, module, params.k);
        module
            .ckks_packed_affine_into(
                &mut result,
                &ciphertext,
                plan,
                &mut workspace,
                &keys,
                &mut scratch.borrow(),
            )
            .unwrap();
        assert_decrypt_precision_at_log_delta(
            name,
            &params,
            module,
            &encoder,
            &result,
            &sk,
            &want.re,
            &want.im,
            20,
            &mut scratch.borrow(),
        );
    }

    let identity_map = compile_lut::<F, _, _>(&input, &input, &[0, 1, 2, 3, 4]).unwrap();
    let identity = PackedAffinePlan::compile(
        module,
        layout,
        &identity_map,
        params.base2k.into(),
        CoeffsMeta::from_delta_budget(24, 4),
        TransformStrategy::Auto,
        &mut scratch.borrow(),
    )
    .unwrap();
    assert!(identity.galois_elements().is_empty());
    let mut workspace = identity.alloc_workspace(module, &ciphertext);
    let mut result = alloc_ct(&params, module, params.k);
    module
        .ckks_packed_affine_into(
            &mut result,
            &ciphertext,
            &identity,
            &mut workspace,
            &keys,
            &mut scratch.borrow(),
        )
        .unwrap();
    assert_decrypt_precision_at_log_delta(
        "packed_identity",
        &params,
        module,
        &encoder,
        &result,
        &sk,
        &slots.re,
        &slots.im,
        35,
        &mut scratch.borrow(),
    );
}

pub fn test_asymmetric_multivariate<BE, F, E>(
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
    let encoder = ReferenceEncoder::<E>::new::<F>(params.n / 2).unwrap();
    let (sk_raw, sk) = gen_sk_with_raw(&params, module, host_module, [0x64; 32]);
    let mut scratch = alloc_scratch(&params, module);
    let tensor_key = gen_tsk(&params, module, &sk_raw, &mut scratch.borrow());
    let bit = Bru::new(2).unwrap();
    let trit = Bru::new(3).unwrap();
    let output = Indicator::new(3, 0).unwrap();
    let table = (0..2)
        .flat_map(|bit| (0..3).map(move |trit| (bit + 2 * trit) % 3))
        .collect::<Vec<_>>();
    let tensor = compile_multivariate_lut::<F, _>(&[&bit, &trit], &output, &table).unwrap();

    let layout = PackedLayout::new(params.n / 2, tensor.feature_width()).unwrap();
    let bit_values = (0..layout.block_count())
        .map(|value| value % 2)
        .collect::<Vec<_>>();
    let trit_values = (0..layout.block_count())
        .map(|value| (2 * value + 1) % 3)
        .collect::<Vec<_>>();
    let bit_slots = layout.encode_slots::<F, _>(&bit, &bit_values).unwrap();
    let trit_slots = layout.encode_slots::<F, _>(&trit, &trit_values).unwrap();
    let bit_ct = ckks_encrypt(
        &params,
        module,
        host_module,
        &encoder,
        &sk,
        params.k,
        &bit_slots.re,
        &bit_slots.im,
        &mut scratch.borrow(),
    );
    let trit_ct = ckks_encrypt(
        &params,
        module,
        host_module,
        &encoder,
        &sk,
        params.k,
        &trit_slots.re,
        &trit_slots.im,
        &mut scratch.borrow(),
    );
    let plan = PackedMultivariatePlan::compile(
        module,
        layout,
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
                &params,
                module,
                galois_element,
                &sk_raw,
                &mut scratch.borrow(),
            ),
        );
    }
    let mut workspace = plan.alloc_workspace(module, &bit_ct);
    let mut result = alloc_ct(&params, module, params.k);
    module
        .ckks_packed_multivariate_into(
            &mut result,
            &[bit_ct, trit_ct],
            &plan,
            &mut workspace,
            &keys,
            &tensor_key,
            &mut scratch.borrow(),
        )
        .unwrap();
    let mapped = bit_values
        .iter()
        .zip(&trit_values)
        .map(|(&bit, &trit)| (bit + 2 * trit) % 3)
        .collect::<Vec<_>>();
    let want = layout.encode_slots::<F, _>(&output, &mapped).unwrap();
    assert_decrypt_precision_at_log_delta(
        "packed_asymmetric_multivariate",
        &params,
        module,
        &encoder,
        &result,
        &sk,
        &want.re,
        &want.im,
        12,
        &mut scratch.borrow(),
    );

    let split_layout = SplitLayout::new(params.n / 2).unwrap();
    let bit_values = (0..split_layout.slots())
        .map(|value| value % 2)
        .collect::<Vec<_>>();
    let trit_values = (0..split_layout.slots())
        .map(|value| (value + 1) % 3)
        .collect::<Vec<_>>();
    let bit_slots = split_layout
        .encode_slots::<F, _>(&bit, &bit_values)
        .unwrap();
    let trit_slots = split_layout
        .encode_slots::<F, _>(&trit, &trit_values)
        .unwrap();
    let bit_ct = vec![ckks_encrypt(
        &params,
        module,
        host_module,
        &encoder,
        &sk,
        params.k,
        &bit_slots.re[0],
        &bit_slots.im[0],
        &mut scratch.borrow(),
    )];
    let trit_width = <Bru as BlockEncoding<F>>::block_size(&trit);
    let trit_ct = (0..trit_width)
        .map(|coordinate| {
            ckks_encrypt(
                &params,
                module,
                host_module,
                &encoder,
                &sk,
                params.k,
                &trit_slots.re[coordinate],
                &trit_slots.im[coordinate],
                &mut scratch.borrow(),
            )
        })
        .collect::<Vec<_>>();
    let plan = SplitMultivariatePlan::compile(
        module,
        &tensor,
        params.base2k.into(),
        CoeffsMeta::from_delta_budget(24, 4),
        &mut scratch.borrow(),
    )
    .unwrap();
    let mut result = (0..plan.output_width())
        .map(|_| alloc_ct(&params, module, params.k))
        .collect::<Vec<_>>();
    let operation_bytes =
        module.ckks_split_multivariate_tmp_bytes(&plan, &result[0], &bit_ct[0], &tensor_key);
    let mut operation_scratch = ScratchOwned::<BE>::alloc(operation_bytes);
    module
        .ckks_split_multivariate_into(
            &mut result,
            &[&bit_ct, &trit_ct],
            &plan,
            &tensor_key,
            &mut operation_scratch.borrow(),
        )
        .unwrap();
    let mapped = bit_values
        .iter()
        .zip(&trit_values)
        .map(|(&bit, &trit)| (bit + 2 * trit) % 3)
        .collect::<Vec<_>>();
    let want = split_layout.encode_slots::<F, _>(&output, &mapped).unwrap();
    for (coordinate, ciphertext) in result.iter().enumerate() {
        assert_decrypt_precision_at_log_delta(
            "split_asymmetric_multivariate",
            &params,
            module,
            &encoder,
            ciphertext,
            &sk,
            &want.re[coordinate],
            &want.im[coordinate],
            18,
            &mut scratch.borrow(),
        );
    }
}
