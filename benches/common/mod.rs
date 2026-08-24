use std::{
    collections::{BTreeSet, HashMap},
    hint::black_box,
};

use criterion::{Criterion, Throughput};
use poulpy_blck_ecd::{
    algebra::{Bru, CleaningMode, Indicator, compile_lut, compile_multivariate_lut},
    layout::{PackedLayout, SplitLayout},
    transform::{
        CKKSBlockCleaningOps, CKKSBlockMulOps, CKKSMultivariateOps, CKKSPackedAffineOps,
        CKKSSplitAffineOps, PackedAffinePlan, PackedMultivariatePlan, SplitAffinePlan,
        SplitMultivariatePlan, TransformStrategy,
    },
};
use poulpy_ckks::{
    CoeffsMeta,
    api::{CKKSEncodingOps, CKKSLinearTransformationOps},
    test_suite::{
        NTT4X30_PARAMS_F64,
        helpers::{
            TestContextBackend, TestContextHostModule, TestContextModule, alloc_ct, alloc_scratch,
            ckks_encrypt, gen_atk, gen_sk_with_raw, gen_tsk,
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

pub fn bench_blocks<BE, E>(criterion: &mut Criterion, backend: &str)
where
    BE: TestContextBackend,
    Module<BE>: TestContextModule<BE>
        + CKKSEncodingOps<BE, f64>
        + CKKSLinearTransformationOps<BE>
        + CnvPVecAlloc<BE>
        + ModuleN,
    Module<HostBytesBackend>: TestContextHostModule,
    E: NegacyclicFFT<f64> + NegacyclicFFTNew<f64>,
    for<'a> BE::BufRef<'a>: HostDataRef,
    for<'a> BE::BufMut<'a>: HostDataMut,
{
    let params = NTT4X30_PARAMS_F64;
    let module = Module::<BE>::new(params.n as u64);
    let host_module = Module::<HostBytesBackend>::new(params.n as u64);
    let encoder = ReferenceEncoder::<E>::new::<f64>(params.n / 2).unwrap();
    let (sk_raw, sk) = gen_sk_with_raw(&params, &module, &host_module, [0x91; 32]);
    let mut scratch = alloc_scratch(&params, &module);
    let tensor_key = gen_tsk(&params, &module, &sk_raw, &mut scratch.borrow());

    let bru16 = Bru::new(16).unwrap();
    let indicator16 = Indicator::new(16, 0).unwrap();
    let unary_table = (0..16)
        .map(|value| (3 * value + 1) % 16)
        .collect::<Vec<_>>();
    let unary = compile_lut::<f64, _, _>(&bru16, &indicator16, &unary_table).unwrap();
    let packed_layout = PackedLayout::new(params.n / 2, 15).unwrap();
    let packed_values = (0..packed_layout.block_count())
        .map(|value| value % 16)
        .collect::<Vec<_>>();
    let packed_slots = packed_layout
        .encode_slots::<f64, _>(&bru16, &packed_values)
        .unwrap();
    let packed_input = ckks_encrypt(
        &params,
        &module,
        &host_module,
        &encoder,
        &sk,
        params.k,
        &packed_slots.re,
        &packed_slots.im,
        &mut scratch.borrow(),
    );
    let packed_plan = PackedAffinePlan::compile(
        &module,
        packed_layout,
        &unary,
        params.base2k.into(),
        CoeffsMeta::from_delta_budget(24, 4),
        TransformStrategy::Auto,
        &mut scratch.borrow(),
    )
    .unwrap();
    let packed_direct_plan = PackedAffinePlan::compile(
        &module,
        packed_layout,
        &unary,
        params.base2k.into(),
        CoeffsMeta::from_delta_budget(24, 4),
        TransformStrategy::Direct,
        &mut scratch.borrow(),
    )
    .unwrap();

    let split_layout = SplitLayout::new(params.n / 2).unwrap();
    let split_values = (0..split_layout.slots())
        .map(|value| value % 16)
        .collect::<Vec<_>>();
    let split_slots = split_layout
        .encode_slots::<f64, _>(&bru16, &split_values)
        .unwrap();
    let split_inputs = (0..15)
        .map(|coordinate| {
            ckks_encrypt(
                &params,
                &module,
                &host_module,
                &encoder,
                &sk,
                params.k,
                &split_slots.re[coordinate],
                &split_slots.im[coordinate],
                &mut scratch.borrow(),
            )
        })
        .collect::<Vec<_>>();
    let split_plan = SplitAffinePlan::compile(
        &module,
        &unary,
        params.base2k.into(),
        CoeffsMeta::from_delta_budget(24, 4),
        &mut scratch.borrow(),
    )
    .unwrap();

    let bru4 = Bru::new(4).unwrap();
    let table = (0..4)
        .flat_map(|lhs| (0..4).map(move |rhs| (lhs + 2 * rhs) % 4))
        .collect::<Vec<_>>();
    let tensor = compile_multivariate_lut::<f64, _>(&[&bru4, &bru4], &bru4, &table).unwrap();
    let tensor_layout = PackedLayout::new(params.n / 2, tensor.feature_width()).unwrap();
    let lhs_values = (0..tensor_layout.block_count())
        .map(|value| value % 4)
        .collect::<Vec<_>>();
    let rhs_values = lhs_values
        .iter()
        .map(|value| (value + 1) % 4)
        .collect::<Vec<_>>();
    let lhs_slots = tensor_layout
        .encode_slots::<f64, _>(&bru4, &lhs_values)
        .unwrap();
    let rhs_slots = tensor_layout
        .encode_slots::<f64, _>(&bru4, &rhs_values)
        .unwrap();
    let lhs = ckks_encrypt(
        &params,
        &module,
        &host_module,
        &encoder,
        &sk,
        params.k,
        &lhs_slots.re,
        &lhs_slots.im,
        &mut scratch.borrow(),
    );
    let rhs = ckks_encrypt(
        &params,
        &module,
        &host_module,
        &encoder,
        &sk,
        params.k,
        &rhs_slots.re,
        &rhs_slots.im,
        &mut scratch.borrow(),
    );
    let tensor_plan = PackedMultivariatePlan::compile(
        &module,
        tensor_layout,
        &tensor,
        params.base2k.into(),
        CoeffsMeta::from_delta_budget(24, 4),
        CoeffsMeta::from_delta_budget(24, 4),
        TransformStrategy::Auto,
        &mut scratch.borrow(),
    )
    .unwrap();

    let split_tensor_values = (0..split_layout.slots())
        .map(|value| value % 4)
        .collect::<Vec<_>>();
    let split_lhs_slots = split_layout
        .encode_slots::<f64, _>(&bru4, &split_tensor_values)
        .unwrap();
    let split_rhs_values = split_tensor_values
        .iter()
        .map(|value| (value + 1) % 4)
        .collect::<Vec<_>>();
    let split_rhs_slots = split_layout
        .encode_slots::<f64, _>(&bru4, &split_rhs_values)
        .unwrap();
    let split_lhs = (0..3)
        .map(|coordinate| {
            ckks_encrypt(
                &params,
                &module,
                &host_module,
                &encoder,
                &sk,
                params.k,
                &split_lhs_slots.re[coordinate],
                &split_lhs_slots.im[coordinate],
                &mut scratch.borrow(),
            )
        })
        .collect::<Vec<_>>();
    let split_rhs = (0..3)
        .map(|coordinate| {
            ckks_encrypt(
                &params,
                &module,
                &host_module,
                &encoder,
                &sk,
                params.k,
                &split_rhs_slots.re[coordinate],
                &split_rhs_slots.im[coordinate],
                &mut scratch.borrow(),
            )
        })
        .collect::<Vec<_>>();
    let split_tensor_plan = SplitMultivariatePlan::compile(
        &module,
        &tensor,
        params.base2k.into(),
        CoeffsMeta::from_delta_budget(24, 4),
        &mut scratch.borrow(),
    )
    .unwrap();

    let required_keys = packed_plan
        .galois_elements()
        .iter()
        .chain(packed_direct_plan.galois_elements())
        .chain(tensor_plan.galois_elements())
        .copied()
        .collect::<BTreeSet<_>>();
    let mut automorphism_keys = HashMap::new();
    for galois_element in required_keys {
        automorphism_keys.insert(
            galois_element,
            gen_atk(
                &params,
                &module,
                galois_element,
                &sk_raw,
                &mut scratch.borrow(),
            ),
        );
    }

    let mut packed_workspace = packed_plan.alloc_workspace(&module, &packed_input);
    let mut packed_direct_workspace = packed_direct_plan.alloc_workspace(&module, &packed_input);
    let mut packed_output = alloc_ct(&params, &module, params.k);
    let mut packed_direct_output = alloc_ct(&params, &module, params.k);
    let mut split_outputs = (0..15)
        .map(|_| alloc_ct(&params, &module, params.k))
        .collect::<Vec<_>>();
    let mut tensor_workspace = tensor_plan.alloc_workspace(&module, &lhs);
    let tensor_inputs = [lhs, rhs];
    let mut tensor_output = alloc_ct(&params, &module, params.k);
    let mut split_tensor_outputs = (0..split_tensor_plan.output_width())
        .map(|_| alloc_ct(&params, &module, params.k))
        .collect::<Vec<_>>();
    let split_tensor_inputs = [split_lhs.as_slice(), split_rhs.as_slice()];
    let split_tensor_tmp_bytes = module.ckks_split_multivariate_tmp_bytes(
        &split_tensor_plan,
        &split_tensor_outputs[0],
        &split_lhs[0],
        &tensor_key,
    );
    let mut split_tensor_scratch = ScratchOwned::<BE>::alloc(split_tensor_tmp_bytes);
    let mut product_output = alloc_ct(&params, &module, params.k);
    let clean_slots = packed_layout
        .encode_slots::<f64, _>(&indicator16, &packed_values)
        .unwrap();
    let clean_input = ckks_encrypt(
        &params,
        &module,
        &host_module,
        &encoder,
        &sk,
        params.k,
        &clean_slots.re,
        &clean_slots.im,
        &mut scratch.borrow(),
    );
    let mut clean_output = alloc_ct(&params, &module, params.k);

    let mut group = criterion.benchmark_group(format!("blocks/{backend}"));
    group.throughput(Throughput::Elements(packed_layout.block_count() as u64));
    group.bench_function("packed/unary_lut_t16_bsgs", |bench| {
        bench.iter(|| {
            module
                .ckks_packed_affine_into(
                    black_box(&mut packed_output),
                    black_box(&packed_input),
                    &packed_plan,
                    &mut packed_workspace,
                    &automorphism_keys,
                    &mut scratch.borrow(),
                )
                .unwrap()
        })
    });
    group.bench_function("packed/unary_lut_t16_direct", |bench| {
        bench.iter(|| {
            module
                .ckks_packed_affine_into(
                    black_box(&mut packed_direct_output),
                    black_box(&packed_input),
                    &packed_direct_plan,
                    &mut packed_direct_workspace,
                    &automorphism_keys,
                    &mut scratch.borrow(),
                )
                .unwrap()
        })
    });
    group.throughput(Throughput::Elements(split_layout.slots() as u64));
    group.bench_function("split/unary_lut_t16", |bench| {
        bench.iter(|| {
            module
                .ckks_split_affine_into(
                    black_box(&mut split_outputs),
                    black_box(&split_inputs),
                    &split_plan,
                    &mut scratch.borrow(),
                )
                .unwrap()
        })
    });
    group.throughput(Throughput::Elements(tensor_layout.block_count() as u64));
    group.bench_function("packed/bivariate_lut_t4", |bench| {
        bench.iter(|| {
            module
                .ckks_packed_multivariate_into(
                    black_box(&mut tensor_output),
                    black_box(&tensor_inputs),
                    &tensor_plan,
                    &mut tensor_workspace,
                    &automorphism_keys,
                    &tensor_key,
                    &mut scratch.borrow(),
                )
                .unwrap()
        })
    });
    group.throughput(Throughput::Elements(split_layout.slots() as u64));
    group.bench_function("split/bivariate_lut_t4", |bench| {
        bench.iter(|| {
            module
                .ckks_split_multivariate_into(
                    black_box(&mut split_tensor_outputs),
                    black_box(&split_tensor_inputs),
                    &split_tensor_plan,
                    &tensor_key,
                    &mut split_tensor_scratch.borrow(),
                )
                .unwrap()
        })
    });
    group.throughput(Throughput::Elements(packed_layout.block_count() as u64));
    group.bench_function("packed/native_product_t16", |bench| {
        bench.iter(|| {
            module
                .ckks_packed_native_mul_into(
                    black_box(&mut product_output),
                    black_box(&packed_input),
                    black_box(&packed_input),
                    &tensor_key,
                    &mut scratch.borrow(),
                )
                .unwrap()
        })
    });
    group.bench_function("packed/direct_clean_t16", |bench| {
        bench.iter(|| {
            module
                .ckks_clean_into(
                    black_box(&mut clean_output),
                    black_box(&clean_input),
                    CleaningMode::Binary,
                    &tensor_key,
                    &mut scratch.borrow(),
                )
                .unwrap()
        })
    });
    group.finish();
}
