#[cfg(any(feature = "ref", feature = "avx", feature = "avx512", feature = "ifma"))]
use crate::ckks_block_backend_test_suite;

#[cfg(feature = "ref")]
ckks_block_backend_test_suite!(
    mod reference,
    backend = poulpy_cpu_ref::NTT4x30Ref,
    scalar = f64,
    encoder = poulpy_cpu_ref::FFT64ReimTable<f64>,
    params = poulpy_ckks::test_suite::NTT4X30_PARAMS_F64,
);

#[cfg(feature = "avx")]
ckks_block_backend_test_suite!(
    mod avx,
    backend = poulpy_cpu_avx::NTT4x30Avx,
    scalar = f64,
    encoder = poulpy_cpu_avx::FFT64AvxReimTable,
    params = poulpy_ckks::test_suite::NTT4X30_PARAMS_F64,
);

#[cfg(feature = "avx512")]
ckks_block_backend_test_suite!(
    mod avx512,
    backend = poulpy_cpu_avx512::NTT4x30Avx512,
    scalar = f64,
    encoder = poulpy_cpu_avx512::FFT64Avx512ReimTable,
    params = poulpy_ckks::test_suite::NTT4X30_PARAMS_F64,
);

#[cfg(feature = "ifma")]
ckks_block_backend_test_suite!(
    mod ifma,
    backend = poulpy_cpu_avx512::NTT3x42Ifma,
    scalar = f64,
    encoder = poulpy_cpu_avx512::FFT64Avx512ReimTable,
    params = poulpy_ckks::test_suite::NTT4X30_PARAMS_F64,
);
