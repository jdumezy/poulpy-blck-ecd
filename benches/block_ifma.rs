mod common;

use criterion::{Criterion, criterion_group, criterion_main};
use poulpy_cpu_avx512::{FFT64Avx512ReimTable, NTT3x42Ifma};

fn blocks(c: &mut Criterion) {
    common::bench_blocks::<NTT3x42Ifma, FFT64Avx512ReimTable>(c, "ifma");
}

criterion_group!(benches, blocks);
criterion_main!(benches);
