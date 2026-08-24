mod common;

use criterion::{Criterion, criterion_group, criterion_main};
use poulpy_cpu_avx::{FFT64AvxReimTable, NTT4x30Avx};

fn blocks(c: &mut Criterion) {
    common::bench_blocks::<NTT4x30Avx, FFT64AvxReimTable>(c, "avx");
}

criterion_group!(benches, blocks);
criterion_main!(benches);
