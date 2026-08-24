mod common;

use criterion::{Criterion, criterion_group, criterion_main};
use poulpy_cpu_ref::{FFT64ReimTable, NTT4x30Ref};

fn blocks(c: &mut Criterion) {
    common::bench_blocks::<NTT4x30Ref, FFT64ReimTable<f64>>(c, "ref");
}

criterion_group!(benches, blocks);
criterion_main!(benches);
