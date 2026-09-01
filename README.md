# 🐙 poulpy-blck-ecd

`poulpy-blck-ecd` implements block encodings for finite alphabets in discrete CKKS from [the original paper](https://eprint.iacr.org/2026/1200) over [`poulpy-ckks`](https://github.com/poulpy-fhe/poulpy).
It compiles univariate and multivariate lookup tables into reusable affine and nonlinear circuits, with packed and split ciphertext layouts for the Poulpy backends.

## Requirements

`poulpy-blck-ecd` currently requires nightly Rust because Poulpy 0.8.2 uses unstable Rust features.
This repository pins `nightly-2026-05-14`; applications using the crate must select that toolchain, or a compatible newer nightly, in their own workspace.

```sh
rustup toolchain install nightly-2026-05-14
cargo +nightly-2026-05-14 build
```

## API

- Character encodings: `Bru`, `Lbru`, and `WalshHadamard`.
- Indicator and order encodings: `Indicator`, `Thermometer`, `MeetZeta`, and `JoinZeta`, with finite-poset helpers.
- LUT compilation: `compile_lut`, `compile_multivariate_lut`, `AffineMap`, and `TensorMap`.
- Ciphertext layouts: `PackedLayout` stores several encoded blocks in one ciphertext, while `SplitLayout` stores each block coordinate in a separate ciphertext.
- Circuit plans: packed and split affine transforms, multivariate transforms, cleaning circuits, and native block operations.

Common imports are available through `poulpy_blck_ecd::prelude`.

## Use

Choose an encoding for the input and output alphabets, compile the lookup table before encryption, and select a layout for the encoded blocks.

```rust
use poulpy_blck_ecd::prelude::*;

fn main() -> anyhow::Result<()> {
    let input = Bru::new(4)?;
    let output = Indicator::new(4, 0)?;
    let table = [1, 2, 3, 0];
    let lut = compile_lut::<f64, _, _>(&input, &output, &table)?;

    let layout = PackedLayout::for_widths(4096, lut.cols(), lut.rows())?;
    let values = [0, 1, 2, 3];
    let slots = layout.encode_slots::<f64, _>(&input, &values)?;

    assert_eq!(layout.decode_slots(&input, &slots, values.len())?, values);
    Ok(())
}
```

Compile a `PackedAffinePlan` or `SplitAffinePlan` from the resulting map, transfer its prepared values to the selected backend, allocate the reported temporary memory, and evaluate it on compatible ciphertexts.
Multivariate tables follow the same pattern through `compile_multivariate_lut`, `PackedMultivariatePlan`, and `SplitMultivariatePlan`.
Plans can be reused for ciphertexts with compatible CKKS parameters and layouts.

`poulpy-blck-ecd` does not select CKKS security parameters or insert bootstrapping automatically.
Applications must choose parameters for their security target, reserve enough modulus for each planned circuit, generate the required evaluation keys, and bootstrap explicitly when required.

## Backend features

| Feature | Target | Required CPU support |
| --- | --- | --- |
| `ref` | Any supported Poulpy target | None beyond the target baseline |
| `avx` | `x86_64` | AVX2 and FMA |
| `avx512` | `x86_64` | AVX-512F |
| `ifma` | `x86_64` | AVX-512F, AVX-512 IFMA, and AVX-512VL |
| `neon` | `aarch64` | NEON |
| `all-x86-backends` | `x86_64` | All x86 requirements above |

The accelerated features activate Poulpy's corresponding `enable-*` feature gates, which reject incompatible architectures or missing compile-time target features.
Build x86 accelerated backends with suitable `RUSTFLAGS`, such as `-C target-cpu=native` on a compatible machine.
Do not use Cargo's `--all-features` across architectures: it activates both `neon` and the x86-only backends; select `neon` or `all-x86-backends` for the target instead.

## Test

Run the backend-independent algebra tests with:

```sh
cargo test
```

On a compatible x86 host, run the backend-generic circuit suite with:

```sh
RUSTFLAGS='-C target-cpu=native' cargo test --release --features all-x86-backends
```

The suite checks encodings, layouts, transform strategies, multivariate lookup tables, cleaning, and native block operations on the reference, AVX, AVX-512, and IFMA backends.

## Benchmark

The same block-encoding workloads are available for every x86 backend.
Run the target matching the benchmark host.

```sh
RUSTFLAGS='-C target-cpu=native' cargo bench --features ref --bench block_ref
RUSTFLAGS='-C target-cpu=native' cargo bench --features avx --bench block_avx
RUSTFLAGS='-C target-cpu=native' cargo bench --features avx512 --bench block_avx512
RUSTFLAGS='-C target-cpu=native' cargo bench --features ifma --bench block_ifma
```

## Security

This is research software and has not undergone an independent security audit.
Treat its parameter selection, side-channel behavior, and integration into a complete cryptosystem as application responsibilities rather than production-hardening guarantees.

## Citation

```bibtex
@misc{block-encoding,
      author = {Jules Dumezy and Elias Suvanto},
      title = {Character Block Encodings for Discrete {CKKS}: Single-Level {LUTs} and Low-Depth Arithmetic},
      howpublished = {Cryptology {ePrint} Archive, Paper 2026/1200},
      year = {2026},
      url = {https://eprint.iacr.org/2026/1200}
}
```
