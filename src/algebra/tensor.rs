use anyhow::{Result, ensure};

use super::{AffineMap, BlockEncoding, Coefficient};
use crate::scalar::BlockScalar;

#[derive(Clone, Debug, PartialEq)]
pub struct TensorMap<F> {
    pub input_sizes: Vec<usize>,
    pub rows: usize,
    pub matrix: Vec<Coefficient<F>>,
    pub bias: Vec<Coefficient<F>>,
}

impl<F: BlockScalar> TensorMap<F> {
    pub fn feature_width(&self) -> usize {
        self.input_sizes.iter().product::<usize>() - 1
    }

    pub fn input_widths(&self) -> impl ExactSizeIterator<Item = usize> + '_ {
        self.input_sizes.iter().map(|size| size - 1)
    }

    pub fn as_affine(&self) -> AffineMap<F> {
        AffineMap {
            rows: self.rows,
            cols: self.feature_width(),
            matrix: self.matrix.clone(),
            bias: self.bias.clone(),
        }
    }
}

#[allow(clippy::needless_range_loop)]
pub fn compile_multivariate_coordinates<F: BlockScalar>(
    inputs: &[&dyn BlockEncoding<F>],
    output_values: &[Vec<Coefficient<F>>],
) -> Result<TensorMap<F>> {
    ensure!(
        inputs.len() >= 2,
        "a multivariate LUT needs at least two inputs"
    );
    let input_sizes = inputs
        .iter()
        .map(|input| input.alphabet_size())
        .collect::<Vec<_>>();
    let table_len = checked_product(&input_sizes)?;
    ensure!(
        output_values.len() == table_len,
        "coordinate table has {} rows, expected {table_len}",
        output_values.len()
    );
    let rows = output_values.first().map_or(0, Vec::len);
    ensure!(rows != 0, "coordinate table must have at least one output");
    ensure!(
        output_values.iter().all(|row| row.len() == rows),
        "coordinate table rows have different widths"
    );

    let mut transformed = output_values.to_vec();
    for (axis, input) in inputs.iter().enumerate() {
        let size = input_sizes[axis];
        let stride = checked_product(&input_sizes[axis + 1..])?;
        let outer_count = table_len / (size * stride);
        let mut values = vec![Coefficient::zero(); size];
        for outer in 0..outer_count {
            for inner in 0..stride {
                for output in 0..rows {
                    for (symbol, value) in values.iter_mut().enumerate() {
                        *value = transformed[(outer * size + symbol) * stride + inner][output];
                    }
                    let coefficients = input.interpolate(&values)?;
                    transformed[(outer * size) * stride + inner][output] = coefficients.bias;
                    for (coordinate, coefficient) in coefficients.weights.into_iter().enumerate() {
                        transformed[(outer * size + coordinate + 1) * stride + inner][output] =
                            coefficient;
                    }
                }
            }
        }
    }

    let bias = transformed[0].clone();
    let mut matrix = Vec::with_capacity(rows * (table_len - 1));
    for output in 0..rows {
        matrix.extend(
            transformed
                .iter()
                .skip(1)
                .map(|coefficients| coefficients[output]),
        );
    }
    Ok(TensorMap {
        input_sizes,
        rows,
        matrix,
        bias,
    })
}

pub fn compile_multivariate_lut<F, O>(
    inputs: &[&dyn BlockEncoding<F>],
    output: &O,
    table: &[usize],
) -> Result<TensorMap<F>>
where
    F: BlockScalar,
    O: BlockEncoding<F> + ?Sized,
{
    let input_sizes = inputs
        .iter()
        .map(|input| input.alphabet_size())
        .collect::<Vec<_>>();
    let table_len = checked_product(&input_sizes)?;
    ensure!(
        table.len() == table_len,
        "LUT has {} entries, expected {table_len}",
        table.len()
    );
    let output_values = table
        .iter()
        .map(|&value| {
            ensure!(
                value < output.alphabet_size(),
                "LUT output {value} is outside alphabet of size {}",
                output.alphabet_size()
            );
            output.encode(value)
        })
        .collect::<Result<Vec<_>>>()?;
    compile_multivariate_coordinates(inputs, &output_values)
}

fn checked_product(values: &[usize]) -> Result<usize> {
    values.iter().try_fold(1usize, |product, &value| {
        product
            .checked_mul(value)
            .ok_or_else(|| anyhow::anyhow!("tensor basis is too large"))
    })
}
