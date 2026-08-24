use anyhow::{Result, ensure};

use super::{BlockEncoding, Coefficient};
use crate::scalar::BlockScalar;

#[derive(Clone, Debug, PartialEq)]
pub struct AffineFunction<F> {
    pub bias: Coefficient<F>,
    pub weights: Vec<Coefficient<F>>,
}

impl<F: BlockScalar> AffineFunction<F> {
    pub fn evaluate(&self, input: &[Coefficient<F>]) -> Result<Coefficient<F>> {
        ensure!(
            input.len() == self.weights.len(),
            "affine input width {}, expected {}",
            input.len(),
            self.weights.len()
        );
        Ok(self
            .weights
            .iter()
            .zip(input)
            .fold(self.bias, |acc, (&weight, &value)| acc + weight * value))
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct AffineMap<F> {
    pub rows: usize,
    pub cols: usize,
    pub matrix: Vec<Coefficient<F>>,
    pub bias: Vec<Coefficient<F>>,
}

impl<F: BlockScalar> AffineMap<F> {
    pub fn evaluate(&self, input: &[Coefficient<F>]) -> Result<Vec<Coefficient<F>>> {
        ensure!(
            input.len() == self.cols,
            "affine input width {}, expected {}",
            input.len(),
            self.cols
        );
        Ok((0..self.rows)
            .map(|row| {
                (0..self.cols).fold(self.bias[row], |acc, col| {
                    acc + self.matrix[row * self.cols + col] * input[col]
                })
            })
            .collect())
    }

    pub fn compose(&self, input: &Self) -> Result<Self> {
        ensure!(
            self.cols == input.rows,
            "cannot compose {}x{} and {}x{} affine maps",
            self.rows,
            self.cols,
            input.rows,
            input.cols
        );
        let mut matrix = vec![Coefficient::zero(); self.rows * input.cols];
        let mut bias = self.bias.clone();
        for (row, output_bias) in bias.iter_mut().enumerate() {
            for middle in 0..self.cols {
                let outer = self.matrix[row * self.cols + middle];
                *output_bias = *output_bias + outer * input.bias[middle];
                for col in 0..input.cols {
                    let index = row * input.cols + col;
                    matrix[index] = matrix[index] + outer * input.matrix[middle * input.cols + col];
                }
            }
        }
        Ok(Self {
            rows: self.rows,
            cols: input.cols,
            matrix,
            bias,
        })
    }

    pub fn is_exact(&self) -> bool {
        self.matrix
            .iter()
            .chain(&self.bias)
            .all(Coefficient::is_exact)
    }
}

pub fn compile_coordinates<F, I>(
    input: &I,
    output_values: &[Vec<Coefficient<F>>],
) -> Result<AffineMap<F>>
where
    F: BlockScalar,
    I: BlockEncoding<F>,
{
    let alphabet_size = input.alphabet_size();
    ensure!(
        output_values.len() == alphabet_size,
        "coordinate table has {} rows, expected {alphabet_size}",
        output_values.len()
    );
    let rows = output_values.first().map_or(0, Vec::len);
    ensure!(rows != 0, "coordinate table must have at least one output");
    ensure!(
        output_values.iter().all(|row| row.len() == rows),
        "coordinate table rows have different widths"
    );

    let mut matrix = Vec::with_capacity(rows * input.block_size());
    let mut bias = Vec::with_capacity(rows);
    for output in 0..rows {
        let values = output_values
            .iter()
            .map(|row| row[output])
            .collect::<Vec<_>>();
        let function = input.interpolate(&values)?;
        ensure!(
            function.weights.len() == input.block_size(),
            "interpolator returned the wrong width"
        );
        bias.push(function.bias);
        matrix.extend(function.weights);
    }
    Ok(AffineMap {
        rows,
        cols: input.block_size(),
        matrix,
        bias,
    })
}

pub fn compile_lut<F, I, O>(input: &I, output: &O, table: &[usize]) -> Result<AffineMap<F>>
where
    F: BlockScalar,
    I: BlockEncoding<F>,
    O: BlockEncoding<F>,
{
    ensure!(
        table.len() == input.alphabet_size(),
        "LUT has {} entries, expected {}",
        table.len(),
        input.alphabet_size()
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
    compile_coordinates(input, &output_values)
}
