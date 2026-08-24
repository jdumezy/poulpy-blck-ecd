use anyhow::{Result, bail, ensure};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FinitePoset {
    size: usize,
    leq: Vec<bool>,
}

impl FinitePoset {
    pub fn new(size: usize, leq: Vec<bool>) -> Result<Self> {
        ensure!(size >= 2, "a finite poset needs at least two elements");
        ensure!(
            leq.len() == size * size,
            "relation has {} entries, expected {}",
            leq.len(),
            size * size
        );
        for value in 0..size {
            ensure!(
                leq[value * size + value],
                "relation is not reflexive at {value}"
            );
        }
        for lhs in 0..size {
            for rhs in lhs + 1..size {
                ensure!(
                    !(leq[lhs * size + rhs] && leq[rhs * size + lhs]),
                    "relation is not antisymmetric at ({lhs}, {rhs})"
                );
            }
        }
        for lhs in 0..size {
            for middle in 0..size {
                if !leq[lhs * size + middle] {
                    continue;
                }
                for rhs in 0..size {
                    ensure!(
                        !leq[middle * size + rhs] || leq[lhs * size + rhs],
                        "relation is not transitive at ({lhs}, {middle}, {rhs})"
                    );
                }
            }
        }
        Ok(Self { size, leq })
    }

    pub fn from_relation(
        size: usize,
        mut relation: impl FnMut(usize, usize) -> bool,
    ) -> Result<Self> {
        let leq = (0..size * size)
            .map(|index| relation(index / size, index % size))
            .collect();
        Self::new(size, leq)
    }

    pub fn chain(size: usize) -> Result<Self> {
        Self::from_relation(size, |lhs, rhs| lhs <= rhs)
    }

    pub fn boolean_lattice(bits: usize) -> Result<Self> {
        ensure!(bits > 0, "Boolean lattice dimension must be positive");
        let bits = u32::try_from(bits)
            .map_err(|_| anyhow::anyhow!("Boolean lattice dimension is too large"))?;
        let size = 1usize
            .checked_shl(bits)
            .ok_or_else(|| anyhow::anyhow!("Boolean lattice is too large"))?;
        Self::from_relation(size, |lhs, rhs| lhs & rhs == lhs)
    }

    pub fn rooted_tree(parents: &[Option<usize>]) -> Result<Self> {
        let size = parents.len();
        ensure!(size >= 2, "a rooted tree needs at least two vertices");
        let roots = parents
            .iter()
            .enumerate()
            .filter_map(|(value, parent)| parent.is_none().then_some(value))
            .collect::<Vec<_>>();
        ensure!(roots.len() == 1, "a rooted tree must have exactly one root");
        let root = roots[0];
        for (value, &parent) in parents.iter().enumerate() {
            if let Some(parent) = parent {
                ensure!(
                    parent < size,
                    "parent {parent} of vertex {value} is out of range"
                );
                ensure!(parent != value, "vertex {value} is its own parent");
            }
        }
        for start in 0..size {
            let mut seen = vec![false; size];
            let mut value = start;
            loop {
                ensure!(
                    !seen[value],
                    "parent relation contains a cycle through {value}"
                );
                seen[value] = true;
                match parents[value] {
                    Some(parent) => value = parent,
                    None => {
                        ensure!(value == root, "vertex {start} does not reach the root");
                        break;
                    }
                }
            }
        }
        Self::from_relation(size, |ancestor, mut value| {
            loop {
                if ancestor == value {
                    break true;
                }
                match parents[value] {
                    Some(parent) => value = parent,
                    None => break false,
                }
            }
        })
    }

    pub fn divisor_lattice(modulus: usize) -> Result<(Self, Vec<usize>)> {
        ensure!(modulus >= 2, "divisor lattice modulus must be at least 2");
        let divisors = (1..=modulus)
            .filter(|divisor| modulus.is_multiple_of(*divisor))
            .collect::<Vec<_>>();
        let poset = Self::from_relation(divisors.len(), |lhs, rhs| {
            divisors[rhs].is_multiple_of(divisors[lhs])
        })?;
        Ok((poset, divisors))
    }

    pub fn size(&self) -> usize {
        self.size
    }

    pub fn leq(&self, lhs: usize, rhs: usize) -> bool {
        lhs < self.size && rhs < self.size && self.leq[lhs * self.size + rhs]
    }

    pub fn bottom(&self) -> Option<usize> {
        (0..self.size).find(|&candidate| (0..self.size).all(|value| self.leq(candidate, value)))
    }

    pub fn top(&self) -> Option<usize> {
        (0..self.size).find(|&candidate| (0..self.size).all(|value| self.leq(value, candidate)))
    }

    pub fn mobius(&self) -> Result<Vec<i128>> {
        let order = self.linear_extension()?;
        let mut mobius = vec![0i128; self.size * self.size];
        for (rhs_index, &rhs) in order.iter().enumerate() {
            mobius[rhs * self.size + rhs] = 1;
            for &lhs in &order[..rhs_index] {
                if !self.leq(lhs, rhs) {
                    continue;
                }
                let mut sum = 0i128;
                for &middle in &order[..rhs_index] {
                    if self.leq(lhs, middle) && self.leq(middle, rhs) {
                        sum = sum
                            .checked_add(mobius[lhs * self.size + middle])
                            .ok_or_else(|| anyhow::anyhow!("Möbius coefficient overflow"))?;
                    }
                }
                mobius[lhs * self.size + rhs] = sum
                    .checked_neg()
                    .ok_or_else(|| anyhow::anyhow!("Möbius coefficient overflow"))?;
            }
        }
        Ok(mobius)
    }

    pub fn meet_table(&self) -> Result<Vec<usize>> {
        self.bound_table(true)
    }

    pub fn join_table(&self) -> Result<Vec<usize>> {
        self.bound_table(false)
    }

    fn linear_extension(&self) -> Result<Vec<usize>> {
        let mut remaining = vec![true; self.size];
        let mut order = Vec::with_capacity(self.size);
        while order.len() != self.size {
            let next = (0..self.size).find(|&candidate| {
                remaining[candidate]
                    && (0..self.size).all(|predecessor| {
                        !remaining[predecessor]
                            || predecessor == candidate
                            || !self.leq(predecessor, candidate)
                    })
            });
            let Some(next) = next else {
                bail!("partial order has no linear extension");
            };
            remaining[next] = false;
            order.push(next);
        }
        Ok(order)
    }

    fn bound_table(&self, meet: bool) -> Result<Vec<usize>> {
        let mut table = Vec::with_capacity(self.size * self.size);
        for lhs in 0..self.size {
            for rhs in 0..self.size {
                let candidates = (0..self.size)
                    .filter(|&value| {
                        if meet {
                            self.leq(value, lhs) && self.leq(value, rhs)
                        } else {
                            self.leq(lhs, value) && self.leq(rhs, value)
                        }
                    })
                    .collect::<Vec<_>>();
                let bound = candidates.iter().copied().find(|&candidate| {
                    candidates.iter().copied().all(|value| {
                        if meet {
                            self.leq(value, candidate)
                        } else {
                            self.leq(candidate, value)
                        }
                    })
                });
                let Some(bound) = bound else {
                    let operation = if meet { "meet" } else { "join" };
                    bail!("elements {lhs} and {rhs} have no {operation}");
                };
                table.push(bound);
            }
        }
        Ok(table)
    }
}
