/// Returns the natural effective width after a ciphertext-plaintext product.
pub(crate) fn plaintext_output_k(input_k: usize, plaintext_log_delta: usize) -> usize {
    input_k.saturating_sub(plaintext_log_delta)
}

/// Returns the natural effective width after a ciphertext-ciphertext product.
pub(crate) fn multiplication_output_k(
    lhs_k: usize,
    lhs_log_delta: usize,
    rhs_k: usize,
    rhs_log_delta: usize,
) -> usize {
    let lhs_log_budget = lhs_k.saturating_sub(lhs_log_delta);
    let rhs_log_budget = rhs_k.saturating_sub(rhs_log_delta);
    lhs_log_budget
        .min(rhs_log_budget)
        .saturating_sub(lhs_log_delta.max(rhs_log_delta))
        .saturating_add(lhs_log_delta.min(rhs_log_delta))
}

#[cfg(test)]
mod tests {
    use super::{multiplication_output_k, plaintext_output_k};

    #[test]
    fn precision_costs_are_bit_granular() {
        assert_eq!(plaintext_output_k(320, 6), 314);
        assert_eq!(multiplication_output_k(320, 40, 320, 40), 280);
        assert_eq!(multiplication_output_k(300, 40, 280, 30), 240);
    }
}
