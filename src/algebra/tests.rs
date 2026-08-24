use super::{AffineMap, Coefficient, TensorMap};

fn zeros(count: usize) -> Vec<Coefficient<f64>> {
    vec![Coefficient::zero(); count]
}

#[test]
fn affine_map_rejects_invalid_storage() {
    assert!(AffineMap::<f64>::new(0, 1, Vec::new(), Vec::new()).is_err());
    assert!(AffineMap::<f64>::new(1, 0, Vec::new(), zeros(1)).is_err());
    assert!(AffineMap::<f64>::new(2, 3, zeros(5), zeros(2)).is_err());
    assert!(AffineMap::<f64>::new(2, 3, zeros(6), zeros(1)).is_err());
    assert!(AffineMap::<f64>::new(usize::MAX, 2, Vec::new(), Vec::new()).is_err());
    assert!(AffineMap::<f64>::identity(0).is_err());
    assert!(AffineMap::<f64>::identity(usize::MAX).is_err());
}

#[test]
fn affine_map_accessors_and_composition_are_consistent() {
    let map: AffineMap<f64> = AffineMap::new(
        2,
        2,
        vec![
            Coefficient::integer(2),
            Coefficient::integer(1),
            Coefficient::integer(-1),
            Coefficient::integer(3),
        ],
        vec![Coefficient::integer(4), Coefficient::integer(-2)],
    )
    .unwrap();
    assert_eq!(map.rows(), 2);
    assert_eq!(map.cols(), 2);
    assert_eq!(map.matrix().len(), 4);
    assert_eq!(map.bias().len(), 2);
    assert!(map.evaluate(&[Coefficient::one()]).is_err());

    let identity = AffineMap::identity(2).unwrap();
    assert_eq!(map.compose(&identity).unwrap(), map);
    assert!(map.compose(&AffineMap::identity(3).unwrap()).is_err());
}

#[test]
fn tensor_map_rejects_invalid_storage() {
    assert!(TensorMap::<f64>::new(Vec::new(), 1, Vec::new(), zeros(1)).is_err());
    assert!(TensorMap::<f64>::new(vec![2], 1, zeros(1), zeros(1)).is_err());
    assert!(TensorMap::<f64>::new(vec![2, 1], 1, zeros(1), zeros(1)).is_err());
    assert!(TensorMap::<f64>::new(vec![2, 2], 0, Vec::new(), Vec::new()).is_err());
    assert!(TensorMap::<f64>::new(vec![2, 3], 2, zeros(9), zeros(2)).is_err());
    assert!(TensorMap::<f64>::new(vec![2, 3], 2, zeros(10), zeros(1)).is_err());
    assert!(TensorMap::<f64>::new(vec![usize::MAX, 2], 1, Vec::new(), zeros(1)).is_err());
}

#[test]
fn tensor_map_accessors_and_affine_view_are_consistent() {
    let tensor = TensorMap::new(vec![2, 3], 2, zeros(10), zeros(2)).unwrap();
    assert_eq!(tensor.input_sizes(), &[2, 3]);
    assert_eq!(tensor.input_widths().collect::<Vec<_>>(), [1, 2]);
    assert_eq!(tensor.feature_width(), 5);
    assert_eq!(tensor.rows(), 2);
    assert_eq!(tensor.matrix().len(), 10);
    assert_eq!(tensor.bias().len(), 2);

    let affine = tensor.as_affine();
    assert_eq!(affine.rows(), tensor.rows());
    assert_eq!(affine.cols(), tensor.feature_width());
    assert_eq!(affine.matrix(), tensor.matrix());
    assert_eq!(affine.bias(), tensor.bias());
}
