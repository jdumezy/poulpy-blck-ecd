pub mod coverage;
pub mod layouts;

#[macro_export]
macro_rules! ckks_block_backend_test_suite {
    (
        mod $modname:ident,
        backend = $backend:ty,
        scalar = $scalar:ty,
        encoder = $encoder:ty,
        params = $params:expr $(,)?) => {
        mod $modname {
            use std::sync::LazyLock;

            use poulpy_hal::layouts::{HostBytesBackend, Module};

            static MODULE: LazyLock<Module<$backend>> =
                LazyLock::new(|| Module::<$backend>::new($params.n as u64));
            static HOST_MODULE: LazyLock<Module<HostBytesBackend>> =
                LazyLock::new(|| Module::<HostBytesBackend>::new($params.n as u64));

            macro_rules! run_test {
                ($name:ident, $path:path) => {
                    #[test]
                    fn $name() {
                        use $path as test_fn;
                        assert_eq!($params.n, 256);
                        test_fn::<$backend, $scalar, $encoder>($params, &MODULE, &HOST_MODULE);
                    }
                };
            }

            run_test!(layouts, $crate::test_suite::layouts::test_layouts);
            run_test!(characters, $crate::test_suite::coverage::test_characters);
            run_test!(
                zeta_and_indicator,
                $crate::test_suite::coverage::test_zeta_and_indicator
            );
            run_test!(
                transform_strategies,
                $crate::test_suite::coverage::test_transform_strategies
            );
            run_test!(
                asymmetric_multivariate,
                $crate::test_suite::coverage::test_asymmetric_multivariate
            );
        }
    };
}
