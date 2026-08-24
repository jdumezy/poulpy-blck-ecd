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

            #[test]
            fn layouts() {
                assert_eq!($params.n, 256);
                $crate::test_suite::layouts::test_layouts::<$backend, $scalar, $encoder>(
                    $params,
                    &MODULE,
                    &HOST_MODULE,
                );
            }
        }
    };
}
