use std::sync::Arc;

use pyo3::exceptions::PyRuntimeError;
use pyo3::PyResult;
use rayon::{ThreadPool, ThreadPoolBuilder};

pub(crate) const OOV_IDX: usize = 0;
pub(crate) const DEFAULT_PRED: f32 = 0.0;

pub(crate) type CumValues = (i32, i32, f32, usize);

pub(crate) fn create_thread_pool(num_threads: usize) -> PyResult<Arc<ThreadPool>> {
    let pool = ThreadPoolBuilder::new()
        .num_threads(num_threads)
        .build()
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to create thread pool: {}", e)))?;

    Ok(Arc::new(pool))
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use rayon::prelude::*;

    use super::*;

    #[test]
    fn test_create_thread_pool_success() {
        let result = create_thread_pool(4);
        assert!(result.is_ok());

        let pool = result.unwrap();
        assert_eq!(pool.current_num_threads(), 4);
    }

    #[test]
    fn test_thread_pool_actual_execution() {
        let pool = create_thread_pool(2).unwrap();
        let counter = Arc::new(Mutex::new(0));
        let counter_clone = Arc::clone(&counter);

        pool.install(|| {
            (0..10).into_par_iter().for_each(|_| {
                let mut count = counter_clone.lock().unwrap();
                *count += 1;
            });
        });

        let final_count = *counter.lock().unwrap();
        assert_eq!(final_count, 10);
    }
}
