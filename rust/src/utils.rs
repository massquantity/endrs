use pyo3::exceptions::PyRuntimeError;
use pyo3::PyResult;
use rayon::{ThreadPool, ThreadPoolBuilder};

pub(crate) const OOV_IDX: usize = 0;
pub(crate) const DEFAULT_PRED: f32 = 0.0;

#[derive(Debug)]
pub(crate) struct SimVals {
    pub(crate) x1: i32,
    pub(crate) x2: i32,
    pub(crate) prod: f32,
    pub(crate) count: usize,
    pub(crate) cosine: f32,
}

// (prod, count)
pub(crate) type CumValues = (f32, usize);

/// Encodes a pair of i32 values into a single u64 key.
/// x1 is stored in the upper 32 bits, x2 in the lower 32 bits.
#[inline]
pub(crate) fn encode_pair(x1: i32, x2: i32) -> u64 {
    ((x1 as u64) << 32) | (x2 as u64)
}

/// Decodes a u64 key back into a pair of usize values.
#[inline]
pub(crate) fn decode_pair(key: u64) -> (usize, usize) {
    let x1 = (key >> 32) as usize;
    let x2 = (key & 0xFFFFFFFF) as usize;
    (x1, x2)
}

pub(crate) fn create_thread_pool(num_threads: usize) -> PyResult<ThreadPool> {
    let pool = ThreadPoolBuilder::new()
        .num_threads(num_threads)
        .build()
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to create thread pool: {}", e)))?;

    Ok(pool)
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
