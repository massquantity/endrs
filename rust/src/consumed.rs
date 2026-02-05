use fxhash::{FxHashMap, FxHashSet};
use pyo3::prelude::*;
use pyo3::types::{IntoPyDict, PyDict, PyList};

/// Build user_consumed and item_consumed with consecutive duplicates removed.
///
/// Consecutive duplicates are removed, but non-consecutive duplicates are preserved.
/// Example: [10, 10, 20, 10, 30, 30] -> [10, 20, 10, 30]
#[pyfunction]
#[pyo3(name = "build_consumed_unique")]
pub fn build_consumed<'py>(
    py: Python<'py>,
    user_indices: &Bound<'py, PyList>,
    item_indices: &Bound<'py, PyList>,
) -> PyResult<(Bound<'py, PyDict>, Bound<'py, PyDict>)> {
    let user_indices: Vec<u32> = user_indices.extract()?;
    let item_indices: Vec<u32> = item_indices.extract()?;
    let mut user_consumed: FxHashMap<u32, Vec<u32>> = FxHashMap::default();
    let mut item_consumed: FxHashMap<u32, Vec<u32>> = FxHashMap::default();

    for (&u, &i) in user_indices.iter().zip(item_indices.iter()) {
        user_consumed.entry(u).or_default().push(i);
        item_consumed.entry(i).or_default().push(u);
    }

    // Remove consecutive duplicates
    user_consumed.values_mut().for_each(|v| v.dedup());
    item_consumed.values_mut().for_each(|v| v.dedup());

    let user_consumed_py = user_consumed.into_py_dict(py)?;
    let item_consumed_py = item_consumed.into_py_dict(py)?;

    Ok((user_consumed_py, item_consumed_py))
}

/// Get consumed item set for a user if filtering is enabled
pub(crate) fn get_consumed_set(
    user_consumed: &FxHashMap<u32, Vec<u32>>,
    u: u32,
    filter_consumed: bool,
) -> Option<FxHashSet<u32>> {
    filter_consumed
        .then(|| {
            user_consumed
                .get(&u)
                .map(|v| v.iter().copied().collect())
        })
        .flatten()
}
