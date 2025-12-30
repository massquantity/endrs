use fxhash::{FxHashMap, FxHashSet};

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
