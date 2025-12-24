use std::time::Instant;

use fxhash::FxHashMap;
use pyo3::PyResult;

use crate::similarities::{
    aggregate_sims, compute_cosine, compute_pair_stats, insert_sorted_neighbors,
};
use crate::sparse::{get_row, CsrMatrix};
use crate::utils::{decode_pair, CumValues, Neighbor};

#[allow(clippy::needless_range_loop)]
pub(crate) fn update_sum_squares(
    sum_squares: &mut Vec<f32>,
    interactions: &CsrMatrix<u32, f32>,
    num: usize,
) {
    // consider oov 0
    if num + 1 > sum_squares.len() {
        sum_squares.resize(num + 1, 0.0);
    }

    for i in 1..=num {
        if let Some(row) = get_row(interactions, i, false) {
            sum_squares[i] += row.map(|(_, d)| d * d).sum::<f32>()
        }
    }
}

/// Incrementally update cosine similarities with new interaction data.
///
/// Similar to `invert_cosine`, but accumulates new statistics into existing `cum_values`
/// rather than replacing them. Used for incremental/online learning scenarios where
/// new interactions arrive over time.
///
/// # Arguments
/// * `interactions` - CSR matrix containing only the new interactions
/// * `sum_squares` - Updated sum of squared values for each column
/// * `cum_values` - Existing statistics map to be updated in-place
/// * `n` - Number of columns (excluding OOV index 0)
/// * `min_common` - Minimum cumulative co-occurrence count to include in results
///
/// # Returns
/// Vector of (col1, col2, cosine_similarity) tuples with updated similarities
pub(crate) fn update_cosine(
    interactions: &CsrMatrix<u32, f32>,
    sum_squares: &[f32],
    cum_values: &mut FxHashMap<u64, CumValues>,
    n: usize,
    min_common: u32,
) -> PyResult<Vec<(u32, u32, f32)>> {
    let start = Instant::now();
    let pair_stats = compute_pair_stats(interactions, n);
    let mut cosine_sims: Vec<(u32, u32, f32)> = Vec::new();

    for (key, (prod, count)) in pair_stats {
        let (cum_prod, cum_count) = cum_values
            .entry(key)
            .and_modify(|(p, c)| {
                *p += prod;
                *c += count;
            })
            .or_insert((prod, count));

        if *cum_count >= min_common {
            let (x1, x2) = decode_pair(key);
            let cosine = compute_cosine(
                *cum_prod,
                sum_squares[x1 as usize],
                sum_squares[x2 as usize],
            );
            cosine_sims.push((x1, x2, cosine));
        }
    }

    let duration = start.elapsed();
    println!(
        "incremental cosine sim: {} elapsed: {:.4?}",
        cosine_sims.len(),
        duration
    );
    Ok(cosine_sims)
}

/// Incrementally update sorted neighbor lists with new similarity scores.
///
/// Merges new similarity pairs into existing neighbor lists, replacing old similarities
/// for the same neighbor and re-sorting. Used after `update_cosine` to maintain
/// sorted neighbor lists in incremental learning.
///
/// # Arguments
/// * `cosine_sims` - New (node1, node2, similarity) tuples to merge
/// * `sims` - Existing neighbor mapping to update in-place
///
/// # Process
/// 1. Aggregate new similarities by node (bidirectional)
/// 2. For each node with new neighbors, merge with existing neighbors
/// 3. New similarities override old ones for the same neighbor pair
/// 4. Re-sort combined list by similarity (descending)
pub(crate) fn update_by_sims(
    cosine_sims: &[(u32, u32, f32)],
    sims: &mut FxHashMap<u32, Vec<Neighbor>>,
) -> PyResult<()> {
    let start = Instant::now();
    let agg_sims = aggregate_sims(cosine_sims);

    for (key, new_neighbor_sims) in agg_sims {
        // Use remove instead of get to avoid extra copy
        let combined_sims = match sims.remove(&key) {
            Some(neighbors) => {
                let mut original: FxHashMap<u32, f32> = neighbors
                    .into_iter()
                    .map(|n| (n.id, n.sim))
                    .collect();
                original.extend(new_neighbor_sims);
                original.into_iter().collect()
            }
            None => new_neighbor_sims,
        };

        insert_sorted_neighbors(sims, key, combined_sims);
    }

    let duration = start.elapsed();
    println!("incremental sort elapsed: {duration:.4?}");
    Ok(())
}
