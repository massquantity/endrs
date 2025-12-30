use fxhash::FxHashMap;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyList;
use rand::seq::IteratorRandom;

use crate::consumed::get_consumed_set;
use crate::sparse::{get_row, CsrMatrix};
use crate::utils::{Neighbor, DEFAULT_PRED};

/// Predict a score for a single user-item pair using neighbor-based collaborative filtering.
///
/// The function finds the intersection between the user's top-k similar neighbors and
/// their historical interactions, then computes a prediction based on the task type.
/// Since `neighbors` is pre-sorted by similarity in descending order, we iterate through
/// them sequentially to find the top-k overlapping neighbors with interactions.
///
/// # Arguments
/// * `neighbors` - Pre-sorted neighbors by similarity (descending)
/// * `interactions` - Iterator of (item_id, label) pairs from historical data
/// * `k` - Maximum number of neighbors to use for prediction
/// * `task` - "rating" for weighted average, "ranking" for average similarity
///
/// # Returns
/// * `rating` task: weighted_sum / sum_sims (similarity-weighted average of labels)
/// * `ranking` task: sum_sims / count (average similarity score)
/// * Returns DEFAULT_PRED if no intersection found
pub(crate) fn predict_single<I>(
    neighbors: &[Neighbor],
    interactions: I,
    k: usize,
    task: &str,
) -> PyResult<f32>
where
    I: Iterator<Item = (u32, f32)>,
{
    let sim_num = k.min(neighbors.len());
    let label_by_id: FxHashMap<u32, f32> = interactions.collect();

    let mut sum_sims = 0.0f32;
    let mut weighted_sum = 0.0f32;
    let mut count = 0usize;

    for nb in &neighbors[..sim_num] {
        if let Some(&label) = label_by_id.get(&nb.id) {
            sum_sims += nb.sim;
            weighted_sum += nb.sim * label;
            count += 1;
            if count == k {
                break;
            }
        }
    }

    if count == 0 {
        return Ok(DEFAULT_PRED);
    }

    match task {
        "rating" => Ok(weighted_sum / sum_sims),
        "ranking" => Ok(sum_sims / count as f32),
        _ => Err(PyValueError::new_err(format!(
            "Unknown task type: \"{task}\""
        ))),
    }
}

/// Select top-n recommended items from candidate scores.
///
/// # Algorithm
/// - **Random mode** (`random_rec=true`): Randomly sample n_rec items from candidates.
///   Uses `IteratorRandom::choose_multiple` for O(n) single-pass sampling.
///
/// - **Score mode** (`random_rec=false`): Select items with highest scores.
///   Uses `select_nth_unstable_by` (quickselect) for O(n) partitioning to find
///   top n_rec items, then sorts only those n_rec items in O(k log k).
///   This is more efficient than full O(n log n) sort when k << n.
///
/// # Arguments
/// * `item_sim_scores` - HashMap of item_id -> aggregated similarity score
/// * `n_rec` - Number of items to recommend
/// * `random_rec` - If true, randomly sample; if false, select by highest score
///
/// # Returns
/// Vector of recommended item IDs, sorted by score descending (in score mode)
pub(crate) fn get_rec_items(
    item_sim_scores: FxHashMap<u32, f32>,
    n_rec: usize,
    random_rec: bool,
) -> Vec<u32> {
    if random_rec && item_sim_scores.len() > n_rec {
        let mut rng = rand::thread_rng();
        item_sim_scores
            .into_keys()
            .choose_multiple(&mut rng, n_rec)
    } else {
        let mut items: Vec<(u32, f32)> = item_sim_scores.into_iter().collect();

        if n_rec > 0 && items.len() > n_rec {
            // O(n) to select top n_rec
            items.select_nth_unstable_by(n_rec - 1, |(_, a), (_, b)| b.partial_cmp(a).unwrap());
            items.truncate(n_rec);
        }
        // Sort only top n_rec: O(k log k)
        items.sort_unstable_by(|(_, a), (_, b)| b.partial_cmp(a).unwrap());
        items.into_iter().map(|(i, _)| i).collect()
    }
}

pub(crate) fn empty_recs(py: Python, n_rec: usize) -> (Bound<PyList>, usize) {
    (PyList::empty(py), n_rec)
}

pub(crate) fn finalize_recs(
    py: Python,
    item_scores: FxHashMap<u32, f32>,
    n_rec: usize,
    random_rec: bool,
) -> PyResult<(Bound<PyList>, usize)> {
    if item_scores.is_empty() {
        return Ok((PyList::empty(py), n_rec));
    }
    let items = get_rec_items(item_scores, n_rec, random_rec);
    let additional = n_rec - items.len();
    Ok((PyList::new(py, items)?, additional))
}

/// Generate item recommendations for multiple users based on item-item similarities.
///
/// For each user, this function aggregates similarity scores from their historically
/// interacted items to candidate items, then selects top-n recommendations.
///
/// # Algorithm
/// For each user:
/// 1. Get user's historical interactions from the sparse matrix
/// 2. For each interacted item, look up its similar items (neighbors)
/// 3. Aggregate scores: score[candidate] += sim(item, candidate) * interaction_label
/// 4. Filter out already-consumed items (if enabled)
/// 5. Select top n_rec items by aggregated score (or random sampling)
///
/// # Arguments
/// * `py` - Python interpreter reference
/// * `users` - List of user IDs to generate recommendations for
/// * `n_rec` - Number of items to recommend per user
/// * `filter_consumed` - If true, exclude items the user has already interacted with
/// * `random_rec` - If true, randomly sample from candidates; if false, select by score
/// * `k_sim` - Number of similar items to consider per interacted item
/// * `user_consumed` - Map of user_id -> list of consumed item_ids
/// * `user_interactions` - Sparse matrix of user interactions (user_id -> [(item_id, label)])
/// * `item_sims` - Pre-computed item similarities: item_id -> sorted neighbors
///
/// # Returns
/// Tuple of (recommendations, additional_counts):
/// * recommendations: Vec of PyList, each containing recommended item IDs for a user
/// * additional_counts: PyList of how many more items needed to reach n_rec per user
#[rustfmt::skip]
#[allow(clippy::too_many_arguments)]
pub(crate) fn recommend_by_item_sims<'py>(
    py: Python<'py>,
    users: &Bound<'py, PyList>,
    n_rec: usize,
    filter_consumed: bool,
    random_rec: bool,
    k_sim: usize,
    user_consumed: &FxHashMap<u32, Vec<u32>>,
    user_interactions: &CsrMatrix<u32, f32>,
    item_sims: &FxHashMap<u32, Vec<Neighbor>>,
) -> PyResult<(Vec<Bound<'py, PyList>>, Bound<'py, PyList>)> {
    let mut recs = Vec::new();
    let mut additional_rec_counts = Vec::new();

    for u in users {
        let u: u32 = u.extract()?;
        let consumed = get_consumed_set(user_consumed, u, filter_consumed);

        let (rec_items, additional_count) =
            if let Some(row) = get_row(user_interactions, u as usize, false) {
                let mut item_scores: FxHashMap<u32, f32> = FxHashMap::default();
                for (i, i_label) in row {
                    if let Some(neighbors) = item_sims.get(&i) {
                        let sim_num = std::cmp::min(k_sim, neighbors.len());
                        for nb in &neighbors[..sim_num] {
                            if consumed.as_ref().is_some_and(|c| c.contains(&nb.id)) {
                                continue;
                            }

                            let delta = nb.sim * i_label;
                            item_scores
                                .entry(nb.id)
                                .and_modify(|score| *score += delta)
                                .or_insert(delta);
                        }
                    }
                }
                finalize_recs(py, item_scores, n_rec, random_rec)?
            } else {
                empty_recs(py, n_rec)
            };

        recs.push(rec_items);
        additional_rec_counts.push(additional_count);
    }

    let additional_rec_counts = PyList::new(py, additional_rec_counts)?;
    Ok((recs, additional_rec_counts))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_predict_single() -> PyResult<()> {
        pyo3::prepare_freethreaded_python();

        let neighbors = vec![
            Neighbor { id: 4, sim: 0.8 },
            Neighbor { id: 2, sim: 0.6 },
            Neighbor { id: 5, sim: 0.5 },
            Neighbor { id: 3, sim: 0.3 },
            Neighbor { id: 1, sim: 0.1 },
        ];
        let interactions = vec![(1, 2.0), (2, 4.0), (3, 1.0), (4, 3.0), (5, 5.0)];

        // ranking task: sum_sims / len = (0.8 + 0.6) / 2 = 0.7
        let pred = predict_single(&neighbors, interactions.clone().into_iter(), 2, "ranking")?;
        assert!((pred - 0.7).abs() < 1e-6);

        // rating task: weighted average
        // top 2 by sim: id=4 (sim=0.8, label=3.0), id=2 (sim=0.6, label=4.0)
        // pred = (0.8*3.0 + 0.6*4.0) / (0.8 + 0.6) = 4.8 / 1.4
        let pred = predict_single(&neighbors, interactions.clone().into_iter(), 2, "rating")?;
        assert!((pred - 4.8 / 1.4).abs() < 1e-6);

        // unknown task type
        let pred = predict_single(&neighbors, interactions.into_iter(), 2, "unknown");
        assert!(pred.is_err());
        assert_eq!(
            pred.unwrap_err().to_string(),
            "ValueError: Unknown task type: \"unknown\""
        );

        Ok(())
    }

    #[test]
    fn test_compute_rec_items() -> PyResult<()> {
        let mut item_sim_scores = FxHashMap::default();
        item_sim_scores.insert(1, 1.0);
        item_sim_scores.insert(2, 2.0);
        item_sim_scores.insert(3, 3.0);
        item_sim_scores.insert(4, 4.0);
        item_sim_scores.insert(5, 5.0);

        (0..10).for_each({
            |_| {
                let scores = item_sim_scores.clone();
                let rec_items = get_rec_items(scores, 3, true);
                rec_items
                    .iter()
                    .all(|i| item_sim_scores.contains_key(i));
            }
        });

        let rec_items = get_rec_items(item_sim_scores, 3, false);
        assert_eq!(rec_items, vec![5, 4, 3]);
        Ok(())
    }
}
