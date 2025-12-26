use std::collections::BinaryHeap;

use fxhash::FxHashMap;
use pyo3::exceptions::PyValueError;
use pyo3::PyResult;
use rand::prelude::SliceRandom;

use crate::ordering::SimOrd;
use crate::utils::{Neighbor, DEFAULT_PRED};

pub(crate) fn compute_pred(
    task: &str,
    k_neighbor_sims: &[f32],
    k_neighbor_labels: &[f32],
) -> PyResult<f32> {
    let pred = match task {
        "rating" => {
            let sum_sims: f32 = k_neighbor_sims.iter().sum();
            k_neighbor_sims
                .iter()
                .zip(k_neighbor_labels.iter())
                .map(|(&sim, &label)| label * sim / sum_sims)
                .sum()
        }

        "ranking" => {
            let sum_sims: f32 = k_neighbor_sims.iter().sum();
            sum_sims / k_neighbor_sims.len() as f32
        }

        _ => {
            let err_msg = format!("Unknown task type: \"{task}\"");
            return Err(PyValueError::new_err(err_msg));
        }
    };

    Ok(pred)
}

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

    let mut sim_by_id: FxHashMap<u32, f32> = FxHashMap::default();
    sim_by_id.reserve(sim_num);
    for nb in &neighbors[..sim_num] {
        sim_by_id.insert(nb.id, nb.sim);
    }

    let mut max_heap: BinaryHeap<SimOrd> = BinaryHeap::new();
    for (id, label) in interactions {
        if let Some(&sim) = sim_by_id.get(&id) {
            max_heap.push(SimOrd(sim, label));
        }
    }

    let mut k_neighbor_sims = Vec::new();
    let mut k_neighbor_labels = Vec::new();
    for _ in 0..k {
        match max_heap.pop() {
            Some(SimOrd(sim, label)) => {
                k_neighbor_sims.push(sim);
                k_neighbor_labels.push(label);
            }
            None => break,
        }
    }

    if k_neighbor_sims.is_empty() {
        Ok(DEFAULT_PRED)
    } else {
        compute_pred(task, &k_neighbor_sims, &k_neighbor_labels)
    }
}

pub(crate) fn get_rec_items(
    item_sim_scores: FxHashMap<u32, f32>,
    n_rec: usize,
    random_rec: bool,
) -> Vec<u32> {
    if random_rec && item_sim_scores.len() > n_rec {
        let mut rng = &mut rand::thread_rng();
        item_sim_scores
            .keys()
            .copied()
            .collect::<Vec<u32>>()
            .choose_multiple(&mut rng, n_rec)
            .cloned()
            .collect::<Vec<_>>()
    } else {
        let mut item_preds: Vec<(u32, f32)> = item_sim_scores.into_iter().collect();
        item_preds.sort_unstable_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap().reverse());
        item_preds
            .into_iter()
            .take(n_rec)
            .map(|(i, _)| i)
            .collect::<Vec<_>>()
    }
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
        let pred = predict_single(&neighbors, interactions.into_iter(), 2, "rating")?;
        assert!((pred - 4.8 / 1.4).abs() < 1e-6);

        Ok(())
    }

    #[test]
    fn test_compute_pred() -> PyResult<()> {
        let k_neighbor_sims = vec![0.1, 0.2, 0.3];
        let k_neighbor_labels = vec![2.0, 4.0, 1.0];
        let pred = compute_pred("rating", &k_neighbor_sims, &k_neighbor_labels);
        assert!(pred.is_ok());
        assert!((pred? - 2.166_666_7).abs() < 1e-4);

        let pred = compute_pred("ranking", &k_neighbor_sims, &k_neighbor_labels);
        assert!(pred.is_ok());
        assert_eq!(pred?, 0.2);

        pyo3::prepare_freethreaded_python();
        let pred = compute_pred("unknown", &k_neighbor_sims, &k_neighbor_labels);
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
