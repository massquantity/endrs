use std::cmp::Ordering;
use std::sync::Arc;

use dashmap::DashMap;
use fxhash::FxHashMap;
use pyo3::PyResult;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::sparse::{get_row, CsrMatrix};

type ScoredItems = Vec<(i32, f32)>;

#[derive(Serialize, Deserialize)]
pub(crate) struct Graph {
    pub(crate) n_users: usize,
    pub(crate) n_items: usize,
    alpha: f32,
    max_cache_num: usize,
}

impl Graph {
    pub fn new(n_users: usize, n_items: usize, alpha: f32, max_cache_num: usize) -> Self {
        Self {
            n_users,
            n_items,
            alpha,
            max_cache_num,
        }
    }

    fn compute_single_swing(
        &self,
        target_item: usize,
        user_interactions: &CsrMatrix<i32, f32>,
        item_interactions: &CsrMatrix<i32, f32>,
        prev_scores: &FxHashMap<i32, ScoredItems>,
        user_weights: &[f32],
        cached_common_items: Arc<DashMap<usize, Vec<usize>>>,
    ) -> (i32, ScoredItems) {
        let target_i32 = target_item as i32;
        let users = get_row_vec(item_interactions, target_item);
        if users.len() < 2 {
            let scores = prev_scores
                .get(&target_i32)
                .cloned()
                .unwrap_or_default();
            return (target_i32, scores);
        }

        let mut item_scores = init_item_scores(target_i32, self.n_items, prev_scores);
        for (j, &u) in users.iter().enumerate() {
            for &v in &users[(j + 1)..users.len()] {
                let key = u * self.n_users + v;
                let common_items = match cached_common_items.get(&key) {
                    Some(items) => items.to_owned(),
                    None => {
                        let items = get_intersect_items(
                            &get_row_vec(user_interactions, u),
                            &get_row_vec(user_interactions, v),
                        );
                        if cached_common_items.len() < self.max_cache_num {
                            cached_common_items.insert(key, items.clone());
                        }
                        items
                    }
                };

                // exclude item self according to the paper
                let k = (common_items.len() - 1) as f32;
                let score = user_weights[u] * user_weights[v] * (self.alpha + k).recip();
                for i in common_items {
                    if i != target_item {
                        item_scores[i] += score;
                    }
                }
            }
        }

        (target_i32, extract_valid_scores(item_scores))
    }

    pub(crate) fn compute_swing_scores(
        &self,
        user_interactions: &CsrMatrix<i32, f32>,
        item_interactions: &CsrMatrix<i32, f32>,
        prev_mapping: &FxHashMap<i32, ScoredItems>,
    ) -> PyResult<FxHashMap<i32, ScoredItems>> {
        let user_weights = compute_user_weights(user_interactions, self.n_users);
        let cached_common_items = Arc::new(DashMap::new());
        let swing_scores: Vec<(i32, ScoredItems)> = (1..=self.n_items)
            .into_par_iter()
            .map(|i| {
                self.compute_single_swing(
                    i,
                    user_interactions,
                    item_interactions,
                    prev_mapping,
                    &user_weights,
                    Arc::clone(&cached_common_items),
                )
            })
            .filter(|(_, s)| !s.is_empty())
            .collect();

        Ok(FxHashMap::from_iter(swing_scores))
    }
}

fn compute_user_weights(matrix: &CsrMatrix<i32, f32>, n_users: usize) -> Vec<f32> {
    let mut user_weights = vec![0.0; n_users + 1];
    // skip oov 0
    for (i, weight) in user_weights.iter_mut().enumerate().skip(1) {
        let n_items = matrix.indptr[i + 1] - matrix.indptr[i];
        *weight = if n_items == 0 {
            0.0
        } else {
            (n_items as f32).sqrt().recip()
        };
    }
    user_weights
}

fn get_intersect_items(u_items: &[usize], v_items: &[usize]) -> Vec<usize> {
    let mut i = 0;
    let mut j = 0;
    let mut common_items = Vec::new();
    while i < u_items.len() && j < v_items.len() {
        match u_items[i].cmp(&v_items[j]) {
            Ordering::Less => i += 1,
            Ordering::Greater => j += 1,
            Ordering::Equal => {
                common_items.push(u_items[i]);
                i += 1;
                j += 1;
            }
        }
    }
    common_items
}

fn get_row_vec(matrix: &CsrMatrix<i32, f32>, n: usize) -> Vec<usize> {
    if let Some(row) = get_row(matrix, n, false) {
        row.map(|(i, _)| i as usize).collect()
    } else {
        Vec::new()
    }
}

fn init_item_scores(
    target_item: i32,
    n_items: usize,
    prev_scores: &FxHashMap<i32, ScoredItems>,
) -> Vec<f32> {
    let mut item_scores = vec![0.0; n_items + 1];
    if let Some(scores) = prev_scores.get(&target_item) {
        for &(i, s) in scores {
            item_scores[i as usize] = s;
        }
    }
    item_scores
}

fn extract_valid_scores(scores: Vec<f32>) -> ScoredItems {
    let mut non_zero_scores: ScoredItems = scores
        .into_iter()
        .enumerate()
        .filter_map(|(i, score)| {
            if score != 0.0 {
                Some((i as i32, score))
            } else {
                None
            }
        })
        .collect();

    if non_zero_scores.len() > 1 {
        non_zero_scores.sort_unstable_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap().reverse());
    }
    non_zero_scores
}
