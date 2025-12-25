use std::cmp::Ordering;
use std::sync::Arc;

use dashmap::DashMap;
use fxhash::FxHashMap;
use pyo3::PyResult;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::sparse::{get_row, CsrMatrix};
use crate::utils::{Neighbor, SIM_EPS};

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
        user_interactions: &CsrMatrix<u32, f32>,
        item_interactions: &CsrMatrix<u32, f32>,
        prev_sims: &FxHashMap<u32, Vec<Neighbor>>,
        user_weights: &[f32],
        cached_common_items: Arc<DashMap<u64, Vec<usize>>>,
    ) -> (u32, Vec<Neighbor>) {
        let target_u32 = target_item as u32;
        let users = get_row_vec(item_interactions, target_item);
        if users.len() < 2 {
            let sims = prev_sims
                .get(&target_u32)
                .cloned()
                .unwrap_or_default();
            return (target_u32, sims);
        }

        let mut swing_sims = init_swing_sims(target_u32, self.n_items, prev_sims);
        for (j, &u) in users.iter().enumerate() {
            for &v in &users[(j + 1)..users.len()] {
                let key = (u as u64) * (self.n_users as u64) + (v as u64);
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
                        swing_sims[i] += score;
                    }
                }
            }
        }

        (target_u32, extract_valid_sims(swing_sims))
    }

    pub(crate) fn compute_swing_sims(
        &self,
        user_interactions: &CsrMatrix<u32, f32>,
        item_interactions: &CsrMatrix<u32, f32>,
        prev_sims: &FxHashMap<u32, Vec<Neighbor>>,
    ) -> PyResult<FxHashMap<u32, Vec<Neighbor>>> {
        let user_weights = compute_user_weights(user_interactions, self.n_users);
        let cached_common_items: Arc<DashMap<u64, Vec<usize>>> = Arc::new(DashMap::new());
        let swing_sims: Vec<(u32, Vec<Neighbor>)> = (1..=self.n_items)
            .into_par_iter()
            .filter_map(|i| {
                let (item, sims) = self.compute_single_swing(
                    i,
                    user_interactions,
                    item_interactions,
                    prev_sims,
                    &user_weights,
                    Arc::clone(&cached_common_items),
                );

                if sims.is_empty() {
                    None
                } else {
                    Some((item, sims))
                }
            })
            .collect();

        Ok(FxHashMap::from_iter(swing_sims))
    }
}

fn compute_user_weights(matrix: &CsrMatrix<u32, f32>, n_users: usize) -> Vec<f32> {
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

fn get_row_vec(matrix: &CsrMatrix<u32, f32>, n: usize) -> Vec<usize> {
    if let Some(row) = get_row(matrix, n, false) {
        row.map(|(i, _)| i as usize).collect()
    } else {
        Vec::new()
    }
}

fn init_swing_sims(
    target_item: u32,
    n_items: usize,
    prev_sims: &FxHashMap<u32, Vec<Neighbor>>,
) -> Vec<f32> {
    let mut swing_sims = vec![0.0; n_items + 1];
    if let Some(sims) = prev_sims.get(&target_item) {
        for nb in sims {
            swing_sims[nb.id as usize] = nb.sim;
        }
    }
    swing_sims
}

fn extract_valid_sims(sims: Vec<f32>) -> Vec<Neighbor> {
    let mut non_zero_sims: Vec<Neighbor> = sims
        .into_iter()
        .enumerate()
        .filter_map(|(i, sim)| {
            if sim.abs() > SIM_EPS {
                Some(Neighbor { id: i as u32, sim })
            } else {
                None
            }
        })
        .collect();

    if non_zero_sims.len() > 1 {
        non_zero_sims.sort_unstable_by(|a, b| a.sim.partial_cmp(&b.sim).unwrap().reverse());
    }
    non_zero_sims
}
