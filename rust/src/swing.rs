use fxhash::{FxHashMap, FxHashSet};
use pyo3::prelude::*;
use pyo3::types::*;
use serde::{Deserialize, Serialize};

use crate::graph::Graph;
use crate::inference::{compute_pred, get_intersect_neighbors, get_rec_items};
use crate::serialization::{load_model, save_model};
use crate::sparse::{get_row, CsrMatrix};
use crate::utils::{create_thread_pool, DEFAULT_PRED, OOV_IDX};

#[pyclass(module = "endrs_ext", name = "Swing")]
#[derive(Serialize, Deserialize)]
pub struct PySwing {
    top_k: usize,
    alpha: f32,
    max_cache_num: usize,
    n_users: usize,
    n_items: usize,
    graph: Graph,
    swing_score_mapping: FxHashMap<u32, Vec<(u32, f32)>>,
    user_interactions: CsrMatrix<u32, f32>,
    item_interactions: CsrMatrix<u32, f32>,
    user_consumed: FxHashMap<u32, Vec<u32>>,
}

#[pymethods]
impl PySwing {
    #[setter]
    fn set_n_users(&mut self, n_users: usize) {
        self.n_users = n_users;
        self.graph.n_users = n_users;
    }

    #[setter]
    fn set_n_items(&mut self, n_items: usize) {
        self.n_items = n_items;
        self.graph.n_items = n_items;
    }

    #[setter]
    fn set_user_consumed(&mut self, user_consumed: &Bound<'_, PyDict>) -> PyResult<()> {
        self.user_consumed = user_consumed.extract::<FxHashMap<u32, Vec<u32>>>()?;
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    #[new]
    fn new(
        top_k: usize,
        alpha: f32,
        max_cache_num: usize,
        n_users: usize,
        n_items: usize,
        user_interactions: &Bound<'_, PyAny>,
        item_interactions: &Bound<'_, PyAny>,
        user_consumed: &Bound<'_, PyDict>,
    ) -> PyResult<Self> {
        let user_consumed: FxHashMap<u32, Vec<u32>> = user_consumed.extract()?;
        let user_interactions: CsrMatrix<u32, f32> = user_interactions.extract()?;
        let item_interactions: CsrMatrix<u32, f32> = item_interactions.extract()?;
        let graph = Graph::new(n_users, n_items, alpha, max_cache_num);

        Ok(Self {
            top_k,
            alpha,
            max_cache_num,
            n_users,
            n_items,
            graph,
            swing_score_mapping: FxHashMap::default(),
            user_interactions,
            item_interactions,
            user_consumed,
        })
    }

    fn compute_swing(&mut self, num_threads: usize) -> PyResult<()> {
        let pool = create_thread_pool(num_threads)?;
        self.swing_score_mapping.clear();

        self.swing_score_mapping = pool.install(|| {
            self.graph.compute_swing_scores(
                &self.user_interactions,
                &self.item_interactions,
                &self.swing_score_mapping,
            )
        })?;
        Ok(())
    }

    /// update on new sparse interactions
    fn update_swing(
        &mut self,
        user_interactions: &Bound<'_, PyAny>,
        item_interactions: &Bound<'_, PyAny>,
        num_threads: usize,
    ) -> PyResult<()> {
        let pool = create_thread_pool(num_threads)?;
        let new_user_interactions: CsrMatrix<u32, f32> = user_interactions.extract()?;
        let new_item_interactions: CsrMatrix<u32, f32> = item_interactions.extract()?;

        self.swing_score_mapping = pool.install(|| {
            self.graph.compute_swing_scores(
                &new_user_interactions,
                &new_item_interactions,
                &self.swing_score_mapping,
            )
        })?;

        // merge interactions for inference on new users/items
        self.user_interactions = CsrMatrix::merge(
            &self.user_interactions,
            &new_user_interactions,
            self.n_users,
        );
        self.item_interactions = CsrMatrix::merge(
            &self.item_interactions,
            &new_item_interactions,
            self.n_items,
        );
        Ok(())
    }

    fn num_swing_elements(&self) -> PyResult<usize> {
        if self.swing_score_mapping.is_empty() {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                "call `compute_swing` method before calling `num_swing_elements`",
            ));
        }

        let n_elements = self
            .swing_score_mapping
            .values()
            .map(|i| i.len())
            .sum();

        Ok(n_elements)
    }

    fn predict(&self, users: &Bound<'_, PyList>, items: &Bound<'_, PyList>) -> PyResult<Vec<f32>> {
        let mut preds = Vec::new();
        let users: Vec<u32> = users.extract()?;
        let items: Vec<u32> = items.extract()?;

        for (&u, &i) in users.iter().zip(items.iter()) {
            if u == OOV_IDX || i == OOV_IDX {
                preds.push(DEFAULT_PRED);
                continue;
            }

            let pred = match (
                self.swing_score_mapping.get(&i),
                get_row(&self.user_interactions, u as usize, false),
            ) {
                (Some(item_swings), Some(item_labels)) => {
                    let num = self.top_k.min(item_swings.len());
                    let mut item_swing_scores = vec![(0, 0.0); num];
                    item_swing_scores.clone_from_slice(&item_swings[..num]);
                    item_swing_scores.sort_unstable_by_key(|&(i, _)| i);

                    let item_labels: Vec<(u32, f32)> = item_labels.collect();
                    let (k_nb_swings, k_nb_labels) =
                        get_intersect_neighbors(&item_swing_scores, &item_labels, self.top_k);

                    if k_nb_swings.is_empty() {
                        DEFAULT_PRED
                    } else {
                        compute_pred("ranking", &k_nb_swings, &k_nb_labels)?
                    }
                }
                _ => DEFAULT_PRED,
            };

            preds.push(pred);
        }

        Ok(preds)
    }

    fn recommend<'py>(
        &self,
        py: Python<'py>,
        users: &Bound<'py, PyList>,
        n_rec: usize,
        filter_consumed: bool,
        random_rec: bool,
    ) -> PyResult<(Vec<Bound<'py, PyList>>, Bound<'py, PyList>)> {
        let mut recs = Vec::new();
        let mut additional_rec_counts = Vec::new();

        for u in users {
            let u: u32 = u.extract()?;
            let consumed: FxHashSet<u32> = if filter_consumed {
                self.user_consumed
                    .get(&u)
                    .map(|v| v.iter().copied().collect())
                    .unwrap_or_default()
            } else {
                FxHashSet::default()
            };

            let (rec_items, additional_count) =
                if let Some(row) = get_row(&self.user_interactions, u as usize, false) {
                    let mut item_scores: FxHashMap<u32, f32> = FxHashMap::default();
                    for (i, i_label) in row {
                        if let Some(item_swings) = self.swing_score_mapping.get(&i) {
                            let num = self.top_k.min(item_swings.len());
                            for &(j, i_j_swing_score) in &item_swings[..num] {
                                if filter_consumed && consumed.contains(&j) {
                                    continue;
                                }

                                item_scores
                                    .entry(j)
                                    .and_modify(|score| *score += i_j_swing_score * i_label)
                                    .or_insert(i_j_swing_score * i_label);
                            }
                        }
                    }

                    if item_scores.is_empty() {
                        (PyList::empty(py), n_rec)
                    } else {
                        let items = get_rec_items(item_scores, n_rec, random_rec);
                        let additional = n_rec - items.len();
                        (PyList::new(py, items)?, additional)
                    }
                } else {
                    (PyList::empty(py), n_rec)
                };

            recs.push(rec_items);
            additional_rec_counts.push(additional_count);
        }

        let additional_rec_counts = PyList::new(py, additional_rec_counts)?;
        Ok((recs, additional_rec_counts))
    }
}

#[pyfunction]
#[pyo3(name = "save_swing")]
pub fn save(model: &PySwing, path: &str, model_name: &str) -> PyResult<()> {
    let class_name = "Swing";
    save_model(model, path, model_name, class_name)?;
    Ok(())
}

#[pyfunction]
#[pyo3(name = "load_swing")]
pub fn load(path: &str, model_name: &str) -> PyResult<PySwing> {
    let class_name = "Swing";
    let model = load_model(path, model_name, class_name)?;
    Ok(model)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[pyclass]
    struct PySparseMatrix {
        #[pyo3(get)]
        sparse_indices: Vec<u32>,
        #[pyo3(get)]
        sparse_indptr: Vec<usize>,
        #[pyo3(get)]
        sparse_data: Vec<f32>,
    }

    fn get_swing_model() -> Result<PySwing, Box<dyn std::error::Error>> {
        let top_k = 10;
        let alpha = 1.0;
        let cache_common_num = 100;
        let n_users = 3;
        let n_items = 5;
        let swing = Python::with_gil(|py| -> PyResult<PySwing> {
            // item_interactions:
            // [
            //     [0, 0, 0, 0],
            //     [0, 1, 1, 1],
            //     [0, 1, 1, 0],
            //     [0, 1, 0, 1],
            //     [0, 1, 1, 1],
            //     [0, 0, 0, 1],
            // ]
            let item_interactions = Bound::new(
                py,
                PySparseMatrix {
                    sparse_indices: vec![1, 2, 3, 1, 2, 1, 3, 1, 2, 3, 3],
                    sparse_indptr: vec![0, 0, 3, 5, 7, 10, 11],
                    sparse_data: vec![1.0; 11],
                },
            )?;
            let user_interactions = Bound::new(
                py,
                PySparseMatrix {
                    sparse_indices: vec![1, 2, 3, 4, 1, 2, 4, 1, 3, 4, 5],
                    sparse_indptr: vec![0, 0, 4, 7, 11],
                    sparse_data: vec![1.0; 11],
                },
            )?;
            let user_consumed = [
                (1u32, vec![1u32, 2, 3, 4]),
                (2, vec![1, 2, 4]),
                (3, vec![1, 3, 4, 5]),
            ]
            .into_py_dict(py)?;

            let mut swing = PySwing::new(
                top_k,
                alpha,
                cache_common_num,
                n_users,
                n_items,
                &user_interactions,
                &item_interactions,
                &user_consumed,
            )?;
            swing.compute_swing(2)?;
            Ok(swing)
        })?;
        Ok(swing)
    }

    #[test]
    fn test_swing_training() -> Result<(), Box<dyn std::error::Error>> {
        pyo3::prepare_freethreaded_python();

        let test_user_id = 1u32;
        let match_item_fn = |model: &PySwing, p: usize, i: u32, s: f32| {
            let (item, score) = model.swing_score_mapping[&test_user_id][p];
            item == i && (score - s).abs() < 1e-10
        };

        let user_weights = [
            0.0,
            4_f32.sqrt().recip(),
            3_f32.sqrt().recip(),
            4_f32.sqrt().recip(),
        ];
        let common_nums = [2.0, 2.0, 1.0]; // user_1_2, user_1_3, user_2_3, k = len(common) - 1;
        let swing_1_2 = user_weights[1] * user_weights[2] * (1_f32 + common_nums[0]).recip();
        let swing_1_3 = user_weights[1] * user_weights[3] * (1_f32 + common_nums[1]).recip();
        let swing_1_4 = user_weights[1] * user_weights[2] * (1_f32 + common_nums[0]).recip()
            + user_weights[1] * user_weights[3] * (1_f32 + common_nums[1]).recip()
            + user_weights[2] * user_weights[3] * (1_f32 + common_nums[2]).recip();
        let swing_model = get_swing_model()?;

        assert_eq!(swing_model.swing_score_mapping[&test_user_id].len(), 3);
        assert!(match_item_fn(&swing_model, 0, 4, swing_1_4));
        assert!(match_item_fn(&swing_model, 1, 2, swing_1_2));
        assert!(match_item_fn(&swing_model, 2, 3, swing_1_3));
        Ok(())
    }

    #[test]
    fn test_save_model() -> Result<(), Box<dyn std::error::Error>> {
        pyo3::prepare_freethreaded_python();

        let model = get_swing_model()?;
        let cur_dir = std::env::current_dir()?
            .to_string_lossy()
            .to_string();
        let model_name = "swing_model";
        save(&model, &cur_dir, model_name)?;

        let new_model: PySwing = load(&cur_dir, model_name)?;
        Python::with_gil(|py| -> PyResult<()> {
            let users = PyList::new(py, vec![5, 1])?;
            let rec_result = new_model.recommend(py, &users, 10, true, false)?;
            assert_eq!(rec_result.0.len(), 2);
            Ok(())
        })?;

        std::fs::remove_file(std::env::current_dir()?.join(format!("{model_name}.gz")))?;
        Ok(())
    }
}
