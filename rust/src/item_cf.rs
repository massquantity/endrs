use fxhash::FxHashMap;
use pyo3::prelude::*;
use pyo3::types::*;
use serde::{Deserialize, Serialize};

use crate::incremental::{update_by_sims, update_cosine, update_sum_squares};
use crate::inference::{predict_single, recommend_by_item_sims};
use crate::serialization::{load_model, save_model};
use crate::similarities::{compute_sum_squares, forward_cosine, invert_cosine, sort_by_sims};
use crate::sparse::{get_row, CsrMatrix};
use crate::utils::{CumValues, Neighbor, DEFAULT_PRED, OOV_IDX};

#[pyclass(module = "recfarm", name = "ItemCF")]
#[derive(Serialize, Deserialize)]
pub struct PyItemCF {
    task: String,
    k_sim: usize,
    n_users: usize,
    n_items: usize,
    min_common: u32,
    sum_squares: Vec<f32>,
    cum_values: FxHashMap<u64, CumValues>,
    item_sims: FxHashMap<u32, Vec<Neighbor>>,
    user_interactions: CsrMatrix<u32, f32>,
    item_interactions: CsrMatrix<u32, f32>,
    user_consumed: FxHashMap<u32, Vec<u32>>,
    use_inverted_index: bool,
}

#[pymethods]
impl PyItemCF {
    #[setter]
    fn set_n_users(&mut self, n_users: usize) {
        self.n_users = n_users;
    }

    #[setter]
    fn set_n_items(&mut self, n_items: usize) {
        self.n_items = n_items;
    }

    #[setter]
    fn set_user_consumed(&mut self, user_consumed: &Bound<'_, PyDict>) -> PyResult<()> {
        self.user_consumed = user_consumed.extract::<FxHashMap<u32, Vec<u32>>>()?;
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    #[new]
    fn new(
        task: &str,
        k_sim: usize,
        n_users: usize,
        n_items: usize,
        min_common: u32,
        user_interactions: &Bound<'_, PyAny>,
        item_interactions: &Bound<'_, PyAny>,
        user_consumed: &Bound<'_, PyDict>,
    ) -> PyResult<Self> {
        let user_consumed: FxHashMap<u32, Vec<u32>> = user_consumed.extract()?;
        let user_interactions: CsrMatrix<u32, f32> = user_interactions.extract()?;
        let item_interactions: CsrMatrix<u32, f32> = item_interactions.extract()?;
        Ok(Self {
            task: task.to_string(),
            k_sim,
            n_users,
            n_items,
            min_common,
            sum_squares: Vec::new(),
            cum_values: FxHashMap::default(),
            item_sims: FxHashMap::default(),
            user_interactions,
            item_interactions,
            user_consumed,
            use_inverted_index: true,
        })
    }

    /// forward index: sparse matrix of `item` interactions
    /// invert index: sparse matrix of `user` interactions
    fn compute_similarities(&mut self, num_threads: usize) -> PyResult<()> {
        self.sum_squares = compute_sum_squares(&self.item_interactions, self.n_items);

        let cosine_sims = if self.use_inverted_index {
            invert_cosine(
                &self.user_interactions,
                &self.sum_squares,
                &mut self.cum_values,
                self.n_users,
                self.min_common,
                num_threads,
            )?
        } else {
            forward_cosine(
                &self.item_interactions,
                &self.sum_squares,
                &mut self.cum_values,
                self.n_items,
                self.min_common,
                num_threads,
            )?
        };

        sort_by_sims(&cosine_sims, &mut self.item_sims)?;

        Ok(())
    }

    /// update on new sparse interactions
    fn update_similarities(
        &mut self,
        user_interactions: &Bound<'_, PyAny>,
        item_interactions: &Bound<'_, PyAny>,
        num_threads: usize,
    ) -> PyResult<()> {
        let new_user_interactions: CsrMatrix<u32, f32> = user_interactions.extract()?;
        let new_item_interactions: CsrMatrix<u32, f32> = item_interactions.extract()?;

        update_sum_squares(&mut self.sum_squares, &new_item_interactions, self.n_items);

        let cosine_sims = update_cosine(
            &new_user_interactions,
            &self.sum_squares,
            &mut self.cum_values,
            self.n_users,
            self.min_common,
            num_threads,
        )?;

        update_by_sims(&cosine_sims, &mut self.item_sims)?;

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

    fn num_sim_elements(&self) -> PyResult<usize> {
        let n_elements = self.item_sims.values().map(|v| v.len()).sum();
        Ok(n_elements)
    }

    /// sparse matrix of `user` interactions
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
                self.item_sims.get(&i),
                get_row(&self.user_interactions, u as usize, false),
            ) {
                (Some(neighbors), Some(interactions)) => {
                    predict_single(neighbors, interactions, self.k_sim, &self.task)?
                }
                _ => DEFAULT_PRED,
            };

            preds.push(pred);
        }
        Ok(preds)
    }

    /// sparse matrix of `user` interaction
    fn recommend<'py>(
        &self,
        py: Python<'py>,
        users: &Bound<'py, PyList>,
        n_rec: usize,
        filter_consumed: bool,
        random_rec: bool,
    ) -> PyResult<(Vec<Bound<'py, PyList>>, Bound<'py, PyList>)> {
        recommend_by_item_sims(
            py,
            users,
            n_rec,
            filter_consumed,
            random_rec,
            self.k_sim,
            &self.user_consumed,
            &self.user_interactions,
            &self.item_sims,
        )
    }
}

#[pyfunction]
#[pyo3(name = "save_item_cf")]
pub fn save(model: &PyItemCF, path: &str, model_name: &str) -> PyResult<()> {
    save_model(model, path, model_name, "ItemCF")?;
    Ok(())
}

#[pyfunction]
#[pyo3(name = "load_item_cf")]
pub fn load(path: &str, model_name: &str) -> PyResult<PyItemCF> {
    let model = load_model(path, model_name, "ItemCF")?;
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

    fn get_item_cf() -> Result<PyItemCF, Box<dyn std::error::Error>> {
        let task = "ranking";
        let k_sim = 10;
        let n_items = 5;
        let n_users = 4;
        let min_common = 1u32;
        let item_cf = Python::with_gil(|py| -> PyResult<PyItemCF> {
            // item_interactions (6 rows: OOV + items 1-5, 5 cols: OOV + users 1-4):
            // [
            //     [],               // row 0: OOV
            //     [0, 1, 1, 0, 0],  // row 1 (item 1)
            //     [0, 2, 1, 0, 0],  // row 2 (item 2)
            //     [0, 0, 1, 1, 0],  // row 3 (item 3)
            //     [0, 2, 1, 1, 0],  // row 4 (item 4)
            //     [0, 0, 1, 2, 0],  // row 5 (item 5)
            // ]
            let item_interactions = Bound::new(
                py,
                PySparseMatrix {
                    sparse_indices: vec![1, 2, 1, 2, 2, 3, 1, 2, 3, 2, 3],
                    sparse_indptr: vec![0, 0, 2, 4, 6, 9, 11],
                    sparse_data: vec![1., 1., 2., 1., 1., 1., 2., 1., 1., 1., 2.],
                },
            )?;
            // user_interactions (5 rows: OOV + users 1-4, 6 cols: OOV + items 1-5):
            let user_interactions = Bound::new(
                py,
                PySparseMatrix {
                    sparse_indices: vec![1, 2, 4, 1, 2, 3, 4, 5, 3, 4, 5],
                    sparse_indptr: vec![0, 0, 3, 8, 11, 11],
                    sparse_data: vec![1., 2., 2., 1., 1., 1., 1., 1., 1., 1., 2.],
                },
            )?;
            let user_consumed = [
                (1u32, vec![1u32, 2]),
                (2, vec![1, 2]),
                (3, vec![2, 3]),
                (4, vec![1, 2, 3]),
                (5, vec![2, 3]),
            ]
            .into_py_dict(py)?;

            let mut item_cf = PyItemCF::new(
                task,
                k_sim,
                n_users,
                n_items,
                min_common,
                &user_interactions,
                &item_interactions,
                &user_consumed,
            )?;
            item_cf.compute_similarities(1)?;
            Ok(item_cf)
        })?;
        Ok(item_cf)
    }

    #[test]
    fn test_item_cf_training() -> Result<(), Box<dyn std::error::Error>> {
        let get_nbs = |model: &PyItemCF, i: u32| -> Vec<u32> {
            model.item_sims[&i].iter().map(|n| n.id).collect()
        };
        pyo3::prepare_freethreaded_python();
        let item_cf = get_item_cf()?;
        assert_eq!(get_nbs(&item_cf, 1), vec![2, 4, 3, 5]);
        assert_eq!(get_nbs(&item_cf, 2), vec![1, 4, 3, 5]);
        assert_eq!(get_nbs(&item_cf, 3), vec![5, 4, 1, 2]);
        assert_eq!(get_nbs(&item_cf, 4), vec![2, 1, 3, 5]);
        assert_eq!(get_nbs(&item_cf, 5), vec![3, 4, 1, 2]);
        Ok(())
    }

    #[test]
    fn test_item_cf_incremental_training() -> Result<(), Box<dyn std::error::Error>> {
        let get_nbs = |model: &PyItemCF, u: u32| -> Vec<u32> {
            model.item_sims[&u].iter().map(|n| n.id).collect()
        };
        pyo3::prepare_freethreaded_python();
        let mut item_cf = get_item_cf()?;
        Python::with_gil(|py| -> PyResult<()> {
            // larger item_interactions (7 rows: OOV + items 1-6, 6 cols: OOV + users 1-5):
            // [
            //     [],                    // row 0: OOV
            //     [],                    // row 1 (item 1)
            //     [0, 3, 0, 0, 0, 0],    // row 2 (item 2)
            //     [0, 5, 0, 0, 0, 0],    // row 3 (item 3)
            //     [],                    // row 4 (item 4)
            //     [],                    // row 5 (item 5)
            //     [0, 2, 2, 1, 2, 0],    // row 6 (item 6)
            // ]
            let item_interactions = Bound::new(
                py,
                PySparseMatrix {
                    sparse_indices: vec![1, 1, 1, 2, 3, 4],
                    sparse_indptr: vec![0, 0, 0, 1, 2, 2, 2, 6],
                    sparse_data: vec![3.0, 5.0, 2.0, 2.0, 1.0, 2.0],
                },
            )?;
            // user_interactions (6 rows: OOV + users 1-5, 7 cols: OOV + items 1-6):
            let user_interactions = Bound::new(
                py,
                PySparseMatrix {
                    sparse_indices: vec![2, 3, 6, 6, 6, 6],
                    sparse_indptr: vec![0, 0, 3, 4, 5, 6, 6],
                    sparse_data: vec![3.0, 5.0, 2.0, 2.0, 1.0, 2.0],
                },
            )?;
            let _user_consumed = [
                (1u32, vec![1u32, 2]),
                (2, vec![1, 2]),
                (3, vec![2, 3]),
                (4, vec![1, 2, 3]),
                (5, vec![2, 3]),
                (6, vec![1, 2, 3, 4]),
            ]
            .into_py_dict(py)?;

            item_cf.n_items = 6;
            item_cf.n_users = 5;
            item_cf.user_consumed = _user_consumed.extract::<FxHashMap<u32, Vec<u32>>>()?;
            item_cf.update_similarities(&user_interactions, &item_interactions, 1)?;
            let users = PyList::new(py, vec![6, 2])?;
            let rec_result = item_cf.recommend(py, &users, 10, true, false)?;
            assert_eq!(rec_result.0.len(), 2);
            Ok(())
        })?;
        assert_eq!(get_nbs(&item_cf, 1), vec![2, 4, 3, 5]);
        assert_eq!(get_nbs(&item_cf, 2), vec![1, 4, 3, 6, 5]);
        assert_eq!(get_nbs(&item_cf, 3), vec![5, 2, 4, 6, 1]);
        assert_eq!(get_nbs(&item_cf, 4), vec![2, 1, 3, 5]);
        assert_eq!(get_nbs(&item_cf, 5), vec![3, 4, 1, 2]);
        assert_eq!(get_nbs(&item_cf, 6), vec![3, 2]);

        Python::with_gil(|py| -> PyResult<()> {
            // smaller item_interactions (5 rows: OOV + items 1-4, 6 cols: OOV + users 1-5):
            // [
            //     [],                    // row 0: OOV
            //     [0, 0, 0, 0, 3, 2],    // row 1 (item 1)
            //     [],                    // row 2 (item 2)
            //     [],                    // row 3 (item 3)
            //     [0, 0, 1, 0, 4, 3],    // row 4 (item 4)
            // ]
            let item_interactions = Bound::new(
                py,
                PySparseMatrix {
                    sparse_indices: vec![4, 5, 2, 4, 5],
                    sparse_indptr: vec![0, 0, 2, 2, 2, 5],
                    sparse_data: vec![3.0, 2.0, 1.0, 4.0, 3.0],
                },
            )?;
            // user_interactions (6 rows: OOV + users 1-5, 5 cols: OOV + items 1-4):
            let user_interactions = Bound::new(
                py,
                PySparseMatrix {
                    sparse_indices: vec![4, 1, 4, 1, 4],
                    sparse_indptr: vec![0, 0, 0, 1, 1, 3, 5],
                    sparse_data: vec![1.0, 3.0, 4.0, 2.0, 3.0],
                },
            )?;
            item_cf.update_similarities(&user_interactions, &item_interactions, 1)?;
            Ok(())
        })?;
        assert_eq!(get_nbs(&item_cf, 1), vec![4, 2, 3, 5]);
        assert_eq!(get_nbs(&item_cf, 2), vec![1, 4, 3, 6, 5]);
        assert_eq!(get_nbs(&item_cf, 3), vec![5, 2, 4, 6, 1]);
        assert_eq!(get_nbs(&item_cf, 4), vec![1, 2, 3, 5]);
        assert_eq!(get_nbs(&item_cf, 5), vec![3, 4, 1, 2]);
        assert_eq!(get_nbs(&item_cf, 6), vec![3, 2]);
        Ok(())
    }

    #[test]
    fn test_save_model() -> Result<(), Box<dyn std::error::Error>> {
        pyo3::prepare_freethreaded_python();
        let model = get_item_cf()?;
        let cur_dir = std::env::current_dir()?
            .to_string_lossy()
            .to_string();
        let model_name = "item_cf_model";
        save(&model, &cur_dir, model_name)?;

        let new_model: PyItemCF = load(&cur_dir, model_name)?;
        Python::with_gil(|py| -> PyResult<()> {
            let users = PyList::new(py, vec![8, 2])?;
            let rec_result = new_model.recommend(py, &users, 10, true, false)?;
            assert_eq!(rec_result.0.len(), 2);
            Ok(())
        })?;

        std::fs::remove_file(std::env::current_dir()?.join(format!("{model_name}.gz")))?;
        Ok(())
    }
}
