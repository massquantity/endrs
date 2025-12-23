use std::cmp::Ordering;
use std::time::Instant;

use fxhash::FxHashMap;
use pyo3::PyResult;
use rayon::prelude::*;

use crate::sparse::{get_row, CsrMatrix};
use crate::utils::{decode_pair, encode_pair, CumValues, SimVals};

const BATCH_SIZE: usize = 1000;

pub(crate) fn compute_sum_squares(interactions: &CsrMatrix<u32, f32>, num: usize) -> Vec<f32> {
    let mut sum_squares = vec![0.0; num + 1];
    // skip oov 0
    for (i, sum_sq) in sum_squares.iter_mut().enumerate().skip(1) {
        *sum_sq = get_row(interactions, i, false)
            .map_or(0.0, |row| row.fold(0.0, |ss, (_, d)| ss + d * d))
    }
    sum_squares
}

pub(crate) fn compute_cosine(prod: f32, sum_squ1: f32, sum_squ2: f32) -> f32 {
    if prod == 0.0 || sum_squ1 == 0.0 || sum_squ2 == 0.0 {
        0.0
    } else {
        let norm = sum_squ1.sqrt() * sum_squ2.sqrt();
        prod / norm
    }
}

fn compute_row_sims(
    interactions: &CsrMatrix<u32, f32>,
    sum_squares: &[f32],
    n: usize,
    x1: usize,
) -> Vec<SimVals> {
    let (indices, indptr, data) = interactions.values();
    let mut sims = Vec::new();
    for x2 in (x1 + 1)..=n {
        let mut i = indptr[x1];
        let mut j = indptr[x2];
        let end1 = indptr[x1 + 1];
        let end2 = indptr[x2 + 1];
        let mut prod = 0.0;
        let mut count = 0u32;
        while i < end1 && j < end2 {
            let y1 = indices[i];
            let y2 = indices[j];
            match y1.cmp(&y2) {
                Ordering::Less => i += 1,
                Ordering::Greater => j += 1,
                Ordering::Equal => {
                    prod += data[i] * data[j];
                    count += 1;
                    i += 1;
                    j += 1;
                }
            }
        }

        let cosine = compute_cosine(prod, sum_squares[x1], sum_squares[x2]);
        let sim = SimVals {
            x1: x1 as u32,
            x2: x2 as u32,
            prod,
            count,
            cosine,
        };

        sims.push(sim);
    }
    sims
}

pub(crate) fn forward_cosine(
    interactions: &CsrMatrix<u32, f32>,
    sum_squares: &[f32],
    cum_values: &mut FxHashMap<u64, CumValues>,
    n: usize,
    min_common: u32,
) -> PyResult<Vec<(u32, u32, f32)>> {
    let start = Instant::now();
    let sim_vals: Vec<SimVals> = (1..=n)
        .into_par_iter()
        .flat_map(|x| compute_row_sims(interactions, sum_squares, n, x))
        .collect();

    let mut cosine_sims: Vec<(u32, u32, f32)> = Vec::new();

    for SimVals {
        x1,
        x2,
        prod,
        count,
        cosine,
    } in sim_vals
    {
        if count >= min_common {
            cosine_sims.push((x1, x2, cosine));
        }
        if count > 0 {
            let key = encode_pair(x1, x2);
            cum_values.insert(key, (prod, count));
        }
    }

    let duration = start.elapsed();
    println!(
        "forward cosine sim: {} elapsed: {:.4?}",
        cosine_sims.len(),
        duration
    );

    Ok(cosine_sims)
}

#[inline]
fn update_pair_stats(acc: &mut FxHashMap<u64, CumValues>, x1: u32, x2: u32, prod_val: f32) {
    let key = encode_pair(x1, x2);
    acc.entry(key)
        .and_modify(|(p, c)| {
            *p += prod_val;
            *c += 1;
        })
        .or_insert((prod_val, 1));
}

/// Process item pairs in a single row with batching for long rows
fn process_row_batched(acc: &mut FxHashMap<u64, CumValues>, row_indices: &[u32], row_data: &[f32]) {
    let row_len = row_indices.len();
    // ceiling division
    let num_batches = (row_len + BATCH_SIZE - 1) / BATCH_SIZE;

    for batch_a in 0..num_batches {
        let start_a = batch_a * BATCH_SIZE;
        let end_a = (start_a + BATCH_SIZE).min(row_len);

        // Pairs within batch A
        for i in start_a..end_a {
            for j in (i + 1)..end_a {
                update_pair_stats(
                    acc,
                    row_indices[i],
                    row_indices[j],
                    row_data[i] * row_data[j],
                );
            }
        }

        // Pairs between batch A and subsequent batches
        for batch_b in (batch_a + 1)..num_batches {
            let start_b = batch_b * BATCH_SIZE;
            let end_b = (start_b + BATCH_SIZE).min(row_len);
            for i in start_a..end_a {
                for j in start_b..end_b {
                    update_pair_stats(
                        acc,
                        row_indices[i],
                        row_indices[j],
                        row_data[i] * row_data[j],
                    );
                }
            }
        }
    }
}

pub(crate) fn compute_pair_stats(
    interactions: &CsrMatrix<u32, f32>,
    n: usize,
) -> FxHashMap<u64, CumValues> {
    let (indices, indptr, data) = interactions.values();

    (1..=n)
        .into_par_iter()
        .fold(FxHashMap::default, |mut acc, i| {
            let row_start = indptr[i];
            let row_end = indptr[i + 1];
            if row_end - row_start < 2 {
                return acc;
            }
            let row_indices = &indices[row_start..row_end];
            let row_data = &data[row_start..row_end];
            process_row_batched(&mut acc, row_indices, row_data);
            acc
        })
        .reduce(FxHashMap::default, |mut a, b| {
            for (k, (p, c)) in b {
                a.entry(k)
                    .and_modify(|(ap, ac)| {
                        *ap += p;
                        *ac += c;
                    })
                    .or_insert((p, c));
            }
            a
        })
}

pub(crate) fn invert_cosine(
    interactions: &CsrMatrix<u32, f32>,
    sum_squares: &[f32],
    cum_values: &mut FxHashMap<u64, CumValues>,
    n: usize,
    min_common: u32,
) -> PyResult<Vec<(u32, u32, f32)>> {
    let start = Instant::now();
    let pair_stats = compute_pair_stats(interactions, n);
    let mut cosine_sims = Vec::new();

    for (key, (prod, count)) in pair_stats {
        if count > 0 {
            cum_values.insert(key, (prod, count));
        }
        if count >= min_common {
            let (x1, x2) = decode_pair(key);
            let cosine = compute_cosine(prod, sum_squares[x1 as usize], sum_squares[x2 as usize]);
            cosine_sims.push((x1, x2, cosine));
        }
    }

    let duration = start.elapsed();
    println!(
        "invert cosine sim: {} elapsed: {:.4?}",
        cosine_sims.len(),
        duration
    );
    Ok(cosine_sims)
}

pub(crate) fn sort_by_sims(
    n: usize,
    cosine_sims: &[(u32, u32, f32)],
    sim_mapping: &mut FxHashMap<u32, (Vec<u32>, Vec<f32>)>,
) -> PyResult<()> {
    let start = Instant::now();
    let mut agg_sims: Vec<Vec<(u32, f32)>> = vec![Vec::new(); n + 1];

    for &(x1, x2, sim) in cosine_sims {
        agg_sims[x1 as usize].push((x2, sim));
        agg_sims[x2 as usize].push((x1, sim));
    }

    for (i, neighbor_sims) in agg_sims.iter_mut().enumerate().skip(1) {
        if neighbor_sims.is_empty() {
            continue;
        }

        neighbor_sims.sort_unstable_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap().reverse());

        let (neighbors, sims) = neighbor_sims
            .iter()
            .map(|(n, s)| (*n, *s))
            .unzip();

        sim_mapping.insert(i as u32, (neighbors, sims));
    }

    let duration = start.elapsed();
    println!("sort elapsed: {duration:.4?}");
    Ok(())
}
