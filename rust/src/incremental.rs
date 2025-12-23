use std::time::Instant;

use fxhash::FxHashMap;
use pyo3::PyResult;

use crate::similarities::{compute_cosine, compute_pair_stats};
use crate::sparse::{get_row, CsrMatrix};
use crate::utils::{decode_pair, CumValues};

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

pub(crate) fn update_by_sims(
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

    for (i, new_neighbor_sims) in agg_sims.into_iter().enumerate().skip(1) {
        if new_neighbor_sims.is_empty() {
            continue;
        }

        let key = i as u32;
        let mut combined_sims: Vec<(u32, f32)> = if let Some((n, s)) = sim_mapping.get(&key) {
            let pairs = n.iter().zip(s.iter()).map(|(a, b)| (*a, *b));
            let mut original_sims: FxHashMap<u32, f32> = FxHashMap::from_iter(pairs);
            original_sims.extend(new_neighbor_sims); // replace old sims with new ones in map
            original_sims.into_iter().collect()
        } else {
            new_neighbor_sims
        };

        combined_sims.sort_unstable_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap().reverse());
        sim_mapping.insert(key, combined_sims.into_iter().unzip());
    }

    let duration = start.elapsed();
    println!("incremental sort elapsed: {duration:.4?}");
    Ok(())
}
