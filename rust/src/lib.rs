use pyo3::prelude::*;

mod consumed;
mod graph;
mod incremental;
mod inference;
mod item_cf;
mod serialization;
mod similarities;
mod sparse;
mod swing;
mod user_cf;
mod utils;

const VERSION: &str = env!("CARGO_PKG_VERSION");

/// A Python module implemented in Rust. The name of this function must match
/// the `lib.name` setting in the `Cargo.toml`, else Python will not be able to
/// import the module.
#[pymodule]
#[pyo3(name = "endrs_ext")]
fn endrs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<swing::PySwing>()?;
    m.add_function(wrap_pyfunction!(swing::save, m)?)?;
    m.add_function(wrap_pyfunction!(swing::load, m)?)?;

    m.add_class::<user_cf::PyUserCF>()?;
    m.add_function(wrap_pyfunction!(user_cf::save, m)?)?;
    m.add_function(wrap_pyfunction!(user_cf::load, m)?)?;

    m.add_class::<item_cf::PyItemCF>()?;
    m.add_function(wrap_pyfunction!(item_cf::save, m)?)?;
    m.add_function(wrap_pyfunction!(item_cf::load, m)?)?;

    m.add("__version__", VERSION)?;
    Ok(())
}
