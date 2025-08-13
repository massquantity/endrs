use std::fs::{File, OpenOptions};
use std::io::{Read, Write};
use std::path::Path;

use flate2::read::GzDecoder;
use flate2::write::GzEncoder;
use flate2::Compression;
use pyo3::exceptions::PyIOError;
use pyo3::PyResult;
use serde::de::DeserializeOwned;
use serde::Serialize;

fn build_model_path(path: &str, model_name: &str) -> std::path::PathBuf {
    let file_name = format!("{model_name}.gz");
    Path::new(path).join(file_name)
}

pub fn save_model<T: Serialize>(
    model: &T,
    path: &str,
    model_name: &str,
    class_name: &str,
) -> PyResult<()> {
    let model_path = build_model_path(path, model_name);
    let file = OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .open(model_path.as_path())?;

    let model_bytes: Vec<u8> = bincode::serialize(model)
        .map_err(|e| PyIOError::new_err(format!("Failed to serialize {}: {}", class_name, e)))?;

    let mut encoder = GzEncoder::new(file, Compression::new(1));
    encoder.write_all(&model_bytes)?;
    encoder.finish()?;

    println!(
        "Save `{class_name}` model to `{}`",
        model_path.canonicalize()?.display()
    );
    Ok(())
}

pub fn load_model<T: DeserializeOwned>(
    path: &str,
    model_name: &str,
    class_name: &str,
) -> PyResult<T> {
    let model_path = build_model_path(path, model_name);
    if !model_path.exists() {
        return Err(PyIOError::new_err(format!(
            "Model file not found: {}",
            model_path.display()
        )));
    }

    let file = File::open(model_path.as_path())?;
    let mut decoder = GzDecoder::new(file);
    let mut model_bytes: Vec<u8> = Vec::new();
    decoder.read_to_end(&mut model_bytes)?;

    let model: T = bincode::deserialize(&model_bytes)
        .map_err(|e| PyIOError::new_err(format!("Failed to deserialize {}: {}", class_name, e)))?;

    println!(
        "Load `{class_name}` model from `{}`",
        model_path.canonicalize()?.display()
    );
    Ok(model)
}
