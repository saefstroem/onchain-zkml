extern crate alloc;

mod error;
mod nn;
mod result;

pub use error::Error;
pub use nn::{forward, infer, layer_count, net_commitment};
pub use result::Result;


use alloc::vec::Vec;
use sha2::{Digest, Sha256};
use serde::{Deserialize, Serialize};

static NORM: &[u8] = include_bytes!("norm.bin");

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Witness {
    pub input: Vec<f32>,
    pub layer: u8,
}

fn norm_f32(at: usize) -> Result<f32> {
    Ok(f32::from_le_bytes([
        *NORM.get(at).ok_or(Error::Net)?,
        *NORM.get(at + 1).ok_or(Error::Net)?,
        *NORM.get(at + 2).ok_or(Error::Net)?,
        *NORM.get(at + 3).ok_or(Error::Net)?,
    ]))
}

pub fn feature_count() -> Result<usize> {
    Ok(u16::from_le_bytes([*NORM.get(0).ok_or(Error::Net)?, *NORM.get(1).ok_or(Error::Net)?]) as usize)
}

pub fn standardize(features: &[f32]) -> Result<Vec<f32>> {
    let n = feature_count()?;
    if features.len() != n {
        return Err(Error::Shape);
    }
    let mut out = Vec::with_capacity(n);
    for (i, x) in features.iter().enumerate() {
        let mean = norm_f32(2 + i * 4)?;
        let std = norm_f32(2 + n * 4 + i * 4)?;
        out.push((x - mean) / std);
    }
    Ok(out)
}

pub fn target_stats() -> Result<(f32, f32)> {
    let n = feature_count()?;
    Ok((norm_f32(2 + n * 8)?, norm_f32(2 + n * 8 + 4)?))
}

pub fn to_bytes_f32(values: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(values.len() * 4);
    for value in values {
        out.extend_from_slice(&value.to_le_bytes());
    }
    out
}

pub fn journal(output: &[f32], input_bytes: &[u8]) -> Vec<u8> {
    let mut bytes = to_bytes_f32(output);
    let mut hasher = Sha256::new();
    hasher.update(input_bytes);
    bytes.extend_from_slice(hasher.finalize().as_slice());
    bytes
}
