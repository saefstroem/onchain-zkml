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

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Witness {
    pub input: Vec<i16>,
    pub layer: u8,
}

pub fn to_bytes(values: &[i16]) -> Vec<u8> {
    let mut out = Vec::with_capacity(values.len() * 2);
    for value in values {
        out.extend_from_slice(&value.to_le_bytes());
    }
    out
}

pub fn journal(output: &[i16], input: &[i16]) -> Vec<u8> {
    let mut bytes = to_bytes(output);
    let mut hasher = Sha256::new();
    hasher.update(to_bytes(input));
    bytes.extend_from_slice(hasher.finalize().as_slice());
    bytes
}

