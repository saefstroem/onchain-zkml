use crate::error::Error;
use crate::result::Result;
use alloc::vec::Vec;
use sha2::{Digest, Sha256};

static NET: &[u8] = include_bytes!("net.bin");
const MAGIC: &[u8] = b"ZNN2";

pub struct Layer {
    pub in_dim: usize,
    pub out_dim: usize,
    relu: bool,
    weights: Vec<f32>,
    bias: Vec<f32>,
}

pub fn net_commitment() -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(NET);
    let mut out = [0u8; 32];
    out.copy_from_slice(hasher.finalize().as_slice());
    out
}

pub fn infer(input: &[f32]) -> Result<Vec<Vec<f32>>> {
    let mut current = input.to_vec();
    let mut outputs = Vec::new();
    for layer in parse(NET)? {
        current = dense_forward(&current, &layer)?;
        outputs.push(current.clone());
    }
    Ok(outputs)
}

pub fn forward(input: &[f32], layer_idx: usize) -> Result<Vec<f32>> {
    let layers = parse(NET)?;
    let layer = layers.get(layer_idx).ok_or(Error::Shape)?;
    dense_forward(input, layer)
}

pub fn layer_count() -> Result<usize> {
    Ok(parse(NET)?.len())
}

pub fn dense_forward(input: &[f32], layer: &Layer) -> Result<Vec<f32>> {
    if input.len() != layer.in_dim {
        return Err(Error::Shape);
    }
    let mut out = Vec::with_capacity(layer.out_dim);
    for j in 0..layer.out_dim {
        let mut acc = *layer.bias.get(j).ok_or(Error::Shape)?;
        for (i, x) in input.iter().enumerate() {
            let w = *layer.weights.get(j * layer.in_dim + i).ok_or(Error::Shape)?;
            acc += w * x;
        }
        if layer.relu && acc < 0.0 {
            acc = 0.0;
        }
        out.push(acc);
    }
    Ok(out)
}

fn parse(bytes: &[u8]) -> Result<Vec<Layer>> {
    if bytes.get(0..4) != Some(MAGIC) {
        return Err(Error::Net);
    }
    let count = u16_at(bytes, 4)? as usize;
    let mut at = 6usize;
    let mut layers = Vec::with_capacity(count);
    for _ in 0..count {
        let in_dim = u16_at(bytes, at)? as usize;
        let out_dim = u16_at(bytes, at + 2)? as usize;
        let relu = u8_at(bytes, at + 4)? != 0;
        at += 5;
        let mut weights = Vec::with_capacity(in_dim.saturating_mul(out_dim));
        for _ in 0..in_dim.saturating_mul(out_dim) {
            weights.push(f32_at(bytes, at)?);
            at += 4;
        }
        let mut bias = Vec::with_capacity(out_dim);
        for _ in 0..out_dim {
            bias.push(f32_at(bytes, at)?);
            at += 4;
        }
        layers.push(Layer { in_dim, out_dim, relu, weights, bias });
    }
    Ok(layers)
}

fn u8_at(bytes: &[u8], at: usize) -> Result<u8> {
    bytes.get(at).copied().ok_or(Error::Net)
}

fn u16_at(bytes: &[u8], at: usize) -> Result<u16> {
    Ok(u16::from_le_bytes([u8_at(bytes, at)?, u8_at(bytes, at + 1)?]))
}

fn f32_at(bytes: &[u8], at: usize) -> Result<f32> {
    Ok(f32::from_le_bytes([
        u8_at(bytes, at)?,
        u8_at(bytes, at + 1)?,
        u8_at(bytes, at + 2)?,
        u8_at(bytes, at + 3)?,
    ]))
}
