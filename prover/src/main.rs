use risc0_zkvm::sha::{Digest, Digestible};
use risc0_zkvm::{default_prover, ExecutorEnv, ProverOpts};
use std::env as stdenv;
use std::fs;
use std::process::exit;
use onchain_zkml_circuit::{TRANSITION_ELF, TRANSITION_ID};
use onchain_zkml_engine::{forward, journal, layer_count, Witness};

struct Norm {
    mean: Vec<f32>,
    std: Vec<f32>,
    y_mean: f32,
    y_std: f32,
}

fn f32_at(bytes: &[u8], at: usize) -> f32 {
    f32::from_le_bytes([bytes[at], bytes[at + 1], bytes[at + 2], bytes[at + 3]])
}

fn parse_norm(bytes: &[u8]) -> Norm {
    let count = u16::from_le_bytes([bytes[0], bytes[1]]) as usize;
    let mean = (0..count).map(|i| f32_at(bytes, 2 + i * 4)).collect();
    let std = (0..count).map(|i| f32_at(bytes, 2 + count * 4 + i * 4)).collect();
    let y_mean = f32_at(bytes, 2 + count * 8);
    let y_std = f32_at(bytes, 2 + count * 8 + 4);
    Norm { mean, std, y_mean, y_std }
}

fn read_norm() -> Norm {
    let bytes = fs::read("norm.bin").unwrap_or_else(|_| {
        eprintln!("norm.bin not found, run: python3 model/train_dense.py");
        exit(1);
    });
    parse_norm(&bytes)
}

fn to_dollars(price: i16, norm: &Norm) -> f32 {
    f32::from(price) / 256.0 * norm.y_std + norm.y_mean
}

fn quantize(features: &[f64], norm: &Norm) -> Vec<i16> {
    if features.len() != norm.mean.len() {
        eprintln!("expected {} features, got {}", norm.mean.len(), features.len());
        exit(1);
    }
    features
        .iter()
        .enumerate()
        .map(|(i, x)| (((*x as f32 - norm.mean[i]) / norm.std[i]) * 256.0).round().clamp(-32768.0, 32767.0) as i16)
        .collect()
}

fn main() {
    let args: Vec<String> = stdenv::args().collect();
    let layer: usize = args.get(1).and_then(|v| v.parse().ok()).unwrap_or(0);
    let norm = read_norm();
    let count = layer_count().expect("net.bin is malformed");
    if layer >= count {
        eprintln!("layer {layer} out of range (net has {count} layers)");
        exit(1);
    }
    let input: Vec<i16> = match args.get(2) {
        Some(csv) if layer == 0 => {
            let features: Vec<f64> =
                csv.split(',').map(|s| s.trim().parse().expect("features must be numbers")).collect();
            quantize(&features, &norm)
        }
        Some(csv) => csv
            .split(',')
            .map(|s| {
                s.trim().parse::<i16>().unwrap_or_else(|_| {
                    eprintln!("layer {layer} input must be integers (layer {}'s output), got '{}'", layer - 1, s.trim());
                    exit(1);
                })
            })
            .collect(),
        None => vec![0i16; norm.mean.len()],
    };

    let output = forward(&input, layer).expect("layer forward failed");
    let last = layer + 1 == count;
    let dollars = last.then(|| {
        let price = output.first().copied().unwrap_or(0);
        to_dollars(price, &norm)
    });
    match dollars {
        Some(d) => eprintln!("layer {layer} of {count}: predicted price ${d:.0}"),
        None => eprintln!("layer {layer} of {count}: output {output:?}"),
    }

    let witness = Witness { input: input.clone(), layer: layer as u8 };
    let env = ExecutorEnv::builder().write(&witness).unwrap().build().unwrap();
    let receipt = default_prover()
        .prove_with_opts(env, TRANSITION_ELF, &ProverOpts::succinct())
        .unwrap()
        .receipt;
    receipt.verify(TRANSITION_ID).unwrap();

    let succinct = receipt.inner.succinct().unwrap();
    let receipt_bytes = borsh::to_vec(succinct).unwrap();
    let control_digests: Vec<u8> = succinct
        .control_inclusion_proof
        .digests
        .iter()
        .flat_map(|digest| digest.as_bytes().to_vec())
        .collect();
    let seal: Vec<u8> = succinct.seal.iter().flat_map(|word| word.to_le_bytes()).collect();
    let control_index = succinct.control_inclusion_proof.index.to_le_bytes();
    let mut image = [0u8; 32];
    image.copy_from_slice(Digest::from(TRANSITION_ID).as_bytes());
    let record = journal(&output, &input);

    let report = serde_json::json!({
        "layer": layer,
        "final": last,
        "priceUsd": dollars,
        "output": output,
        "hashFn": "poseidon2",
        "imageId": hex::encode(image),
        "controlId": hex::encode(succinct.control_id.as_bytes()),
        "claim": hex::encode(succinct.claim.digest().as_bytes()),
        "controlIndex": hex::encode(control_index),
        "controlDigests": hex::encode(&control_digests),
        "seal": hex::encode(&seal),
        "journal": hex::encode(&record),
        "receipt": hex::encode(&receipt_bytes),
    });
    println!("{report}");
}

#[cfg(test)]
mod tests {
    use super::*;

    fn demo_norm() -> Norm {
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../norm.bin");
        parse_norm(&fs::read(path).expect("norm.bin missing"))
    }

    #[test]
    fn scaled_output_converts_back_to_dollars() {
        let norm = demo_norm();
        println!("{}",to_dollars(6602, &norm));
    }
}
