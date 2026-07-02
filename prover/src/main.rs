use risc0_zkvm::sha::{Digest, Digestible};
use risc0_zkvm::{default_prover, ExecutorEnv, ProverOpts};
use std::env as stdenv;
use std::process::exit;
use onchain_zkml_circuit::{TRANSITION_ELF, TRANSITION_ID};
use onchain_zkml_engine::{
    feature_count, forward, journal, layer_count, standardize, target_stats, to_bytes_f32, Witness,
};

fn to_dollars(prediction: f32, y_mean: f32, y_std: f32) -> f32 {
    prediction * y_std + y_mean
}

fn main() {
    let args: Vec<String> = stdenv::args().collect();
    let layer: usize = args.get(1).and_then(|v| v.parse().ok()).unwrap_or(0);
    let count = layer_count().expect("net.bin is malformed");
    if layer >= count {
        eprintln!("layer {layer} out of range (net has {count} layers)");
        exit(1);
    }

    let raw: Vec<f32> = match args.get(2) {
        Some(csv) => csv
            .split(',')
            .map(|s| {
                s.trim().parse::<f32>().unwrap_or_else(|_| {
                    eprintln!("input must be numbers, got '{}'", s.trim());
                    exit(1);
                })
            })
            .collect(),
        None => vec![0.0f32; feature_count().expect("norm.bin is malformed")],
    };

    let input = if layer == 0 {
        standardize(&raw).expect("standardize failed")
    } else {
        raw.clone()
    };
    let output = forward(&input, layer).expect("layer forward failed");
    let input_bytes = to_bytes_f32(&raw);

    let last = layer + 1 == count;
    let (y_mean, y_std) = target_stats().expect("norm.bin is malformed");
    let dollars = last.then(|| to_dollars(output.first().copied().unwrap_or(0.0), y_mean, y_std));
    match dollars {
        Some(d) => eprintln!("layer {layer} of {count}: predicted price ${d:.0}"),
        None => eprintln!("layer {layer} of {count}: output {output:?}"),
    }

    let record = journal(&output, &input_bytes);
    let witness = Witness { input: raw, layer: layer as u8 };
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

    let report = serde_json::json!({
        "layer": layer,
        "final": last,
        "priceUsd": dollars,
        "output": output,
        "inputHex": hex::encode(&input_bytes),
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

    #[test]
    fn standardized_output_converts_back_to_dollars() {
        let (y_mean, y_std) = target_stats().expect("norm.bin is malformed");
        println!("y_mean = {y_mean:.2} USD, y_std = {y_std:.2} USD");
        for pred in [-1.0f32, -0.2, 0.0, 1.0, 2.0] {
            println!("standardized {pred:+.4} -> ${:.0}", to_dollars(pred, y_mean, y_std));
        }
        assert!((to_dollars(0.0, y_mean, y_std) - y_mean).abs() < 1.0);
        assert!(to_dollars(1.0, y_mean, y_std) > y_mean);
        assert!(to_dollars(-1.0, y_mean, y_std) < y_mean);
    }
}
