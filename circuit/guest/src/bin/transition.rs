#![no_main]
#![deny(dead_code, unused_must_use)]

use risc0_zkvm::guest::env;
use onchain_zkml_engine::{forward, journal, standardize, to_bytes_f32, Witness};

risc0_zkvm::guest::entry!(main);

fn main() {
    let witness: Witness = env::read();
    let layer = witness.layer as usize;
    let input_bytes = to_bytes_f32(&witness.input);
    let input = if layer == 0 {
        standardize(&witness.input).expect("standardize failed")
    } else {
        witness.input.clone()
    };
    let output = forward(&input, layer).expect("layer forward failed");
    env::commit_slice(&journal(&output, &input_bytes));
}
