#![no_main]
#![deny(dead_code, unused_must_use)]

use risc0_zkvm::guest::env;
use onchain_zkml_engine::{forward, journal, Witness};

risc0_zkvm::guest::entry!(main);

fn main() {
    let witness: Witness = env::read();
    let output = forward(&witness.input, witness.layer as usize).expect("layer forward failed");
    env::commit_slice(&journal(&output, &witness.input));
}
