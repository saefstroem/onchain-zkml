import { readFileSync } from "node:fs";
import { createHash } from "node:crypto";
import { execFileSync } from "node:child_process";
import { createInterface } from "node:readline";
import { stdin as input, stdout as output } from "node:process";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import init, {
  RpcClient,
  Encoding,
  Transaction,
  ZkScriptBuilder,
  payToScriptHashScript,
  payToAddressScript,
  addressFromScriptPublicKey,
  calculateTransactionMass,
} from "./kaspa/kaspa.js";

const ROOT = dirname(fileURLToPath(import.meta.url));
const ARGS = process.argv.slice(2);
const MODE = ARGS.find((a) => a === "address" || a === "--dry-run");
const TREASURY = "kaspatest:qz9agq5pr6yrnrxh3m5ure9yy0eggwup9jjhp32zw8zknmfkqunf7c9tt5x6u";
const NETWORK = "testnet-10";
const RPC_URL = "wss://tn10.stroem.finance";
const PAYOUT = TREASURY;
const LAYERS = 1;
const COMPUTE_BUDGET = 2550;
const COMPUTE_MASS = 500000n;
const COMPUTE_FEERATE = 100n;
const FEE_FLOOR = 0n;

const FEATURE_PROMPTS = [
  ["Longitude", "-122.11"],
  ["Latitude", "37.67"],
  ["House age (years)", "32"],
  ["Total rooms in the block", "3028"],
  ["Total bedrooms in the block", "811"],
  ["Population of the block", "2037"],
  ["Households in the block", "703"],
  ["Median income (tens of thousands)", "3.0645"],
];

async function askFeatures() {
  const rl = createInterface({ input, output });
  const buffered = [];
  const waiters = [];
  rl.on("line", (line) => (waiters.length ? waiters.shift()(line) : buffered.push(line)));
  const ask = (query) =>
    new Promise((resolve) => {
      output.write(query);
      buffered.length ? resolve(buffered.shift()) : waiters.push(resolve);
    });
  const values = [];
  for (const [name, example] of FEATURE_PROMPTS) {
    let value = "";
    while (value === "") {
      const answer = (await ask(`${name} [${example}]: `)).trim();
      const chosen = answer === "" ? example : answer;
      if (Number.isNaN(Number(chosen))) {
        console.log(`  '${answer}' is not a number, try again`);
      } else {
        value = chosen;
      }
    }
    values.push(value);
  }
  rl.close();
  return values.join(",");
}

function proveLayer(layer, features) {
  const args = ["run", "--release", "-q", "-p", "onchain-zkml-prover", "--", String(layer)];
  if (features) {
    args.push(features);
  }
  const stdout = execFileSync("cargo", args, { cwd: ROOT, maxBuffer: 64 * 1024 * 1024 });
  return JSON.parse(stdout.toString());
}

function covenant(proof) {
  const journalDigest = createHash("sha256").update(Buffer.from(proof.journal, "hex")).digest();
  const builder = ZkScriptBuilder.newR0({ flags: { covenantsEnabled: true } });
  builder.commitToSuccinct(proof.imageId, proof.controlId, proof.hashFn);
  return builder.finalizeWithSuccinctProof(proof.receipt, journalDigest);
}

function depositAddress(redeemScript) {
  return addressFromScriptPublicKey(payToScriptHashScript(redeemScript), NETWORK).toString();
}

function makeTx(utxo, finalized, outValue) {
  return {
    version: 1,
    inputs: [
      {
        previousOutpoint: utxo.outpoint,
        signatureScript: finalized.sigScript,
        sequence: 0n,
        sigOpCount: 0,
        computeBudget: COMPUTE_BUDGET,
        utxo,
      },
    ],
    outputs: [{ value: outValue, scriptPublicKey: payToAddressScript(PAYOUT) }],
    lockTime: 0n,
    subnetworkId: "0000000000000000000000000000000000000000",
    gas: 0n,
    payload: "",
  };
}

function show(label, value) {
  console.log(label, JSON.stringify(value, (_k, v) => (typeof v === "bigint" ? v.toString() : v), 2));
}

async function settle(rpc, proof, dryRun) {
  const finalized = covenant(proof);
  const address = depositAddress(finalized.redeemScript);
  const priceNote = proof.priceUsd == null ? "" : ` (price $${Math.round(proof.priceUsd)})`;
  console.log(`layer ${proof.layer}${priceNote}: deposit P2SH ${address}`);
  const { entries } = await rpc.getUtxosByAddresses({ addresses: [address] });
  if (!entries.length) {
    console.log(`layer ${proof.layer}: no funds at ${address} — deposit TKAS and rerun`);
    return false;
  }
  const utxo = entries.reduce((a, b) => (BigInt(a.amount) >= BigInt(b.amount) ? a : b));
  const amount = BigInt(utxo.amount);
  const sizing = makeTx(utxo, finalized, amount);
  const sizeMass = calculateTransactionMass(NETWORK, sizing);
  const mass = COMPUTE_MASS > sizeMass ? COMPUTE_MASS : sizeMass;
  const computed = mass * COMPUTE_FEERATE;
  const fee = computed > FEE_FLOOR ? computed : FEE_FLOOR;
  const value = amount - fee;
  const txObj = makeTx(utxo, finalized, value);
  console.log(`layer ${proof.layer}: ${amount} sompi at ${address}; fee ${fee}; reroute ${value} -> ${PAYOUT}`);
  if (dryRun) {
    show("  tx", txObj);
    return true;
  }
  const tx = new Transaction(txObj);
  for (const inp of tx.inputs) inp.computeBudget = COMPUTE_BUDGET;
  const { transactionId } = await rpc.submitTransaction({ transaction: tx, allowOrphan: false });
  console.log(`  settled tx ${transactionId} journal ${proof.journal.slice(0, 16)}…`);
  return true;
}

async function main() {
  await init({ module_or_path: readFileSync(join(ROOT, "kaspa", "kaspa_bg.wasm")) });

  if (MODE === "address") {
    console.log(depositAddress(covenant(proveLayer(0)).redeemScript));
    return;
  }

  const features = await askFeatures();
  const rpc = new RpcClient({ url: RPC_URL, encoding: Encoding.Borsh, networkId: NETWORK });
  await rpc.connect();
  let layerInput = features;
  for (let layer = 0; layer < LAYERS; layer++) {
    const proof = proveLayer(layer, layerInput);
    const ok = await settle(rpc, proof, MODE === "--dry-run");
    if (!ok) break;
    layerInput = proof.output.join(",");
  }
  await rpc.disconnect();
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
