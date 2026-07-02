import { readFileSync, writeFileSync } from "node:fs";
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
const flag = (name) => {
  const pref = `--${name}=`;
  const a = ARGS.find((x) => x.startsWith(pref));
  return a ? a.slice(pref.length) : null;
};
const IS_ADDRESS = ARGS.includes("address");
const DRY_RUN = ARGS.includes("--dry-run");
const FROM = flag("from-layer-idx");
const TO = flag("to-layer-idx");
const LAYER_INPUT_FILE = flag("layer-input");

const TREASURY = "kaspatest:qz9agq5pr6yrnrxh3m5ure9yy0eggwup9jjhp32zw8zknmfkqunf7c9tt5x6u";
const NETWORK = "testnet-10";
const RPC_URL = "wss://tn10.stroem.finance";
const PAYOUT = TREASURY;
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

function proveLayer(layer, layerInput) {
  const args = ["run", "--release", "-q", "-p", "onchain-zkml-prover", "--", String(layer)];
  if (layerInput) {
    args.push(layerInput);
  }
  const stdout = execFileSync("cargo", args, { cwd: ROOT, maxBuffer: 64 * 1024 * 1024 });
  return JSON.parse(stdout.toString());
}

const OP_CAT = "7e";
const OP_SHA256 = "a8";

function covenant(proof) {
  const outputHex = proof.journal.slice(0, proof.output.length * 8);
  const inputHashHex = proof.journal.slice(proof.output.length * 8);

  const verifier = ZkScriptBuilder.newR0({ flags: { covenantsEnabled: true } });
  verifier.appendR0SuccinctVerifier(proof.imageId, proof.controlId, proof.hashFn);
  const redeemScript = OP_CAT + OP_SHA256 + verifier.drain();

  const sig = ZkScriptBuilder.newR0({ flags: { covenantsEnabled: true } });
  sig.pushR0SuccinctWitness(proof.receipt);
  sig.addData(outputHex);
  sig.addData(inputHashHex);
  sig.addData(redeemScript);

  return { sigScript: sig.drain(), redeemScript };
}

function depositAddress(redeemScript) {
  return addressFromScriptPublicKey(payToScriptHashScript(redeemScript), NETWORK).toString();
}

function makeTx(utxo, finalized, outValue, destScript) {
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
    outputs: [{ value: outValue, scriptPublicKey: destScript }],
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
  const p2sh = payToScriptHashScript(finalized.redeemScript);
  const address = addressFromScriptPublicKey(p2sh, NETWORK).toString();
  const journalDigest = createHash("sha256").update(Buffer.from(proof.journal, "hex")).digest("hex");
  const priceNote = proof.priceUsd == null ? "" : `  price $${Math.round(proof.priceUsd)}`;

  const outHex = proof.journal.slice(0, proof.output.length * 8);
  const inputHashHex = proof.journal.slice(proof.output.length * 8);

  console.log(`layer ${proof.layer}${proof.final ? " (final)" : ""}: output [${proof.output.join(", ")}]${priceNote}`);
  console.log("")
  console.log("")
  console.log("")
  console.log("")
  console.log(`on-chain output (${proof.output.length} x f32 LE): ${outHex}`);
  console.log(`input (f32 LE, hashed): ${proof.inputHex}`);
  console.log(`on-chain input hash: ${inputHashHex}`);
  console.log(`journal digest: ${journalDigest}`);

  const { entries } = await rpc.getUtxosByAddresses({ addresses: [address] });
  if (!entries.length) {
    return false;
  }
  const utxo = entries.reduce((a, b) => (BigInt(a.amount) >= BigInt(b.amount) ? a : b));
  const amount = BigInt(utxo.amount);
  const dest = payToAddressScript(PAYOUT);
  const sizing = makeTx(utxo, finalized, amount, dest);
  const sizeMass = calculateTransactionMass(NETWORK, sizing);
  const mass = COMPUTE_MASS > sizeMass ? COMPUTE_MASS : sizeMass;
  const computed = mass * COMPUTE_FEERATE;
  const fee = computed > FEE_FLOOR ? computed : FEE_FLOOR;
  const value = amount - fee;
  const txObj = makeTx(utxo, finalized, value, dest);
  console.log("")
  console.log(`reroute ${value} -> treasury ${PAYOUT}`);

  if (dryRun) {
    show("  tx", txObj);
    return true;
  }
  const tx = new Transaction(txObj);
  for (const inp of tx.inputs) inp.computeBudget = COMPUTE_BUDGET;
  const { transactionId } = await rpc.submitTransaction({ transaction: tx, allowOrphan: false });
  console.log(`settled tx ${transactionId}`);
  return true;
}

async function main() {
  await init({ module_or_path: readFileSync(join(ROOT, "kaspa", "kaspa_bg.wasm")) });

  if (IS_ADDRESS) {
    console.log(depositAddress(covenant(proveLayer(0)).redeemScript));
    return;
  }

  const from = Number(FROM);
  const to = Number(TO);
  if (FROM == null || TO == null || !Number.isInteger(from) || !Number.isInteger(to) || from < 0 || to < 0) {
    console.error("usage:");
    console.error("  node run.js --from-layer-idx=<n> --to-layer-idx=<m> [--layer-input=<file>] [--dry-run]");
    console.error("  node run.js address");
    console.error("proves + settles layer <m>, using layer <n>'s saved output (--layer-input) or the prompted features");
    process.exit(1);
  }

  let layerInput;
  if (LAYER_INPUT_FILE) {
    layerInput = readFileSync(join(ROOT, LAYER_INPUT_FILE), "utf8").trim();
    console.log(`layer ${to} input: ${LAYER_INPUT_FILE} (layer ${from} output, ${layerInput.split(",").length} values)`);
  } else if (to === 0) {
    layerInput = await askFeatures();
  } else {
    console.error(`layer ${to} needs --layer-input=layer${to - 1}.out (the feature prompt only feeds layer 0)`);
    process.exit(1);
  }

  const rpc = new RpcClient({ url: RPC_URL, encoding: Encoding.Borsh, networkId: NETWORK });
  await rpc.connect();
  const proof = proveLayer(to, layerInput);
  const outFile = `layer${to}.out`;
  writeFileSync(join(ROOT, outFile), proof.output.join(",") + "\n");
  console.log(`saved ${outFile}`);
  await settle(rpc, proof, DRY_RUN);
  await rpc.disconnect();
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
