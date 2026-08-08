// Headless HTJ2K wasm decode benchmark (same codec the browser page uses).
// Usage: node web/bench.mjs [streamsDir]
import { createRequire } from "module";
import { readFileSync } from "fs";
import { join } from "path";

const require = createRequire(import.meta.url);
// path (not bare specifier) to bypass the package's exports map
const factory = require(
  new URL("./node_modules/@cornerstonejs/codec-openjph/dist/openjphjs.js", import.meta.url).pathname
);

const streamsDir = process.argv[2] ?? new URL("./streams", import.meta.url).pathname;
const manifest = JSON.parse(readFileSync(join(streamsDir, "manifest.json")));
const REPS = 5;

const Module = await (factory.default ? factory.default() : factory());
const decoder = new Module.HTJ2KDecoder();

function decodeOnce(bytes) {
  const buf = decoder.getEncodedBuffer(bytes.length);
  buf.set(bytes);
  decoder.decode();
  return decoder.getDecodedBuffer().length;
}

const variants = Object.keys(manifest.slices[0].files);
console.log(`volume ${manifest.shape}, ${manifest.slices.length} test slices, ${REPS} reps`);
for (const v of variants) {
  const datas = manifest.slices.map((s) =>
    new Uint8Array(readFileSync(join(streamsDir, s.files[v].path)))
  );
  decodeOnce(datas[0]); // warm up
  const t0 = performance.now();
  let pixels = 0;
  for (let r = 0; r < REPS; r++) for (const d of datas) pixels += decodeOnce(d) / 2;
  const ms = performance.now() - t0;
  const msPerSlice = ms / (REPS * datas.length);
  const mb = datas.reduce((a, d) => a + d.length, 0) / 1e6;
  const fullVolumeS = ((msPerSlice * manifest.shape[0]) / 1000).toFixed(2);
  console.log(
    `${v.padEnd(12)} ${mb.toFixed(2).padStart(6)} MB  ` +
      `${msPerSlice.toFixed(1).padStart(6)} ms/slice  ` +
      `${(pixels / 1e6 / (ms / 1000)).toFixed(0).padStart(5)} MP/s  ` +
      `full ${manifest.shape[0]}-slice volume: ~${fullVolumeS}s`
  );
}
