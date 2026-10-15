# Vendored ONNX Runtime Web

`onnxruntime-web` **1.29.0**, taken from the npm package and committed here so the
demo page has no third-party runtime dependency: the site makes no external
requests, cannot break when a CDN does, and keeps working offline once cached.

Files

| File | Purpose |
|------|---------|
| `ort.wasm.min.mjs` | ES-module entry point (CPU/wasm build only) |
| `ort-wasm-simd-threaded.mjs` | wasm loader glue |
| `ort-wasm-simd-threaded.wasm` | the SIMD wasm build (run single-threaded) |

The page sets `ort.env.wasm.numThreads = 1` because GitHub Pages cannot send the
`Cross-Origin-Opener-Policy` / `Cross-Origin-Embedder-Policy` headers that
`SharedArrayBuffer`, and therefore multi-threaded wasm, requires.

ONNX Runtime is published by Microsoft under the MIT licence. See
<https://github.com/microsoft/onnxruntime> for the source and the full notices.

To refresh this directory:

```bash
npm pack onnxruntime-web@<version>
tar xzf onnxruntime-web-<version>.tgz
cp package/dist/ort.wasm.min.mjs package/dist/ort-wasm-simd-threaded.{mjs,wasm} docs/vendor/onnxruntime/
```
