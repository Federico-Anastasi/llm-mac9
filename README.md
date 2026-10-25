# llm-mac9

A small LLM inference engine in plain C that runs [Nicu](https://huggingface.co/federico-anastasi/nicu-20m), a 20-million-parameter Italian chat model, on a Power Mac G4 under Mac OS 9. On a 400 MHz G4 with 8-bit weights and AltiVec it generates about 5.6 tokens per second.

## Requirements

- Power Mac G4 (the app is built with AltiVec instructions and will not run on a G3).
- Mac OS 9 (tested on 9.1).
- 64 MB of memory for the app (40 MB minimum).

## Running it on the Mac

1. Download `Nicu.bin`, `nicu.dat` and `nicutok.dat` from the [latest release](../../releases/latest).
2. Put the three files in the same folder on the Mac. `Nicu.bin` is MacBinary: Internet Explorer 5 decodes it while downloading, otherwise open it with StuffIt Expander.
3. Open Nicu. After loading, the second line should say `AltiVec attivo`.
4. Type a message and press Return. Type `esci` or close the window to quit.

Nicu speaks Italian, so the app's messages are in Italian too.

## Files

- `nicu.c`: the engine (nanoGPT-style GPT, byte-level BPE tokenizer, top-p sampling with repetition penalty) and the chat loop. It also builds on a PC for testing.
- `nicu.r`: the app's memory partition.
- `console/`: a modified copy of the Retro68 console with a larger font (Monaco 18) and a window title.
- `export.py`: converts the Hugging Face `checkpoint.pt` and `tokenizer.json` into `nicu.dat` (weights) and `nicutok.dat` (tokenizer), big-endian.
- `ref.py`: numpy forward pass on the exported weights, compared with the official fp32 ONNX model.
- `test_engine.c`: prints logits and token ids from the C engine, to compare with `ref.py` and with Hugging Face `tokenizers`.
- `.github/workflows/build.yml`: builds the PowerPC app with the Retro68 Docker image on every push.

## Building

The Mac app is built by GitHub Actions, and the result is in the `nicu-ppc` artifact of each run. To build it locally with Docker:

```
docker run --rm -v "$PWD":/root -w /root ghcr.io/autc04/retro68 /bin/bash -c '
  mkdir -p build && cd build &&
  cmake .. -DCMAKE_TOOLCHAIN_FILE=/Retro68-build/toolchain/powerpc-apple-macos/cmake/retroppc.toolchain.cmake &&
  make'
```

To export the weights and check them (the Hugging Face folder must contain `checkpoint.pt`, `tokenizer.json` and, for `ref.py`, `model.onnx`):

```
python export.py HF_DIR OUT_DIR
python ref.py OUT_DIR HF_DIR
```

To test the engine on a PC with any C compiler:

```
cc -O2 -o nicu nicu.c -lm                 # chat; run it next to nicu.dat and nicutok.dat
cc -O2 -o test_engine test_engine.c -lm
./test_engine OUT_DIR "Ciao, come stai?"
```

## `nicu.dat` format (version 3)

The header is `NICU` followed by 7 big-endian integers: version, vocabulary size, context length, layers, heads, embedding size, tied weights. The tensors follow in model order. The large matrices (embedding, qkv, projections, MLP) are int8 with one float scale per row, and each int8 block starts on a 16-byte boundary. Everything else is float32. Compared with the fp32 ONNX model, the most likely next token is the same in 100% of the test positions.

## Notes on Mac OS 9 and Retro68

- **The stack is not 16-byte aligned** (measured offset: 12 bytes). AltiVec loads and stores ignore the low address bits, so any vector that goes through the stack is read or written at the wrong address. That is why `matvec_altivec` is `noinline`, takes its constants and its sum from aligned globals, and why the project builds with `-fno-tree-vectorize -fno-tree-slp-vectorize`: with auto-vectorization, plain code gave wrong results as well.
- At startup `altivec_check` compares the AltiVec path with the scalar one on a real weight matrix. If they differ, the app uses the scalar path and says so.
- Retro68 ships no `altivec.h`, so the code calls the `__builtin_altivec_*` functions directly, with `-mcpu=7400 -maltivec -mabi=altivec -mvrsave -Wa,-maltivec`.
- Multitasking is cooperative: `yield_cpu()` calls `WaitNextEvent` after every token, otherwise the screen and other apps freeze. Weights load in 1 MB chunks, with a dot and a pause after each one.
- The console works in Mac Roman and the model in UTF-8. `mac_to_utf8` and `emit_bytes` convert between them.

## License

The code is under the MIT license, except `console/`, which comes from [Retro68](https://github.com/autc04/Retro68) and keeps its GPLv3 license with the GCC Runtime Library Exception. The Nicu weights are under [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/): non-commercial use only, with attribution.
