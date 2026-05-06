## `rhymesum` -- Hash files into LLM-generated poems locally

> Note: This is a toy project and far from cryptographically safe, better don't use it in production.

### What is this?
This project lets you fingerprint your files using LLM-generated poems instead of hard-to-remember checksums. It computes a `BLAKE3` hash of the passed file, and based on that injects entropy in both the LLM sampler as well as the prompt itself. The output is a five-line poem representing a fingerprint of your file.

### Example outputs

```
In twilight's hush, a phantom sloped
Across the meadow's trembling rim
A fleeting yodel echoed clear
As moonlight danced and stars drew near
In darkness's silence, all was grim
```

```
In a lovely abode I rest my head
Behind an ancient obelisk I softly tread
In the still of night a player draws near
She shakes her tambourine and whispers sweet cheer
The stars above twinkle with gentle delight
```

### Build
Build with `make`, then download `meta-llama-3.1-8b-instruct-q4_0.gguf` model using `bash download_model.sh`.

### Run
```
rhymesum <file>
```

### Credits
This project builds heavily on [llama.cpp](https://github.com/ggml-org/llama.cpp)
