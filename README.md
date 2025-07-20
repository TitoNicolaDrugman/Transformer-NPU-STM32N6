# Text Generation STM32 Model Zoo

* **`c_code/`** contains the code to run the transfomer on the board + some evaluations methods.
* **`st_modelzoo/`** contains the files to train, quantized en evaluate the transformer. This folder provides a **reference implementation and tooling to train, quantize, evaluate, benchmark and deploy compact text generation (language modeling) networks on STM32 devices** (MCU / MPU) using the STM32Cube.AI toolchain.

> Focus: lightweight Transformer‐style (e.g. *tiny BERT generator*) word level language models suitable for on‑device next‑token generation.

---

## Directory components

* **`datasets/`** – Placeholder + utilities (`combine_corpus.py`) for assembling training / validation corpora. Contains example public domain texts (Shakespeare, Moby Dick, etc.) and a `README.md` / `Source.md` describing provenance.
* **`src/`** – Core Python packages implementing training, evaluation, prediction (generation), quantization, preprocessing, benchmarking, utilities, models, and configuration examples.

  * `models/` – Model definitions (e.g. `tiny_bert_generator.py`).
  * `preprocessing/` – Data loading / tokenization & corpus preparation helpers.
  * `training/` – Training loop, callbacks, learning rate scheduling, augmentation (if any), and documentation.
  * `evaluation/` – Accuracy / perplexity / quality evaluation utilities.
  * `prediction/` – Text generation (greedy / sampling) scripts.
  * `quantization/` – Post‑training quantization & representative dataset handling.
  * `benchmarking/` – Host‑side scripts to benchmark runtime, memory, throughput on STM32 targets.
  * `config_file_examples/` – Minimal YAML examples for each operation mode.
  * `experiments_outputs/` – Auto‑generated run artifacts (Hydra configs, logs, TensorBoard, quantized & float models). **Not intended for manual editing.**
  * `utils/` – Auxiliary functions (model management, config parsing, header generation, etc.).
* **`deployment/`** – Scripts + generated C sources (`c_project/Middlewares/ST/AI/*`) to package a quantized model for STM32 boards (MCU or MPU variants). Includes board‑specific READMEs (`README_STM32H7.md`, `README_STM32N6.md`, `README_MPU.md`).
* **`conversion_scripts/`** – Helper(s) to convert exported `.tflite` or `.h5` artifacts into C arrays / headers.
* **`c_headers/`** – Generated headers (e.g. `model.h`, `tokenizer.h`, `embedding.h`) for embedding inside firmware projects.
* **`pretrained_models/`** – Collection (or pointers) to example pre‑quantized / quantized language models (e.g. MobileNet reference left here as template — replace with text models).
* **`mlruns/`** – MLflow tracking directory (local) storing experiment metadata, parameters, metrics, artifacts. Each run subfolder contains logs, TensorBoard events, quantized & float checkpoints.
* **`st_ai_ws/`** – (Optional) Workspace integration placeholder for STM32 AI ecosystem tools.
* **Top‑level scripts** – `stm32ai_main.py` (entrypoint orchestrating chains), `check_tflite*.py` sanity checks, evaluation helpers, `user_config.yaml` exemplar configuration.

---

## Core Concepts

The workflow is orchestrated by `stm32ai_main.py` using an `operation_mode` value from the YAML configuration. Modes encapsulate single services (training, quantization, evaluation, prediction, benchmarking, deployment) or *chains* that sequence them.

**Typical metrics:** loss (cross‑entropy), accuracy (if framed as next‑token classification), validation loss, and derived perplexity (can be computed as `exp(loss)`).

---

## Operation Modes

In chain names: **t** = training, **q** = quantization, **e** = evaluation, **b** = benchmarking, **d** = deployment.

| `operation_mode` | Operations (in order)                                                                    | Notes                                                   |
| :--------------- | :--------------------------------------------------------------------------------------- | :------------------------------------------------------ |
| `training`       | Train a text generation model (BYOD – bring your own dataset / BYOM – your architecture) | Produces float model + checkpoints                      |
| `evaluation`     | Evaluate a float or quantized model                                                      | Computes metrics (loss / accuracy / perplexity)         |
| `quantization`   | Quantize a float model                                                                   | Creates `.tflite` + representative data usage           |
| `prediction`     | Generate text using a float or quantized model                                           | Supports greedy / temperature sampling (if implemented) |
| `benchmarking`   | Benchmark model on STM32 board                                                           | Measures latency / memory                               |
| `deployment`     | Package & deploy model on STM32 board                                                    | Generates C project in `deployment/c_project`           |
| `chain_tqe`      | training → quantization → evaluation                                                     | Fast end‑to‑end quality check                           |
| `chain_tqeb`     | training → quantization → evaluation → benchmarking                                      | Adds on‑device performance profile                      |
| `chain_eqe`      | evaluation (float) → quantization → evaluation (quantized)                               | Compare float vs quantized metrics                      |
| `chain_eqeb`     | evaluation (float) → quantization → evaluation (quantized) → benchmarking                | Quality + performance delta                             |
| `chain_qb`       | quantization → benchmarking                                                              | For existing float model                                |
| `chain_qd`       | quantization → deployment                                                                | Direct field deployment                                 |

---

## Model Types

Initial reference model: **`tiny_bert_generator`** (compact Transformer encoder producing next‑token logits). Additional model variants can be added under `src/models/` – ensure each exposes a factory function and is referenced in configuration.

*Add more lightweight architectures (e.g. distilled transformer, GRU / LSTM baselines) for comparative benchmarking.*

---

## Datasets & Tokenization

Place raw corpus files (plain text, UTF‑8) inside `datasets/`. Use `combine_corpus.py` to merge or preprocess multiple sources into `full_corpus.txt`.

Tokenization strategy (example): simple whitespace / word index (see `tokenizer.json`). You may substitute a character‑level or subword tokenizer; update preprocessing + config accordingly.

**Representative data for quantization**: by default pulled from the training corpus unless a dedicated quantization path is specified.

---

## Configuration Files

All tunable settings reside in YAML (Hydra/OmegaConf). Main sections typically include:

* `general`: run naming, model path, output directory.
* `dataset`: paths, sequence length, vocabulary specs, batch sizes.
* `model`: architecture hyperparameters (embedding dim, heads, layers, FFN size, etc.).
* `training`: epochs, optimizer params, learning rate schedule, early stopping.
* `quantization`: representative dataset size, calibration parameters.
* `prediction`: prompt, max tokens, decoding temperature / top‑k (if available).
* `deployment` & `benchmarking`: board type, clock config, runtime profiling options.
* `tools.stm32ai`: on‑cloud / local settings, version.

Override any parameter from the CLI (Hydra pattern): `section.param=value`.

---

## Typical Workflows

**1. Train & Evaluate**

```
operation_mode=training
```

Inspect metrics (TensorBoard) in `src/experiments_outputs/<run>/logs`.

**2. Quantize & Compare**

```
operation_mode=chain_eqe model.float_model_path=<path_to_h5>
```

Check metric deltas (accuracy / loss / perplexity) pre vs post quantization.

**3. Benchmark On Device**

```
operation_mode=chain_qb deployment.hardware_setup.board=STM32H7
```

Review latency / memory usage reported by benchmarking scripts.

**4. Deploy**

```
operation_mode=chain_qd deployment.hardware_setup.board=STM32N6570-DK
```

Generated C sources land under `deployment/c_project/Middlewares/ST/AI/`.

**5. Generate Text**

```
operation_mode=prediction prediction.prompt="STM32 embedded AI" prediction.max_tokens=50
```

Retrieve generated sequence from console / saved logs.

---

## Benchmarking & MLflow

Each run logs parameters + metrics to `mlruns/` (MLflow). TensorBoard events stored in `.../logs/`. Use these to compare architectures or quantization impacts across runs.

---

## Deployment

Deployment scripts integrate with STM32Cube.AI (local or Developer Cloud if `tools.stm32ai.on_cloud=true`). After `chain_qd`, follow board README instructions to flash and (if needed) toggle boot switches (e.g. STM32N6570-DK notice).

Generated headers (`model.h`, `tokenizer.h`, `embedding.h`) are prepared for inclusion in application firmware.

---

## Extending the Zoo

1. Add new model file under `src/models/`.
2. Expose a build function returning a compiled `tf.keras.Model`.
3. Register / reference it in configuration (`model.name` or similar parameter).
4. Provide a minimal config example for training & quantization.
5. (Optional) Supply a pre‑trained & quantized artifact in `pretrained_models/`.

---

## License

See `LICENSE.md` and any embedded license files inside generated middleware folders.

---

## Getting Help

Check individual READMEs inside subfolders for mode‑specific guidance (training, quantization, evaluation, deployment). For quick orientation, start with a small corpus + reduced model dimensions, then scale once the pipeline works end‑to‑end.

Happy generating on STM32! 🚀
