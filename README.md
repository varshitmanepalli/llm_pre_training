# LLM pretraining (notebook)

End-to-end **causal LM pretraining** workflow in [`LLM_pre_training.ipynb`](LLM_pre_training.ipynb), aimed at **Google Colab** .

## What it does

1. **Data** — Loads [FineWeb](https://huggingface.co/datasets/HuggingFaceFW/fineweb) (`sample-10BT`), fetches several **Python files from GitHub** as extra text, and merges them with `concatenate_datasets`. An [Alpaca-style](https://huggingface.co/datasets/c-s-ale/alpaca-gpt4-data) set is loaded only to **inspect** instruction format; it is **not** concatenated into the pretraining mix.

2. **Cleaning** — Filters short/noisy pages (`paragraph_length_filter`) and high internal repetition (`paragraph_repetition_filter` with duplicate paragraph/character ratios). Saves **Parquet**; includes logic to move large artifacts to **Drive** when over a size threshold.

3. **Packaging** — Tokenizes with a **Llama-compatible** tokenizer (notebook uses **Upstage Solar** checkpoints, e.g. `upstage/SOLAR-10.7B-v1.0` / `TinySolar`), flattens token IDs, trims to a multiple of `max_seq_length`, and writes a **packed** Parquet for training.

4. **Model** — Builds a small **LLaMA**-style config (`LlamaConfig` / `LlamaForCausalLM`), with sections on **random init**, loading **TinySolar** weights, **depth downscaling** (fewer layers), and **depth upscaling** (more layers). Training uses a checkpoint such as `TinySolar-308m-4k-init` loaded from Drive.

5. **Training** — `datasets` + PyTorch `Dataset`, extended `TrainingArguments` (`CustomArguments`), and Hugging Face **`Trainer`** with a simple **loss logging** callback.

## Stack

Python, **PyTorch**, **Hugging Face** (`datasets`, `transformers`, `Trainer`).

## Note

Hyperparameters in the notebook (e.g. `max_seq_length`, steps, model size) look tuned for **demos / limited GPU**; adjust paths, dataset splits, and training args before serious runs.
