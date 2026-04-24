# Multiplication in Multimodal LLMs

This repository accompanies the paper
[*Multiplication in Multimodal LLMs: Computation with Text, Image, and Audio Inputs*](https://arxiv.org/abs/2604.18203),
to appear in *Findings of the Association for Computational Linguistics: ACL 2026*.
It contains the benchmark generators, experiment code, analysis scripts, and
LaTeX source used for the paper.

## Links

- Paper: https://arxiv.org/abs/2604.18203
- Associated dataset repository: https://huggingface.co/datasets/cjerzak/MultimodalMathBenchmarks
- Project page: https://neuristemic.ai/multiplication-in-multimodal-llms/
- Venue:
  *Findings of the Association for Computational Linguistics: ACL 2026*

## Repository Scope

The repository includes:

- paired benchmark generation for multiplication problems across text, image,
  and audio modalities,
- probe-set generation for heuristic-disagreement and trap-style evaluation,
- experiment scripts for fingerprinting, contrastive probes, LoRA training, and
  nudge tests,
- analysis code that writes figures and LaTeX macros, and
- the paper source in `PaperTexFolder/`.

The public repository is intended to stay lightweight. Large generated media
artifacts, local caches, logs, and model outputs are not part of the release
repository. Use the associated Hugging Face dataset repository for the paper's
dataset home, or regenerate assets locally from the checked-in scripts.

## Repository Layout

```text
PaperTexFolder/             LaTeX source and generated paper figures
Scripts/
  core/                     Shared helpers, logging, and Tinker clients
  generators/               Benchmark and modality asset generation
  experiments/              Fingerprinting, probes, LoRA, and nudge tests
  analysis/                 Figure and macro generation
SavedData/                  Versioned benchmark CSVs and lightweight data files
run_all.py                  Pipeline configuration and scheduler
run_all.sh                  Convenience launcher for the staged pipeline
tests/                      Lightweight regression tests
```

## Setup

System dependencies for the full pipeline:

- GNU `parallel`
- TeX (`pdflatex`, `bibtex`)
- `poppler`

Python environment:

```bash
conda env create -f environment.yml
conda activate tm_env
cp .env.example .env
```

Then fill in:

- `TINKER_API_KEY` for experiment runs
- `HF_TOKEN` if your selected model or tokenizer requires Hugging Face
  authentication

## Quick Start

Inspect the configured pipeline:

```bash
python run_all.py --show-config
python run_all.py --show-models
```

Preview the data phase without launching real jobs:

```bash
./run_all.sh M4 data 1
```

Run a real data phase:

```bash
./run_all.sh M4 data
```

The full experiment phase requires `TINKER_API_KEY`.

## Data Availability

Small benchmark CSVs are kept in `SavedData/` for reproducibility. The
associated Hugging Face dataset repository is the public home for the paper's
benchmark release:

- https://huggingface.co/datasets/cjerzak/MultimodalMathBenchmarks

## Paper Source

The paper source is in `PaperTexFolder/`.

To rebuild the paper locally:

```bash
cd PaperTexFolder
pdflatex ms.tex
bibtex ms
pdflatex ms.tex
pdflatex ms.tex
```

## Citation

If you use this repository, please cite:

```bibtex
@inproceedings{balter2026multiplication,
  title={{Multiplication in Multimodal LLMs: Computation with Text, Image, and Audio Inputs}},
  author={Balter, Samuel Gideon and Jerzak, Ethan and Jerzak, Connor Thomas},
  booktitle={Findings of the Association for Computational Linguistics: ACL 2026},
  year={2026}
}
```

The same citation is available in `CITATION.bib`.

## License

GPL v3. See `LICENSE`.
