# Lobals Code

This repository is built on top of [LMMS-EVAL](https://github.com/EvolvingLMMs-Lab/lmms-eval) and provides an implementation of **Lobals**: a training-free test-time latent-space re-forwarding method for improving visual perception in multimodal large language models.

---

## Main Files

The main implementation files are:

```text
lmms_eval/models/simple/internvl3_5_reforward_visual_attention_after_layer.py
lmms_eval/models/simple/qwen3_vl_reforward_visual_attention_after_layer.py
```

These files implement the re-forwarding logic for InternVL3.5 and Qwen3-VL models.

---

## Installation

This repository follows the standard LMMS-EVAL installation workflow.

### Environment for InternVL3.5

```bash
conda create -n internvl python=3.10
conda activate internvl

cd Lobals_code
pip install -e .
```

### Environment for Qwen3-VL

```bash
conda create -n qwen python=3.10
conda activate qwen

cd Lobals_code
pip install -e .
```

---

## Model Checkpoints

Before running evaluation, please replace `model_checkpoint_path` in the commands below with your own local model checkpoint path.

For example:

```bash
pretrained=/path/to/your/model/checkpoint
```

---

## Evaluation

All evaluation commands should be run from the repository root:

```bash
cd Lobals_code
```

The following commands evaluate Lobals on the `pope_full` task as an example.

---

### Qwen3-VL-8B

```bash
python -m lmms_eval \
    --model qwen3_vl_reforward_visual_attention_after_layer \
    --model_args pretrained=model_checkpoint_path,top_k=9,after_layer=35,candidate_end=35,candidate_start=8 \
    --tasks pope_full \
    --batch_size 1 \
    --device cuda:0 \
    --output_path ./results \
    --log_samples
```

---

### Qwen3-VL-2B

```bash
python -m lmms_eval \
    --model qwen3_vl_reforward_visual_attention_after_layer \
    --model_args pretrained=model_checkpoint_path,top_k=3,after_layer=27,candidate_end=27,candidate_start=4 \
    --tasks pope_full \
    --batch_size 1 \
    --device cuda:0 \
    --output_path ./results \
    --log_samples
```

---

### InternVL3.5-14B

```bash
python -m lmms_eval \
    --model internvl3_5_reforward_visual_attention_after_layer \
    --model_args pretrained=model_checkpoint_path,top_k=9,after_layer=39,candidate_end=39,candidate_start=8 \
    --tasks pope_full \
    --batch_size 1 \
    --device cuda:0 \
    --output_path ./results \
    --log_samples
```

---

## Arguments

The main Lobals-related arguments are passed through `--model_args`.

| Argument | Description |
| --- | --- |
| `pretrained` | Path to the local model checkpoint. |
| `top_k` | Length of the latent revisit window. |
| `after_layer` | The layer after which the hidden state is taken for re-forwarding. |
| `candidate_start` | Start layer of the candidate reusable region. |
| `candidate_end` | End layer of the candidate reusable region. |

For example:

```bash
--model_args pretrained=model_checkpoint_path,top_k=9,after_layer=35,candidate_end=35,candidate_start=8
```

This means that the model selects a revisit window of length `9` from candidate layers `[8, 35]`, takes the hidden state after layer `35`, and re-forwards it through the selected visually grounded window.

---

## Output

Evaluation outputs will be saved to:

```text
./results
```

When `--log_samples` is enabled, LMMS-EVAL will also save detailed prediction logs for later analysis.

---

## Acknowledgement

This repository is based on LMMS-EVAL. We thank the LMMS-EVAL developers for providing a unified evaluation framework for multimodal large language models.
