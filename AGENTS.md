# AI6132 Face Age Progression Project

## Project

Course project for Generative AI for Visual Synthesis.

Topic:

Identity-Preserving Face Age Progression for Missing Children.

The technical task is:

Given a facial image at a younger age and a target age,
generate a plausible age-progressed facial image while preserving
identity-related facial characteristics.

This is a visual generation research project.

The generated image must NOT be described as a prediction of the
person's actual future appearance.

---

## Research Questions

RQ1:
How effectively can generative models perform child-to-adult
face age progression?

RQ2:
How well can diffusion-based age progression preserve identity?

RQ3:
Can identity-aware candidate selection improve identity preservation
without significantly reducing target-age accuracy?

---

## Methods

M0:
Existing GAN-based face age progression baseline.

M1:
Pretrained diffusion-based face age progression.

M2:
Identity-aware candidate selection built on M1.

M2 should generate multiple candidates and select candidates based on
identity similarity and target-age accuracy.

Do NOT train a diffusion model from scratch.

LoRA training is optional and should NOT be implemented unless
explicitly requested.

---

## Primary Benchmark

FG-NET Aging Database.

The primary experimental focus is child-to-adult age progression.

Where possible, use longitudinal pairs where the same identity has
images at younger and older ages.

---

## Data Leakage Rule

This rule is extremely important.

For experiments requiring train/validation/test splits:

SPLIT BY SUBJECT ID.

The same subject must NOT appear in both training and test sets.

Any exception must be explicitly documented for a longitudinal
source-target evaluation protocol.

Never silently introduce identity leakage.

---

## Core Metrics

### Age Accuracy

Mean Absolute Error between target age and estimated generated age.

MAE_age = mean(abs(predicted_age - target_age))

Lower is better.

### Identity Preservation

Cosine similarity between face embeddings.

Use a pretrained face recognition model such as ArcFace when feasible.

Higher is better.

### Image Quality

LPIPS.

FID may be used only when the sample size and evaluation protocol
make it meaningful.

### Efficiency

Record inference time per generated image.

---

## M2 Candidate Ranking

For each source image and target age:

1. Generate N candidates using M1.
2. Estimate the age of every candidate.
3. Calculate identity similarity between source and candidate.
4. Rank candidates.

Initial scoring function:

score =
    lambda_id * identity_similarity
    - lambda_age * normalized_age_error

N, lambda_id and lambda_age must be configurable.

Save scores for ALL candidates.

Do not save only the winner.

---

## Engineering Environment

Primary development:
Codex Cloud.

GPU execution:
Google Colab Free.

Persistent storage:
Google Drive.

Code:
GitHub.

The code must therefore work in Google Colab.

---

## GPU Constraints

Assume limited GPU memory and unpredictable Colab Free availability.

Therefore:

- Default diffusion batch size = 1
- Prefer 512x512 or lower resolution
- Use fp16 when CUDA supports it
- Use torch.no_grad() for inference
- Enable memory-efficient attention where supported
- Allow CPU offloading where practical
- Never require multi-GPU
- Never assume a specific GPU model
- Do not train a diffusion model from scratch

The program must print GPU information before experiments.

---

## Colab Constraints

Google Colab runtimes may disconnect.

Therefore every experiment must be resumable.

For every generated image:

- save immediately;
- save metadata immediately;
- skip completed samples when restarting.

Never require an entire experiment to finish before saving results.

---

## Storage Rules

Do NOT commit:

- datasets;
- model checkpoints;
- Hugging Face cache;
- generated image datasets;
- Google Drive files.

These should be ignored by Git.

Commit:

- source code;
- configuration;
- notebooks;
- tests;
- small CSV result files;
- final plots;
- documentation.

---

## Project Structure

face-age-progression/

    AGENTS.md
    README.md
    requirements.txt

    configs/
        diffusion.yaml
        identity.yaml
        experiment.yaml

    notebooks/
        01_setup.ipynb
        02_dataset.ipynb
        03_diffusion.ipynb
        04_identity.ipynb
        05_evaluation.ipynb

    src/
        data/
        models/
        evaluation/
        utils/

    scripts/
        preprocess.py
        run_diffusion.py
        run_identity.py
        evaluate_all.py

    tests/

    outputs/
        metrics/
        figures/

---

## Reproducibility

Every experiment must record:

- model name;
- model version when available;
- source image;
- source age;
- target age;
- random seed;
- inference steps;
- guidance scale;
- conditioning strength;
- runtime;
- selected device;
- candidate number.

All important hyperparameters must come from configuration files.

Do not hard-code experimental parameters throughout the source code.

---

## Research Integrity

Never fabricate results.

Never put placeholder experimental numbers into final result tables.

Never report an experiment as successful unless it actually ran.

Clearly distinguish:

- actual experimental result;
- expected result;
- placeholder;
- example.

Do not silently remove failed samples.

Log failures.

---

## Coding Rules

Use:

- Python
- PyTorch
- Hugging Face diffusers when appropriate
- pandas
- numpy
- matplotlib

Prefer simple implementations.

Avoid unnecessary abstractions.

All core code must run from Python scripts, not only notebooks.

Notebooks should call reusable functions from src/.

Before completing a coding task:

1. inspect existing code;
2. make the smallest reasonable change;
3. run relevant tests;
4. run a smoke test where possible;
5. report what was actually tested;
6. report what could not be tested due to GPU or missing data.

Do not redesign the research methodology unless explicitly requested.
