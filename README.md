# VBVR-Pro-Bench

<div align="center">

<p align="center">
    <a href="https://video-reason.com/?v=pro" target="_blank">
        <img alt="Project Page" src="https://img.shields.io/badge/Project%20-%20Homepage-4285F4" height="20" />
    </a>
    <a href="https://arxiv.org/abs/2608.26105" target="_blank">
        <img alt="arXiv" src="https://img.shields.io/badge/arXiv-VBVR_Pro-red?logo=arxiv" height="20" />
    </a>
    <a href="https://github.com/Video-Reason/VBVR-Pro" target="_blank">
        <img alt="Code" src="https://img.shields.io/badge/Training_&_Inference-VBVR_Pro-100000?style=flat-square&logo=github&logoColor=white" height="20" />
    </a>
    <a href="https://huggingface.co/datasets/Video-Reason/VBVR-Pro-SFT-Video" target="_blank">
        <img alt="Dataset" src="https://img.shields.io/badge/%F0%9F%A4%97%20_VBVR_Pro_Dataset-Video-ffc107?color=ffc107&logoColor=white" height="20" />
    </a>
    <a href="https://huggingface.co/datasets/Video-Reason/VBVR-Pro-SFT-Image" target="_blank">
        <img alt="Dataset" src="https://img.shields.io/badge/%F0%9F%A4%97%20_VBVR_Pro_Dataset-Image-ffc107?color=ffc107&logoColor=white" height="20" />
    </a>
    <a href="https://huggingface.co/datasets/Video-Reason/VBVR-Pro-RL" target="_blank">
        <img alt="Dataset" src="https://img.shields.io/badge/%F0%9F%A4%97%20_VBVR_Pro_Dataset-RL-ffc107?color=ffc107&logoColor=white" height="20" />
    </a>
    <a href="https://huggingface.co/datasets/Video-Reason/VBVR-Pro-Bench/tree/main" target="_blank">
        <img alt="Bench Data" src="https://img.shields.io/badge/%F0%9F%A4%97%20_VBVR_Pro_Bench-Data-ffc107?color=ffc107&logoColor=white" height="20" />
    </a>
    <a href="https://video-reason.com/pro/bench/#leaderboard" target="_blank">
        <img alt="Leaderboard" src="https://img.shields.io/badge/%F0%9F%A4%97%20_VBVR_Pro_Bench-Leaderboard-ffc107?color=ffc107&logoColor=white" height="20" />
    </a>
    <a href="LICENSE.md#code--apache-license-20">
        <img alt="Code License" src="https://img.shields.io/badge/Code-Apache_2.0-blue.svg" height="20" />
    </a>
    <a href="LICENSE.md#data-and-benchmark-materials--cc-by-nc-40">
        <img alt="Data License" src="https://img.shields.io/badge/Data-CC_BY--NC_4.0-blue.svg" height="20" />
    </a>
</p>

</div>

The evaluation kit for **VBVR-Pro-Bench** — 100 visual-reasoning tasks, each with
its own specific evaluator.

Every score is computed with classical computer vision and combinatorial
algorithms (OpenCV + NumPy, Hungarian matching, multi-object tracking). There is no neural judge
and no API call anywhere in this repository — the same input always produces the
same score, and each score comes with a `details` dict recording how it was
derived.

The benchmark ships in two settings, scored by two entry points:

| Setting | Entry point | Model output |
|---|---|---|
| Video (TI2V) | `run_evaluation_video.py` | one `.mp4` per instance |
| Image (interleaved) | `run_evaluation_image.py` | one directory of `.png` steps per instance |

---

## 1. Quick Start

### 1.1 Install

```bash
git clone https://github.com/Video-Reason/VBVR-Pro-Bench.git
cd VBVR-Pro-Bench

pip install -r requirements.txt
```

Python 3.9+. `requirements.txt` covers everything the evaluators need.

Some tasks read on-screen text or digits with EasyOCR, which downloads its
weights to `~/.EasyOCR` on first use. To point it at a copy you already have:

```bash
export VBVR_EASYOCR_MODELS=/path/to/easyocr_models
```

### 1.2 Download Ground Truth Data

> **Dataset:** [https://huggingface.co/datasets/Video-Reason/VBVR-Pro-Bench](https://huggingface.co/datasets/Video-Reason/VBVR-Pro-Bench)

```bash
# Install huggingface_hub if needed
pip install huggingface_hub

# Download the dataset
huggingface-cli download Video-Reason/VBVR-Pro-Bench --repo-type dataset --local-dir /path/to/VBVR-Pro-Bench

cd /path/to/VBVR-Pro-Bench
tar xzf VBVR-Pro-Bench-Video.tar.gz     # video setting
tar xzf VBVR-Pro-Bench-Image.tar.gz     # image setting
```

100 tasks (`In-Domain_50` 50 + `Out-of-Domain_50` 50), 5 instances each, so
**500 instances per setting**. Resolution 1024 × 1024; reference videos are 16 fps.

After extracting, the **video** setting looks like this:

```
/path/to/VBVR-Pro-Bench-Video/
├── In-Domain_50/
│   ├── G-131_select_next_figure_.../
│   │   ├── 00000/
│   │   │   ├── first_frame.png     # Input image (condition frame)
│   │   │   ├── prompt.txt          # Text prompt
│   │   │   ├── ground_truth.mp4    # Reference video
│   │   │   ├── final_frame.png     # Last frame of the reference video
│   │   │   └── metadata.json       # Task parameters and symbolic ground truth
│   │   ├── 00001/
│   │   └── ...                     # 5 instances per task
│   └── ...                         # 50 tasks
└── Out-of-Domain_50/
    └── ...                         # 50 tasks
```

The **image** setting has the same shape, with the reference output as an image
sequence instead of a video:

```
/path/to/VBVR-Pro-Bench-Image/
├── In-Domain_50/
│   ├── G-131_select_next_figure_.../
│   │   ├── 00000/
│   │   │   ├── first_frame.png     # Input image (condition frame)
│   │   │   ├── prompt.txt          # Text prompt
│   │   │   ├── frame_1.png         # Reference output, step 1
│   │   │   ├── frame_2.png         # ... step 2, when the task has more steps
│   │   │   ├── ...                 # up to frame_N.png
│   │   │   └── metadata.json       # Task parameters and symbolic ground truth
│   │   └── ...
│   └── ...
└── Out-of-Domain_50/
    └── ...
```

> **Note:** `metadata.json` is required. It carries the symbolic ground truth
> (grid layouts, object positions, correct answers) that the evaluators check
> against — scoring will not work without it.

### 1.3 Generate Model Output (Inference)

For each instance, condition on `first_frame.png` and `prompt.txt` and produce one
video (or one image sequence).

**Video setting** — one `.mp4` per instance:

```
/path/to/model_outputs/
├── In-Domain_50/
│   ├── G-131_select_next_figure_.../
│   │   ├── 00000.mp4               # Generated video for instance 0
│   │   ├── 00001.mp4
│   │   ├── 00002.mp4
│   │   ├── 00003.mp4
│   │   └── 00004.mp4               # 5 videos per task
│   └── ...                         # Same task folders as GT
└── Out-of-Domain_50/
    └── ...                         # Same task folders as GT
```

**Image setting** — one directory per instance, holding its output steps:

```
/path/to/model_outputs/
├── In-Domain_50/
│   ├── G-131_select_next_figure_.../
│   │   ├── 00000/
│   │   │   ├── 001.png             # Step 1
│   │   │   ├── 002.png             # Step 2, if the task has more steps
│   │   │   └── ...
│   │   ├── 00001/
│   │   └── ...                     # 5 instances per task
│   └── ...
└── Out-of-Domain_50/
    └── ...
```

> **Note:** The folder names (`In-Domain_50/`, `Out-of-Domain_50/`, task names)
> and instance ids (`00000` … `00004`) must match the ground truth structure
> exactly. Only `.mp4` is recognised in the video setting.

Every `.png` inside an instance directory is one output step, ordered by the
number in its filename — so `1.png, 2.png, 10.png` and `001.png, 002.png, 010.png`
both order correctly. Single-step tasks just have one file.

### 1.4 Run Evaluation

**Video**

```bash
python run_evaluation_video.py \
    --model_path /path/to/model_outputs \
    --gt_base /path/to/VBVR-Pro-Bench-Video \
    --output_dir ./video_eval_results
```

**Image**

```bash
python run_evaluation_image.py \
    --model_path /path/to/model_outputs \
    --gt_image_base /path/to/VBVR-Pro-Bench-Image \
    --output_dir ./image_eval_results
```

**Batch evaluation (multiple models).** Point at the directory that contains them,
and optionally restrict the list:

```bash
python run_evaluation_video.py \
    --models_base /path/to/all_model_outputs \
    --gt_base /path/to/VBVR-Pro-Bench-Video

python run_evaluation_video.py \
    --models_base /path/to/all_model_outputs \
    --gt_base /path/to/VBVR-Pro-Bench-Video \
    --models model_A model_B
```

#### Arguments

| Argument | Applies to | Description |
|---|---|---|
| `--model_path` | both | Path to a single model's output directory |
| `--models_base` | both | Base directory containing multiple model folders |
| `--models` | both | Specific model names to evaluate (with `--models_base`) |
| `--gt_base` | video | **(Required)** Path to `VBVR-Pro-Bench-Video` |
| `--gt_image_base` | image | **(Required)** Path to `VBVR-Pro-Bench-Image` |
| `--output_dir` | both | Output directory for results |
| `--device` | both | `cuda` or `cpu` (default: `cuda`); affects OCR only |

`--model_path` and `--models_base` are mutually exclusive, and one is required.

Use a GPU if you have one. The geometric and graph algorithms are CPU-only, but
the OCR step runs on the GPU when `--device cuda` (the default) is set, and is
noticeably slower without it. `--device cpu` still works if no GPU is available.

---

## 2. Output Format

Each model produces `{output_dir}/{model_name}_vbvr_results.json`. With
`--models_base`, the video entry point additionally writes
`all_models_summary.json` alongside it.

```json
{
  "model_name": "my_model",
  "summary": {
    "In_Domain":     { "mean_score": 0.48, "num_samples": 250, "by_task": {}, "by_category": {} },
    "Out_of_Domain": { "mean_score": 0.65, "num_samples": 250, "by_task": {}, "by_category": {} },
    "overall":       { "mean_score": 0.56, "num_samples": 500, "by_task": {}, "by_category": {} }
  },
  "samples": [
    {
      "task_name": "G-131_select_next_...",
      "video_file": "00000.mp4",
      "folder": "In-Domain_50",
      "split": "In_Domain",
      "category": "Abstraction",
      "score": 0.85,
      "dimensions": { "task_specific": 0.85 },
      "error": null
    }
  ]
}
```

Aggregates are reported per split, per task, and per cognitive category. The 100
tasks break down as:

| Category | Tasks |
|---|---|
| Perception | 28 |
| Abstraction | 25 |
| Knowledge | 19 |
| Spatiality | 16 |
| Transformation | 12 |

Report `Out_of_Domain` for generalization: those task families do not appear in
the VBVR-Pro training splits.

---

## 3. Repository Structure

```
VBVR-Pro-Bench/
├── run_evaluation_video.py         # Entry point: video (I2V) setting
├── run_evaluation_image.py         # Entry point: interleaved image setting
├── requirements.txt                # Python dependencies
└── vbvr_bench/
    ├── __init__.py                 # VBVRBench class
    ├── utils.py                    # Shared CV primitives (color/shape/frame ops)
    └── evaluators/
        ├── __init__.py             # Evaluator registry, task categories, split definitions
        ├── base_evaluator.py       # BaseEvaluator: dimensions, weights, frame handling
        ├── In_Domain_50_part1..5.py       # 50 in-domain task evaluators
        └── Out_of_Domain_50_part1..5.py   # 50 out-of-domain task evaluators

```

`TASK_EVALUATOR_MAP` in `vbvr_bench/evaluators/__init__.py` maps each of the 100
task names to its evaluator class. To score a single instance directly:

```python
from vbvr_bench.evaluators import get_evaluator

task = "G-45_key_door_matching_data-generator"
gt_dir = f"/path/to/VBVR-Pro-Bench-Video/In-Domain_50/{task}/00000"

evaluator = get_evaluator(task, device="cpu")
result = evaluator.evaluate({
    "video_path": "pred/00000.mp4",
    "task_name": task,
    "gt_path": gt_dir,
    "metafile_path": [f"{gt_dir}/metadata.json"],
}, task_specific_only=True)

print(result["score"])    # float in [0, 1]
print(result["details"])  # per-evaluator diagnostics
```

---

## License

VBVR-Pro source code, scripts, configuration files and task-specific scoring
software — including everything in this repository — are licensed under the
Apache License 2.0. VBVR-Pro data and benchmark materials are separately
licensed under CC BY-NC 4.0. Model weights and third-party materials remain
subject to their applicable model-card and upstream terms. See
[LICENSE.md](LICENSE.md) for details.

## Citation

```bibtex
@misc{xu2026vbvrproscalableverifiablesuite,
      title={VBVR-Pro: A Scalable and Verifiable Suite for Native Visual Reasoning},
      author={Junxiang Xu and Ruisi Wang and Fanyi Pu and Maijunxian Wang and Ran Ji and Tongxi Zhou and Chenyang Gu and Jing Zuo and Hongcan Xiao and Yimeng Geng and Wanqi Yin and Wei Chen and Oscar Qian and Zhengan Yan and Ziqi Huang and Haiwen Diao and Liang Pan and Bo Li and Xiangyu Fan and Dezhi Luo and Fengyuan Yu and Zehong Zhao and Qingying Gao and Tinghui Zhu and Yilan Zhang and Jingqi Tong and Pinyuan Feng and Zhengze Jiang and Letian Wang and Ziyu Guo and Renrui Zhang and Jieneng Chen and Sonia Joseph and Constantin Venhoff and Saman Motamed and Mengyue Yang and Chandra Sripada and Alan Yuille and Philip Torr and Lvmin Zhang and Vikash Kumar and Daniel Khashabi and Nikolaus Kriegeskorte and Rapha\"{e}l Milli\`{e}re and Vincent C. M\"{u}ller and Anyi Rao and Quan Wang and Ziwei Liu and Dahua Lin and Lei Yang and Hokin Deng and Zhongang Cai},
      year={2026},
      eprint={2608.26105},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2608.26105},
}
```
