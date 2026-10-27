# CoMET

CoMET is a framework for verifying robotic task execution in RLBench using
behavior-tree conditions and live failure detectors. It combines visual condition
checks with geometric and force-based monitoring, and provides a pipeline for
task description generation, behavior-tree construction, detector calibration,
execution, and evaluation.

## Workflow

| Stage | Entry point | Output |
| --- | --- | --- |
| Scene inspection | `01_ttm_context/main.py` | Object, waypoint, and reference geometry from task models |
| Task descriptions | `02_descriptions/main.py` | Camera images, gripper sequences, and task descriptions |
| Behavior trees | `03_behavior_trees/main.py` | Per-stage conditions and detector assignments |
| Execution | `04_running/main.py` | Clean and injected-failure episodes, telemetry, and visual verdicts |
| Evaluation | `05_scoring/main.py` | Checkpoint samples and detection metrics |

Detector calibration is performed with `calibration/main.py` before monitored
execution. Scene inspection reads existing `.ttm` models; it does not modify them.
All experiment artifacts are generated under `outputs/` by default.

## Requirements

- Ubuntu 22.04 on `x86_64`.
- Conda, available through [Miniforge](https://github.com/conda-forge/miniforge#install)
  or an existing Conda installation. The installer creates a Python 3.10 environment.
- A C/C++ compiler, graphics drivers, and an OpenGL-capable X display.
- CoppeliaSim 4.1, PyRep, RLBench, and FailGen, installed as described below.
- An OpenAI API key for description generation, behavior-tree generation, and
  runtime visual verification.

The simulator workflow requires Linux: the pinned PyRep backend supports
simulator communication only on Linux
([PyRep installation notes](https://github.com/stepjam/PyRep#install)).
Native macOS and Windows simulator execution are not supported by this setup.

## Installation

### System dependencies

```bash
sudo apt-get update
sudo apt-get install -y git curl ca-certificates xz-utils build-essential \
  libgl1 libegl1 libglu1-mesa libgl1-mesa-dri libxrender1 libxi6 libxrandr2 \
  libxkbcommon-x11-0 libxcb-xinerama0 libfontconfig1 mesa-utils xvfb xauth
```

### Repository

```bash
git clone https://github.com/ramtin-tabatabaei/CoMET.git
cd CoMET
```

Run the remaining installation commands from the repository root.

### Simulator dependencies

The following revisions are used with the bundled backend patches. FailGen is
obtained from the upstream AHA repository; the patches provide the task fixes,
failure injectors, and backend adaptations used by CoMET.

```bash
mkdir -p external

git clone https://github.com/stepjam/PyRep.git external/PyRep
git -C external/PyRep checkout 8f420be8064b1970aae18a9cfbc978dfb15747ef

git clone --branch peract https://github.com/MohitShridhar/RLBench.git external/RLBench
git -C external/RLBench checkout ad991951bc53e4f3b73b803a75cf4b7d55295cf7

git clone https://github.com/NVlabs/AHA.git external/AHA-sources
git -C external/AHA-sources checkout 0d39c0591566ddaf997be5822f3dead8e08501aa
cp -a external/AHA-sources/aha/Data_Generation/rlbench-failgen external/rlbench-failgen

git apply --check --directory=external/RLBench patches/rlbench.patch
git apply --directory=external/RLBench patches/rlbench.patch
git apply --check --directory=external/rlbench-failgen patches/failgen.patch
git apply --directory=external/rlbench-failgen patches/failgen.patch
```

Apply each patch once to an unmodified copy of its pinned source revision.

Install the Ubuntu 20.04 build of CoppeliaSim 4.1:

```bash
curl -fL --retry 3 \
  https://downloads.coppeliarobotics.com/V4_1_0/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04.tar.xz \
  -o external/CoppeliaSim.tar.xz
mkdir -p external/CoppeliaSim
tar -xJf external/CoppeliaSim.tar.xz --strip-components=1 -C external/CoppeliaSim
ln -sf libcoppeliaSim.so external/CoppeliaSim/libcoppeliaSim.so.1
```

Additional simulator installation details are available in the
[AHA dependency instructions](https://github.com/NVlabs/AHA#2-pyrep-and-coppelia-simulator).

### Python environment

```bash
bash scripts/create_local_env.sh
source scripts/activate_local.sh
python scripts/check_local_env.py
```

The installer creates `.conda/aha-publish`, installs the Python dependencies,
and builds PyRep against the local simulator. The dependency check completes
with `Local environment checks passed.`

Run `source scripts/activate_local.sh` in each terminal before using CoMET.
After relocating the checkout, recreate the environment: Conda environments
and editable installs contain absolute paths. The `external/`, `.conda/`, and
cache directories are excluded from version control.

### API configuration

```bash
export OPENAI_API_KEY="your-key"
```

Description generation, behavior-tree generation and review, and runtime visual
verification default to
[GPT-5.6 Sol](https://developers.openai.com/api/docs/models/gpt-5.6-sol)
(`gpt-5.6-sol`). The API account must have access to the selected model.
Use `--model` on an individual stage to override its model. The full pipeline
provides separate `--generation-model` and `--verification-model` options.

Credentials are read from environment variables. `.env.example` documents the
settings and is not loaded automatically.

## Usage

### Complete pipeline

```bash
python run_pipeline.py --task basketball_in_hoop
```

This runs scene inspection, evidence capture, description generation,
behavior-tree generation, calibration, execution, and scoring in order. The
defaults are 10 clean calibration episodes, one worker, and all supported injected
failure cases plus a clean baseline.

The pipeline regenerates task evidence and behavior trees, replaces the selected
tasks' calibration data, and reruns evaluation cases. Set a separate output root
to retain an earlier experiment.

Preview the commands without simulator or API calls:

```bash
python run_pipeline.py --task basketball_in_hoop --dry-run
```

Select multiple tasks or all configured tasks:

```bash
python run_pipeline.py --task basketball_in_hoop --task close_box
python run_pipeline.py --all
```

Task names correspond to YAML files in `external/rlbench-failgen/failgen/configs/`.
The bundled condition-rule catalogue is used automatically; customization is
optional (see [Condition-rule catalogue](#condition-rule-catalogue)).

### Individual stages

```bash
python 01_ttm_context/main.py --task basketball_in_hoop
python 02_descriptions/main.py --task basketball_in_hoop
python 03_behavior_trees/main.py --task basketball_in_hoop --generate-only
python calibration/main.py all --task basketball_in_hoop --episodes 10
python 04_running/main.py --task basketball_in_hoop --failures all
python 05_scoring/main.py --task basketball_in_hoop
```

Steps 1–4 accept repeated `--task` arguments or `--all`. Calibration accepts
repeated `--task` arguments. Without a task selection, calibration `collect` and
`all` discover configured task YAMLs; `compute` discovers recorded clean data.
Use `--workers` on calibration and execution to control simulator concurrency.

Stage 2 captures waypoint images and gripper sequences before generating the
description. `--reuse-evidence` reuses previously captured evidence in the
configured output directory:

```bash
python 02_descriptions/main.py --task basketball_in_hoop --reuse-evidence --model YOUR_MODEL
```

Omit `--generate-only` from stage 3 to open BT Condition Studio for interactive
review. Keep its server running while reviewing and stop it with Ctrl+C before
continuing in the same terminal. Multiple-task runs and the full pipeline use
`--generate-only` automatically.

Stage 4 resumes completed cases. Use `--redo` after changing behavior trees,
detector settings, or calibration. `--failures none` selects clean evaluation;
`--waypoints` restricts injection waypoints while retaining the clean baseline.

```bash
python 04_running/main.py --task basketball_in_hoop --failures collision --waypoints 1,2 --redo
```

For detector-only execution and scoring:

```bash
python 04_running/main.py --task basketball_in_hoop --checks detectors --failures all
python 05_scoring/main.py --task basketball_in_hoop --methods A2
```

`--checks detectors` disables runtime visual verification. Description and
behavior-tree generation still use the API. On `run_pipeline.py`, this option
also selects A2 for scoring.

All numbered entry points, `calibration/main.py`, and `run_pipeline.py` support
`--dry-run`. Their `--help` commands work without a configured simulator.

### Headless execution

Simulator image capture requires an X display even when the simulator window is
hidden. For execution without a desktop display, use Xvfb:

```bash
xvfb-run -a -s "-screen 0 1400x900x24" \
  python 01_ttm_context/main.py --task basketball_in_hoop

xvfb-run -a -s "-screen 0 1400x900x24" \
  python run_pipeline.py --task basketball_in_hoop
```

Scene inspection provides a simulator startup check without API calls. Further
graphics configuration is described in the
[PyRep headless documentation](https://github.com/stepjam/PyRep#running-headless).

## Configuration

### Output paths

Generated artifacts default to `outputs/`. To use another location, set both
variables after activating the environment and use the same settings for every
stage:

```bash
export AHA_OUTPUT_ROOT=/absolute/path/to/comet-results
export AHA_CALIBRATION_ROOT="$AHA_OUTPUT_ROOT/calibration"
```

`AHA_CALIBRATION_ROOT` can also point to separate calibration storage. Activation
resets both paths to the checkout defaults. Shared path definitions are in
`src/aha_publish/paths.py`.

### Condition-rule catalogue

The shared catalogue is stored in
`src/aha_publish/behavior_trees/generated_condition_rules.json`. To customize it:

1. Edit `config/condition_taxonomy_input.json` to define categories, predicates,
   occurrence, and stage rules.
2. If needed, edit `PRIMITIVE_DEFINITIONS`, `ACTION_SEQUENCE`, and `SYSTEM_PROMPT`
   in `src/aha_publish/behavior_trees/condition_rule_generator.py`.
3. Validate the input, generate the catalogue, and rebuild the affected trees:

```bash
python src/aha_publish/behavior_trees/condition_rule_generator.py \
  --input config/condition_taxonomy_input.json --validate-only

python src/aha_publish/behavior_trees/condition_rule_generator.py \
  --input config/condition_taxonomy_input.json

python 03_behavior_trees/main.py --task basketball_in_hoop --generate-only
```

Validation does not call the API; catalogue generation requires `OPENAI_API_KEY`.
Generation replaces the catalogue JSON. The full pipeline reads this catalogue
but does not regenerate it automatically.

The BT generator receives the complete task description. Existing failure links
retain their supplied names. Conditions without a failure link are matched to
catalogue predicates; unmatched conditions remain unlinked.

### Runtime configuration files

| File | Purpose |
| --- | --- |
| `config/condition_definitions.json` | Predicate definitions and visual guidance for runtime condition checks |
| `config/detector_primitives.json` | Detector assignments by action primitive |
| `config/detector_vlm.json` | Default model and camera views for detector confirmation |
| `config/thresholds.json` | Detector calibration parameters |
| `config/condition_failure_mapping.json` | Reference description of condition-to-scoring-label mappings |

The scoring mappings are implemented in
`src/aha_publish/scoring/make_detection_samples.py`; editing the reference JSON
alone does not change scoring behavior.

For example, the default grasp-loss detector assignment is:

```json
{"slip": ["transport", "release", "place"]}
```

Available primitives are `move`, `grasp`, `transport`, `place`, `release`, and
`push`. Configured assignments apply during generation, review, and execution,
and override saved per-stage selections. An empty list disables that detector;
a stage without a primitive does not receive it. Detectors absent from the
configuration retain their normal BT selections.

The slip detector collects its held-force baseline during grasping. A commanded
gripper opening ends its grasp-loss check, including during a release stage.

## Calibration

Run scene inspection for each task before collecting calibration data.
Calibration uses the task models and YAMLs and does not require descriptions or
behavior trees.

```bash
python calibration/main.py all --task basketball_in_hoop --episodes 10
python calibration/main.py collect --task basketball_in_hoop --episodes 10
python calibration/main.py compute --task basketball_in_hoop
```

`collect` records clean telemetry and resumes completed episodes. `compute`
recalculates thresholds from recorded data. `all` performs both operations.
Use `--force` with `collect` or `all` to replace an existing clean baseline:

```bash
python calibration/main.py all --task basketball_in_hoop --episodes 10 --force
```

To recompute using an explicit configuration or external telemetry:

```bash
python calibration/main.py compute --task basketball_in_hoop \
  --config config/thresholds.json --raw-dir /path/to/clean_data
```

### Threshold calculation

| Detector | Calculation |
| --- | --- |
| Transition | Maximum of the arrival floor and scaled median clean arrival error, per waypoint |
| Orientation | Maximum of the arrival floor and scaled maximum clean arrival error, per waypoint |
| Collision | Per-joint De Luca residual thresholds: `max(residual_floor, clean_max_abs_residual) * residual_joint_multipliers` |
| Slip | Mean held force minus `grip_k` times its standard deviation, bounded by the floor and optional ceiling |

Distances use meters, orientations radians, grip forces newtons, and joint
torques newton-meters. `grip_ceiling: null` disables the ceiling. Slip statistics
average the filtered per-episode means and standard deviations. Repeated
waypoint occurrences retain separate arrival histories during calibration.

Each task receives `manifests/<task>.json` under `AHA_CALIBRATION_ROOT`
(default: `outputs/calibration/`). Execution loads its thresholds and runtime
settings automatically. Tasks with different runtime settings must be run
separately or recalibrated together.

### Collision observer

Collision detection uses the De Luca generalized-momentum observer (method 3).
Its settings are defined in the `collision` section of `config/thresholds.json`:

| Setting | Default | Meaning |
| --- | --- | --- |
| `residual_joint_multipliers` | `[5.0, 1.1, 1.1, 1.1, 1.1, 2.0, 1.1]` | Multipliers for joints 1–7 |
| `residual_floor` | `3.0` Nm | Minimum clean residual baseline before multiplication |
| `residual_warmup_multiplier` | `1.1` | Scalar warmup fallback; not applied to calibrated joint thresholds |
| `momentum_gain` | `25.0` s⁻¹ | Observer gain |
| `time_step` | `0.05` s | Observer integration step; must match simulation |
| `momentum_version` | `1` | Supported observer update |
| `residual_settle_steps` | `3` | Startup frames excluded from calibration |
| `residual_min_joints` | `1` | Simultaneous joint exceedances required |

The collector writes `clean_data/<task>/epN.residual.json` alongside each episode
CSV. These files record maximum absolute residuals and observer parameters.
Computation aggregates the maxima across episodes into
`residual_stats/<task>_residual_stats.json`; thresholds are reported in
`threshold_report/<task>_collision_residual.csv`.

After changing multipliers or the residual floor, rerun `compute` and execute
with `--redo`. Changes to observer gain, integration step, or settling frames
require fresh collection with `all --force`. Calibration must contain residual
data from the same observer configuration. Tasks without held-force samples use
the slip detector's fallback threshold.

## Evaluation

| Method | Signals |
| --- | --- |
| A1 | Visual condition checks |
| A2 | Raw detector results |
| A3 | Detector results with visual confirmation |
| A4 | Visual condition checks combined with confirmed detector results |

Execution with full checks enables condition verification and slip/collision
visual confirmation. Scoring reads checkpoint verdicts and confirmations from
`run.log`, together with detector telemetry.

```bash
python 05_scoring/main.py --task basketball_in_hoop
```

Rescore an existing sample CSV without simulator access:

```bash
python 05_scoring/main.py --samples outputs/scores/detection_samples.csv --methods A4
```

The default evaluation unit is a checkpoint, and the scoring window ends at the
injection waypoint. Use `--window full` to include later checkpoints or
`--unit episode` for episode-level scoring. Missing method results are excluded
for that method. Uncertain verdicts count as success by default;
`--uncertain-as failure` changes this policy.

Injections that could not manifest are excluded. `ObjectNotFound` labels are
mapped to wrong-sequence labels, and the batch runner excludes grasp-injection
cases. These policies define the evaluated benchmark coverage. Frame sidecars
provide detection latency when available.

## Repository and artifacts

```text
CoMET/
├── 01_ttm_context/       # scene inspection
├── 02_descriptions/      # evidence capture and descriptions
├── 03_behavior_trees/    # behavior-tree generation and review
├── 04_running/           # monitored execution
├── 05_scoring/           # evaluation
├── calibration/          # shared calibration entry point
├── config/               # conditions, detector settings, and calibration
├── patches/              # pinned backend adaptations
├── scripts/              # environment setup and dependency checks
├── src/aha_publish/      # implementation modules
├── tests/
└── run_pipeline.py
```

Source models and task YAMLs are supplied by the simulator dependencies.
Generated experiment data, local environments, caches, and credentials are
excluded from version control.

| Artifact | Default location |
| --- | --- |
| Scene context | `outputs/ttm_context/<task>.llm_context.json` |
| Waypoint images | `outputs/waypoint_screenshots/` |
| Combined camera grid | `outputs/descriptions/<task>_ALL_WAYPOINTS_COMBINED.png` |
| Task description | `outputs/descriptions/<task>_ALL_WAYPOINTS_COMBINED.openai.multimodal_analysis.json` |
| Gripper sequence | `outputs/gripper_sequences/<task>.json` |
| Behavior tree | `outputs/behavior_trees/<task>.bt_conditions.json` |
| Calibration manifest | `outputs/calibration/manifests/<task>.json` |
| Clean execution log | `outputs/runs/<task>/clean/run.log` |
| Checkpoint samples | `outputs/scores/detection_samples.csv` |
| Aggregate metrics | `outputs/scores/metrics_overall.csv` |
| Per-failure metrics | `outputs/scores/metrics_per_failure.csv` |

## Troubleshooting

| Issue | Resolution |
| --- | --- |
| `conda: command not found` | Initialize Conda for Bash or source its `etc/profile.d/conda.sh` before setup. |
| Missing simulator dependency | Complete the pinned source downloads and patch application before creating the environment. |
| Python import errors | Activate the project environment and run commands from the repository root. Rebuild PyRep if its compiled binding is missing. |
| Qt `xcb`, shared-library, or display errors | Verify the system dependencies and activation settings. Use the Xvfb commands above and inspect the renderer with `glxinfo -B` on the same display. |
| Missing generated input | Run the preceding stage for the same task and output-root configuration. |
| API authentication or model-access error | Set `OPENAI_API_KEY` and verify access to the selected model. |

## Tests

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```

The suite covers entry points, artifact isolation, behavior-tree generation,
scoring, waypoint reference geometry, and agreement between calibration outputs
and live detector thresholds. Simulator and API calls are mocked in the
isolation tests.

## License

See [LICENSE](LICENSE) for the code license. Simulator and backend dependencies
retain their respective upstream licenses.
