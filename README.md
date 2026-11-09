# CoMET

CoMET is a framework for verifying robotic task execution in RLBench using
behavior-tree conditions and live failure detectors. It combines visual condition
checks with geometric and force-based monitoring, and provides a pipeline for
task description generation, behavior-tree construction, detector calibration,
and execution.

## Workflow

| Stage | Entry point | Output |
| --- | --- | --- |
| Scene inspection | `01_ttm_context/main.py` | Object, waypoint, and reference geometry from task models |
| Task descriptions | `02_descriptions/main.py` | Camera images, gripper sequences, and task descriptions |
| Behavior trees | `03_behavior_trees/main.py` | Per-stage conditions and detector assignments |
| Execution | `04_running/main.py` | Clean and injected-failure episodes, telemetry, and visual verdicts |

Detector calibration is performed with `calibration/main.py` before monitored
execution. Scene inspection reads existing `.ttm` models; it does not modify them.
All experiment artifacts are generated under `outputs/` by default.

## Requirements

- Ubuntu 22.04 on `x86_64`.
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
behavior-tree generation, calibration, and execution in order. The defaults are
10 clean calibration episodes, one worker, and all supported injected failure
cases plus a clean baseline.

The pipeline regenerates task evidence and behavior trees, replaces the selected
tasks' calibration data, and reruns execution cases. Set a separate output root
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
python 03_behavior_trees/main.py --task basketball_in_hoop
python calibration/main.py all --task basketball_in_hoop --episodes 10
python 04_running/main.py --task basketball_in_hoop --failures all
```

Steps 1–4 accept repeated `--task` arguments or `--all`. Calibration accepts
repeated `--task` arguments. Without a task selection, calibration `collect` and
`all` discover configured task YAMLs.
Use `--workers` on calibration and execution to control simulator concurrency.

To open an existing BT in the web interface without regenerating it:

```bash
python 03_behavior_trees/main.py --task basketball_in_hoop --no-auto-generate
```

This requires a saved BT in `outputs/behavior_trees/` and accepts one task.

Detailed simulator output and condition
evidence are retained in `outputs/runs/<task>/<case>/run.log` for inspection.
Completed cases are reused unless `--redo` is supplied.
Detector plots are saved as PNG and SVG.

For detector-only execution:

```bash
python 04_running/main.py --task basketball_in_hoop --checks detectors --failures all
```

`--checks detectors` disables runtime visual verification.

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
`src/paths.py`.

### Condition-rule catalogue

The shared catalogue is stored in
`src/behavior_trees/generated_condition_rules.json`. To customize it:

1. Edit `config/condition_taxonomy_input.json` to define categories, predicates,
   occurrence, and stage rules.
2. If needed, edit `PRIMITIVE_DEFINITIONS`, `ACTION_SEQUENCE`, and `SYSTEM_PROMPT`
   in `src/behavior_trees/condition_rule_generator.py`.
3. Validate the input, generate the catalogue, and rebuild the affected trees:

```bash
python src/behavior_trees/condition_rule_generator.py \
  --input config/condition_taxonomy_input.json --validate-only

python src/behavior_trees/condition_rule_generator.py \
  --input config/condition_taxonomy_input.json

python 03_behavior_trees/main.py --task basketball_in_hoop --generate-only
```

Validation does not call the API; catalogue generation requires `OPENAI_API_KEY`.
Generation replaces the catalogue JSON. The full pipeline reads this catalogue
but does not regenerate it automatically.

The BT generator receives the complete task description. Existing failure links
retain their supplied names. Conditions without a failure link are matched to
catalogue predicates; unmatched conditions remain unlinked.

In BT Condition Studio, edit conditions directly or enter a request under
**What is missing in this stage?** and click **Update stage** to have the model
edit, add, or remove conditions for that stage. Other stages and unsaved edits
are retained. Changing a stage's primitive offers to regenerate its preconditions
and postconditions or keep them; hold detectors always follow the selected
primitive. Click **Save Review** to save the changes. Stage updates do not change
the source task description.

### Runtime configuration files

| File | Purpose |
| --- | --- |
| `config/condition_definitions.json` | Predicate definitions and visual guidance for runtime condition checks |
| `config/detector_primitives.json` | Detector assignments by action primitive |
| `config/detector_vlm.json` | Default model and camera views for detector confirmation |
| `config/thresholds.json` | Detector calibration parameters |

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

`collect` records clean telemetry and reuses complete task baselines. `compute`
recalculates thresholds from recorded data. `all` performs both operations.
Collection automatically runs up to 8 simulator processes, bounded by available
CPU and memory. Episodes of the same task run in parallel too. Each simulator
collects its own batch in an isolated directory; complete, validated batches are
merged into the task baseline. The terminal shows episode progress and elapsed
time. Detailed simulator output is saved under `outputs/calibration/logs/`.
Use `--workers 1` for sequential collection or set a specific parallel limit:

```bash
python calibration/main.py all --task basketball_in_hoop --episodes 10 --workers 4
```

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

## Repository and artifacts

```text
CoMET/
├── 01_ttm_context/       # scene inspection
├── 02_descriptions/      # evidence capture and descriptions
├── 03_behavior_trees/    # behavior-tree generation and review
├── 04_running/           # monitored execution
├── calibration/          # shared calibration entry point
├── config/               # conditions, detector settings, and calibration
├── patches/              # pinned backend adaptations
├── scripts/              # environment setup and dependency checks
├── src/                  # implementation modules
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

## Troubleshooting

| Issue | Resolution |
| --- | --- |
| `conda: command not found` | Initialize Conda for Bash or source its `etc/profile.d/conda.sh` before setup. |
| Missing simulator dependency | Complete the pinned source downloads and patch application before creating the environment. |
| Python import errors | Activate the project environment and run commands from the repository root. Rebuild PyRep if its compiled binding is missing. |
| Qt `xcb`, shared-library, or display errors | Verify the system dependencies and activation settings. Use the Xvfb commands above and inspect the renderer with `glxinfo -B` on the same display. |
| Missing generated input | Run the preceding stage for the same task and output-root configuration. |
| API authentication or model-access error | Set `OPENAI_API_KEY` and verify access to the selected model. |
