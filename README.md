# CoMET

CoMET verifies robotic task execution in RLBench with behavior-tree conditions
and live failure detectors. Its workflow covers scene inspection, task
descriptions, behavior-tree generation, detector calibration, execution
monitoring, and scoring. Each numbered stage has one `main.py`.

Starting with an empty `outputs/` folder, CoMET generates its TTM context,
screenshots, gripper-sequence JSON, descriptions, behavior trees, clean
calibration data, thresholds, execution logs, and scores. The starting inputs
are source models (`.ttm`), task YAMLs, and the simulator/backend installations.

## macOS compatibility

The full simulator workflow currently requires Linux. Opening CoppeliaSim on a
Mac does not establish compatibility with CoMET: the pinned PyRep dependency
requires CoppeliaSim 4.1 and explicitly supports simulator communication only on
Linux ([PyRep installation notes](https://github.com/stepjam/PyRep#install)).
The bundled installer and dependency check also expect Linux simulator libraries
such as `libcoppeliaSim.so`. Native macOS simulation would require a separate
port and validation, including matching the simulator, Python bindings, and CPU
architecture. A newer macOS CoppeliaSim installation is not a replacement for
the pinned simulator in these instructions.

Offline scoring is separate from simulation and can be tried on a Mac using a
sample CSV exported from an Ubuntu run. With a Python 3.10 environment active,
run the following from the repository root:

```bash
python -m pip install -r requirements.txt
python 05_scoring/main.py --samples /path/to/detection_samples.csv --out-dir outputs/scores
```

This offline route has not been tested on macOS. Use Ubuntu for the complete
capture, calibration, and execution workflow below.

## Workflow

| Step | Main entry point | What it produces |
| --- | --- | --- |
| 1. TTM context | `01_ttm_context/main.py` | Scene, object, waypoint, and reference-rule context from existing `.ttm` models |
| 2. Descriptions | `02_descriptions/main.py` | Waypoint screenshots, camera grids, gripper sequences, and task descriptions |
| 3. Behavior trees | `03_behavior_trees/main.py` | Conditions used to verify each waypoint |
| 4. Running | `04_running/main.py` | Clean/failure episode logs, detector traces, and VLM events |
| 5. Scoring | `05_scoring/main.py` | Checkpoint samples and detection metrics |

Before running step 4 for the first time, use `calibration/main.py all` to collect
clean episodes and calculate detector thresholds. Later, use `compute` to change
threshold settings without rerunning the simulator.

Step 1 **inspects existing TTM models**; it does not create or modify `.ttm` models.

## Setup

These instructions target **Ubuntu on an Intel/AMD 64-bit laptop (`x86_64`)**,
using Bash. The development host runs Ubuntu 22.04 with the Ubuntu 20.04 build
of CoppeliaSim 4.1. Native Windows, macOS, ARM, and WSL simulator installations
have not been validated by this project.

You need a working OpenGL renderer and an X display for simulator image capture,
including when the simulator runs without its own window. Install your laptop's
graphics driver before the simulator check. Description generation, BT generation,
and runtime VLM checks also need an OpenAI API key with access to the selected model.

### Prepare a new Ubuntu laptop

Install the host tools and graphics libraries:

```bash
sudo apt-get update
sudo apt-get install -y git curl ca-certificates xz-utils build-essential \
  libgl1 libegl1 libglu1-mesa libgl1-mesa-dri libxrender1 libxi6 libxrandr2 \
  libxkbcommon-x11-0 libxcb-xinerama0 libfontconfig1 mesa-utils xvfb xauth
```

Check whether Conda is available:

```bash
conda --version
```

If that reports `conda: command not found`, install
[Miniforge](https://github.com/conda-forge/miniforge#unix-like-platforms-macos--linux)
for Linux `x86_64`. The following installs it into a new `~/miniforge3` folder:

```bash
curl -fL --retry 3 \
  https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh \
  -o /tmp/Miniforge3-Linux-x86_64.sh
bash /tmp/Miniforge3-Linux-x86_64.sh -b -p "$HOME/miniforge3"
source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda --version
```

In a new terminal, source that `conda.sh` again before using the project scripts,
or run `conda init bash` once and reopen your terminal. An existing working Conda
installation can be used instead. The project script creates its own Python 3.10
environment; you do not need to install a separate system Python version.

### Clone CoMET

Clone the project into a `CoMET` directory and enter it:

```bash
git clone https://github.com/ramtin-tabatabaei/CoMET.git CoMET
cd CoMET
```

CoppeliaSim 4.1, PyRep, RLBench task models, and FailGen are separate dependencies;
a Git clone does not include `external/`. Obtain them below before creating the
environment. You do not need to copy an existing `outputs/` folder or other
generated experiment files from another machine.

### Obtain simulator dependencies

Run these commands from this repository's root. The source revisions match the
locally validated dependencies. The two bundled patches provide the backend
adaptations, task fixes, failure injectors, and portable configuration paths
used by CoMET. FailGen is obtained from the upstream AHA repository below.

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

Apply the patches once to fresh source copies. Existing local copies prepared
for CoMET already contain these changes.

Download the CoppeliaSim 4.1 archive appropriate for your Linux host from the
[AHA installation instructions](https://github.com/NVlabs/AHA#2-pyrep-and-coppelia-simulator).
For the Ubuntu 20.04 build used here:

```bash
curl -fL --retry 3 \
  https://downloads.coppeliarobotics.com/V4_1_0/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04.tar.xz \
  -o external/CoppeliaSim.tar.xz
mkdir -p external/CoppeliaSim
tar -xJf external/CoppeliaSim.tar.xz --strip-components=1 -C external/CoppeliaSim
ln -sf libcoppeliaSim.so external/CoppeliaSim/libcoppeliaSim.so.1
```

The host-package command above includes Xvfb and X authentication support.
Headless runs still need an X server with OpenGL support. See the
[PyRep headless setup](https://github.com/stepjam/PyRep#running-headless)
for graphics-server configuration.

### Create and activate the environment

Create a fresh Python 3.10 Conda environment **inside this folder**:

```bash
bash scripts/create_local_env.sh
source scripts/activate_local.sh
python scripts/check_local_env.py
```

The dependency check should finish with `Local environment checks passed.`
Then test simulator startup without calling an API:

```bash
python 01_ttm_context/main.py --task basketball_in_hoop
```

Successful inspection creates
`outputs/ttm_context/basketball_in_hoop.llm_context.json`.

The installer uses `.conda/aha-publish`, installs the Python dependencies, and
builds PyRep against the local simulator. For each new terminal, run
`source scripts/activate_local.sh`; this selects the new environment and replaces
inherited paths with paths inside this checkout. It does not source your `.bashrc`.

The large `external/`, `.conda/`, and cache directories are ignored by Git.
After moving the checkout, recreate `.conda/aha-publish`; Conda environments and
editable installs contain absolute paths. Keep dependency licenses with their
source copies.

For a headless simulator check that does not call an API:

```bash
xvfb-run -a -s "-screen 0 1400x900x24" \
  python 01_ttm_context/main.py --task basketball_in_hoop
```

### API credentials and output paths

For description generation, BT generation, or VLM verification, set your OpenAI
key in the shell before running the command:

```bash
export OPENAI_API_KEY="your-key"
```

Provider calls use your API account. Credentials are read from environment
variables. `.env.example` documents settings; the code does not automatically
read that file or shell profiles. `--model` selects a model for a stage. Steps 2
and 3, along with step 4's VLM verification, use OpenAI.
OpenAI calls default to [GPT-5.6 Sol](https://developers.openai.com/api/docs/models/gpt-5.6-sol)
(`gpt-5.6-sol`), including condition-rule generation,
descriptions, behavior-tree generation/review, and runtime verification.
Explicit model arguments and model environment variables override this default.

All generated files default to `outputs/` inside this folder. To use another
location, set `AHA_OUTPUT_ROOT` **consistently for all stages**, after sourcing
`scripts/activate_local.sh`. Activation resets the output paths to this checkout:

```bash
export AHA_OUTPUT_ROOT=/absolute/path/to/comet-results
export AHA_CALIBRATION_ROOT="$AHA_OUTPUT_ROOT/calibration"
# To store calibration separately, override AHA_CALIBRATION_ROOT with that path.
```

Set both variables when changing the output location after activation, because
the activation script exports an explicit calibration path as well.

The defaults and paths are defined once in `src/aha_publish/paths.py`.

Workers run from this project folder with its own Python source path and
artifact locations. Old `PYTHONPATH`, per-detector calibration paths, and
waypoint-chain paths inherited from another checkout are not used. The explicit
top-level `AHA_OUTPUT_ROOT` and `AHA_CALIBRATION_ROOT` settings still select where
CoMET stores its generated files.

### First complete trial on a new laptop

After dependency checks and simulator inspection pass, activate the environment
and set `OPENAI_API_KEY` in the same terminal. Start with one task, one clean
calibration episode, and a clean evaluation:

```bash
python run_pipeline.py --task basketball_in_hoop --episodes 1 --failures none --dry-run
python run_pipeline.py --task basketball_in_hoop --episodes 1 --failures none
```

This trial includes API calls for descriptions, BTs, and runtime verification.
It checks installation and execution; one clean episode is not a failure-detection
benchmark. Use the normal calibration count and injected failures for evaluation.

If there is no usable desktop X display, wrap each simulator command, including
the pipeline, in Xvfb:

```bash
xvfb-run -a -s "-screen 0 1400x900x24" \
  python run_pipeline.py --task basketball_in_hoop --episodes 1 --failures none
```

With the default output paths, a completed trial produces:

| Result | File |
| --- | --- |
| Task inspection | `outputs/ttm_context/basketball_in_hoop.llm_context.json` |
| Description | `outputs/descriptions/basketball_in_hoop_ALL_WAYPOINTS_COMBINED.openai.multimodal_analysis.json` |
| Behavior tree | `outputs/behavior_trees/basketball_in_hoop.bt_conditions.json` |
| Calibration manifest | `outputs/calibration/manifests/basketball_in_hoop.json` |
| Clean execution log | `outputs/runs/basketball_in_hoop/clean/run.log` |
| Checkpoint samples | `outputs/scores/detection_samples.csv` |
| Scores | `outputs/scores/metrics_overall.csv` and `outputs/scores/metrics_per_failure.csv` |

### Setup troubleshooting

| Symptom | What to check |
| --- | --- |
| `conda: command not found` | Source the `conda.sh` from your Conda installation before running setup or activation. |
| `Missing local dependency` | Complete the simulator/source downloads and patches before running `scripts/create_local_env.sh`. |
| `No module named aha_publish` or `pyrep` | Run from the repository root after `source scripts/activate_local.sh`; rerun the environment installer if PyRep did not build. |
| Qt `xcb` errors, missing shared libraries, or display errors | Check the host packages, source the activation script, and try the Xvfb simulator check above. Use `glxinfo -B` on the same display to inspect the OpenGL renderer. |
| Missing descriptions, BTs, or calibration manifests | Run the preceding stages for the same task with the same output-root settings. |
| API authentication or model-access error | Set `OPENAI_API_KEY` in the current terminal and check that the API account can use the selected model. |

## Generate everything with one command

After setup, start the complete workflow from an empty output folder. The
bundled condition-rule catalogue is ready to use; customize it first only if
you need different rules (see [Condition-rule catalogue](#condition-rule-catalogue)):

```bash
python run_pipeline.py --task basketball_in_hoop

# Preview the stage commands without simulation or API calls.
python run_pipeline.py --task basketball_in_hoop --dry-run

# Multiple tasks, or all tasks in the configured backend:
python run_pipeline.py --task basketball_in_hoop --task close_box
python run_pipeline.py --all
```

This runs steps 1, 2, 3, clean calibration, 4, and 5 in order. Step 2 captures
ordinary waypoint images for the combined grid; extra gripper open/close image
rows are excluded. It captures
the gripper sequence before sending the description prompt to the model. Its
file is `outputs/gripper_sequences/<task>.json`, generated by the extractor
included in CoMET. Missing or failed gripper capture stops description generation.

The full pipeline regenerates evidence and BTs, replaces the selected tasks'
clean baselines, and reruns evaluation cases. It defaults to all supported
failure cases plus clean runs, 10 calibration episodes, and one worker. Use
`--failures none` for clean evaluation only. `--checks detectors` disables
runtime VLM verification and scores A2; descriptions and BT generation still
use the selected API provider. To resume individual stages, use the commands
below. Use a new `AHA_OUTPUT_ROOT` to keep a previous experiment separately.

## Run one task, in order

Replace `basketball_in_hoop` with a task name from your failgen configs. Step 3a
is only needed when customizing the bundled catalogue, as described in
[Condition-rule catalogue](#condition-rule-catalogue).

```bash
# 1. Inspect the existing task model.
python 01_ttm_context/main.py --task basketball_in_hoop

# 2. Capture images and gripper evidence, then generate the description.
python 02_descriptions/main.py --task basketball_in_hoop

# 3a. Optional: regenerate the shared catalogue after changing its inputs.
python src/aha_publish/behavior_trees/condition_rule_generator.py \
  --input config/condition_taxonomy_input.json

# 3b. Generate the task's behavior-tree conditions using that catalogue.
python 03_behavior_trees/main.py --task basketball_in_hoop

# Calibrate before the first monitored evaluation.
python calibration/main.py all --task basketball_in_hoop --episodes 10

# 4. Run a clean episode with condition checks and slip/collision VLM confirmation.
python 04_running/main.py --task basketball_in_hoop

# Include the supported injected failure cases and a clean baseline.
python 04_running/main.py --task basketball_in_hoop --failures all

# 5. Build checkpoint samples and score all available methods.
python 05_scoring/main.py
```

Append `--dry-run` to a numbered stage command, `calibration/main.py`, or
`run_pipeline.py` to print its planned commands without launching simulation,
calling an API, or writing output. `--help` works without
a configured simulator. Failures in a subprocess stop the current public command.

Steps 1–4 accept repeated `--task` arguments or `--all`. Calibration accepts
repeated `--task`; with none specified, `collect`/`all` discover task YAMLs, while
`compute` discovers existing clean-data folders. Calibration can run after step 1;
it uses the local task model and task YAML and does not require descriptions or
behavior trees. Use `--workers` on calibration and running to control
simultaneous simulator instances; it defaults to one.

Useful variants:

The single-episode runner and batch coordinator share `src/aha_publish/running/bt_runner.py`.
Step 4 uses its `batch` command. Use `python -m aha_publish.running.bt_runner single --help`
or `python -m aha_publish.running.bt_runner batch --help` for direct runner options.

```bash
# Reuse step 2's captured evidence while changing the description model.
python 02_descriptions/main.py --task basketball_in_hoop --reuse-evidence --model YOUR_MODEL

# Detector-only evaluation, without runtime VLM calls.
python 04_running/main.py --task basketball_in_hoop --checks detectors --failures collision

# Restrict injection waypoints; retain the clean baseline.
python 04_running/main.py --task basketball_in_hoop --failures collision --waypoints 1,2

# Score detector-only results using the A2 column.
python 05_scoring/main.py --methods A2

# Rescore an existing CSV without reading simulator runs.
python 05_scoring/main.py --samples outputs/scores/detection_samples.csv --methods A4
```

`--reuse-evidence` only reuses files already generated in the project's
configured output folder. It is optional and is never enabled by the full
pipeline.

Step 4 resumes existing cases. Use `--redo` after changing BTs, detector settings,
or calibration if you need fresh results. A new output root can keep experiments
separate. Step 3 opens BT Condition Studio in your browser by default and
generates a fresh BT for the selected task. Keep the terminal running while
using the interface; press Ctrl+C when finished before starting the next stage.
Use `--generate-only` to generate and save without opening the interface.
Multiple-task runs, `--all`, and `run_pipeline.py` use this terminal-only mode.
Step 1 can skip existing
reports with `--skip-existing`.

Choose the primitives that run grasp-loss checks in `config/detector_primitives.json`:

```json
{"slip": ["transport", "release", "place"]}
```

`slip` is the grasp-loss detector. The available primitives are `move`, `grasp`,
`transport`, `place`, `release`, and `push`. The config assigns this detector when
generating, reopening, or running a BT, including previously saved trees. An empty
list disables its checks; a stage without a primitive does not receive it.
Detectors absent from this config retain their normal BT selections. Configured
assignments take precedence over saved per-stage detector selections. The slip
detector still collects its grip baseline during grasping, and a commanded
gripper opening ends the grasp-loss check even in a release stage.

### Condition-rule catalogue

To customize the shared rules before generating behavior trees, edit the inputs
and run `src/aha_publish/behavior_trees/condition_rule_generator.py`:

1. Edit the generator's `PRIMITIVE_DEFINITIONS`, `ACTION_SEQUENCE`, and
   `SYSTEM_PROMPT` if your primitive semantics or rule-generation instructions
   need to change.
2. Edit the bundled `config/condition_taxonomy_input.json`.
   Customize its categories and conditions,
   including `failure`, `predicate_template`, optional `predicate_meaning`,
   `occurrence`, and `stage_rule` to define what should be checked and when.
3. Activate the local environment, set `OPENAI_API_KEY`, validate the input,
   and generate the catalogue:

```bash
source scripts/activate_local.sh

# Check the authored input without an API call.
python src/aha_publish/behavior_trees/condition_rule_generator.py \
  --input config/condition_taxonomy_input.json --validate-only

# Generate the rules; requires OPENAI_API_KEY.
python src/aha_publish/behavior_trees/condition_rule_generator.py \
  --input config/condition_taxonomy_input.json

# Rebuild each affected task's BT after changing the shared catalogue.
python 03_behavior_trees/main.py --task basketball_in_hoop
```

The generator writes `generated_condition_rules.json` beside its script in
`src/aha_publish/behavior_trees/`. Step 3 reads that JSON automatically.
Generation replaces this file; edits to the output JSON will be overwritten
the next time the generator runs.

The BT generator passes the full task description to the model. Existing failure
links retain their supplied names. When a saved condition lacks a failure link,
the loader matches its predicate against this catalogue; unmatched conditions
remain unlinked. Legacy failure-name aliases and keyword guessing are not used.

The catalogue is shared across tasks, so regenerate it when the taxonomy or
generator changes. `run_pipeline.py` uses the existing catalogue and does not
run the condition-rule generator automatically.

### Runtime condition guidance and scoring labels

`config/condition_definitions.json` supplies the predicate definitions and visual
guidance used for runtime VLM condition checks. Edit it to change how the auditor
interprets a condition. It is separate from the catalogue used to generate BTs
and from `config/detector_primitives.json`, which assigns detectors to primitives.

`config/condition_failure_mapping.json` documents how failed conditions map to
scoring labels. It is a reference file; the scoring implementation in
`src/aha_publish/scoring/make_detection_samples.py` applies those rules. Editing
the reference JSON alone does not change scoring behavior.

## Straightforward threshold calculation

Run step 1 for the task first. Calibration collects simulator telemetry directly,
so you can run it before description and behavior-tree generation (steps 2–3).

Edit **`config/thresholds.json`**, then run:

```bash
# First time: collect clean telemetry, compute thresholds, save runtime settings.
python calibration/main.py all --task basketball_in_hoop --episodes 10

# Separate collection, if desired. This resumes completed clean telemetry.
python calibration/main.py collect --task basketball_in_hoop --episodes 10

# Replace an existing clean baseline with fresh simulation episodes.
python calibration/main.py all --task basketball_in_hoop --episodes 10 --force

# Recalculate from the recorded clean episodes; no simulator or API calls.
python calibration/main.py compute --task basketball_in_hoop

# Use another tuning file or existing raw telemetry.
python calibration/main.py compute --task basketball_in_hoop \
  --config config/thresholds.json --raw-dir /path/to/clean_data
```

The configuration groups parameters by detector. Distances use meters,
orientations radians, grip forces newtons, and joint torques newton-meters.
`grip_ceiling: null` means no ceiling. Invalid settings, missing telemetry, and
empty episodes produce an error rather than a successful calibration.

Use `calibration/main.py` for the shared calibration workflow. `collect` saves
clean pose/grip telemetry and De Luca residual maxima.
`compute` produces arrival and grip baselines, De Luca thresholds, reports, and
runtime manifests. `all` performs both actions. Calibration supports the default
arrival checks and collision method 3; standalone detector evaluation, drift
tuning, proportional/runup statistics, and collision methods 1 and 2 have been
removed. Custom threshold configs must use the current `config/thresholds.json`
schema.

| Detector | Calculation retained from the existing implementation |
| --- | --- |
| Transition | Arrival threshold is the larger of the floor and scaled median clean arrival error, per waypoint |
| Orientation | Arrival threshold is the larger of the floor and scaled maximum clean arrival error, per waypoint |
| Collision — De Luca (method 3) | Per-joint thresholds: max(residual floor, clean maximum absolute residual) × joint multiplier |
| Slip | Held-force threshold: mean − k × standard deviation, bounded by floor and optional ceiling; statistics average the filtered per-episode statistics |

Repeated occurrences of a waypoint retain separate arrival histories during
calibration. Pose baselines contain the statistics used by the arrival checks.

Calibration artifacts below are relative to `AHA_CALIBRATION_ROOT`, which defaults
to `outputs/calibration/`. Each task gets `manifests/<task>.json` with its parameters
and runtime settings. Step 4 loads this manifest automatically, including the
collision and slip settings; you do not need to manually translate tuning parameters into
separate environment variables. Tasks with different runtime settings must be
run separately or recalibrated together.

The collision detector is the **De Luca generalized-momentum observer
(method 3)**. The clean collector saves `clean_data/<task>/epN.residual.json`
alongside each episode CSV. These files store the seven maximum absolute joint
residuals after the warmup and settling frames, plus the observer parameters.
Offline `compute` takes the maximum for each joint across the clean episodes and
writes `residual_stats/<task>_residual_stats.json`. The active method-3 thresholds
are listed in `threshold_report/<task>_collision_residual.csv`.

All De Luca tuning values are in the `collision` section of
**`config/thresholds.json`**:

| Setting | Default | Meaning |
| --- | --- | --- |
| `residual_joint_multipliers` | `[5.0, 1.1, 1.1, 1.1, 1.1, 2.0, 1.1]` | Threshold multipliers for joints 1–7 (indices 0–6), in order |
| `residual_floor` | `3.0` Nm | Minimum baseline before applying each joint multiplier |
| `residual_warmup_multiplier` | `1.1` | Scalar warmup fallback only; not applied to calibrated joint thresholds |
| `momentum_gain` | `25.0` s⁻¹ | Observer gain |
| `time_step` | `0.05` s | Observer integration step; must match the simulation step |
| `momentum_version` | `1` | De Luca observer update; this is the supported version |
| `residual_settle_steps` | `3` | Observer startup frames excluded from calibration |
| `residual_min_joints` | `1` | Number of joints that must exceed their thresholds simultaneously |

There is no collision-method selector; the runner always uses De Luca.
VLM confirmation camera and model defaults are in `config/detector_vlm.json`.

For each joint `i`, the calibrated threshold is:

```text
threshold[i] = max(residual_floor, clean_max_abs_residual[i])
               * residual_joint_multipliers[i]
```

For example, a 2 Nm clean residual with floor 3 Nm and multiplier 5 has a 15 Nm
threshold. Increasing a multiplier raises that joint's threshold and makes it
less sensitive.

After changing joint multipliers or the residual floor, rerun `calibration/main.py
compute --task TASK` and then step 4 with `--redo`. The generated manifest carries
the settings into the runner automatically. Changing the observer gain,
integration step, or settle count requires fresh collection with
`calibration/main.py all --task TASK --force`. Old CSV-only calibrations also need
fresh collection: they do not contain De Luca residuals. Manifests recorded with
collision methods 1 or 2 must be replaced by method-3 calibration before running.

Tasks with no held-force samples retain the slip detector's fallback.
Inspect `threshold_report/` and the manifests before interpreting results.

## Scoring policy

Scoring reads checkpoint verdicts and detector confirmations from each case's
`run.log`, alongside its detector telemetry. The checkpoint label mapping and
metric formulas are preserved:

- `A1`: condition VLM; `A2`: raw detectors; `A3`: detectors with VLM confirmation;
  `A4`: condition checks combined with confirmed detector results.
- The default scoring window ends at the injection waypoint. `--window full`
  includes subsequent checkpoints. `--unit episode` scores episodes instead.
- Missing method results are excluded for that method. Uncertain verdicts count
  as success by default; `--uncertain-as failure` changes this.
- The inherited extraction excludes injections that could not manifest. It maps
  `ObjectNotFound` into wrong-sequence labels.
  The batch runner excludes grasp-injection cases. These are benchmark-specific
  choices, not a general evaluation of all failures supported by failgen.

Frame sidecars provide latency when available. No experiment data or published performance claims
are bundled with this folder.

## Layout

```text
CoMET/
├── README.md
├── LICENSE
├── requirements.txt
├── requirements-simulator.txt
├── scripts/                 # local Conda setup, activation, dependency checks
├── .env.example
├── run_pipeline.py           # generate every stage in order
├── 01_ttm_context/main.py
├── 02_descriptions/main.py
├── 03_behavior_trees/main.py
├── 04_running/main.py
├── 05_scoring/main.py
├── calibration/main.py
├── config/                  # thresholds, taxonomy, VLM guidance, scoring reference
├── patches/                 # changes to the pinned simulator backends
├── external/                # local simulator and backend copies; ignored by Git
├── .conda/aha-publish/       # independent Python environment; ignored by Git
├── src/aha_publish/
│   ├── paths.py              # shared locations
│   ├── commands.py           # command execution and validation
│   ├── ttm/                  # scene and waypoint inspection
│   ├── descriptions/         # image evidence, prompts, providers
│   ├── behavior_trees/       # generation, review, condition rules
│   ├── running/              # simulation, BT execution, live monitors
│   ├── calibration/          # clean telemetry and threshold calculation
│   ├── scoring/              # sample extraction and metrics
│   ├── detectors/            # collision, slip, orientation, transition
│   └── simulation/           # RLBench tasks, failure options, and waypoint references
├── tests/
└── outputs/                 # created when stages run; ignored by Git
```

This README contains the setup and workflow instructions for CoMET.

Output folders are `ttm_context/`, `waypoint_screenshots/`, `descriptions/`,
`gripper_sequences/`, `behavior_trees/`, `calibration/`, `runs/`, and `scores/`.
Source modules and small condition-rule assets are included. Prepared BTs,
inspection reports, generated descriptions, images, logs, caches, credentials,
and archived experiments are excluded.

## Validation and license

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```

Tests cover entry points, relocation, scoring, waypoint reference geometry, and
agreement between offline calibration reports and live detector thresholds.
They also check fresh gripper capture before generation and run a copied
project worker with reads from the source checkout explicitly blocked.
Simulator capture and provider calls are mocked in these isolation tests.
Calibration was also checked against one task's recorded clean telemetry.
Python dependencies were also installed into a fresh temporary environment and
all 110 tests passed from a fresh Git clone. That check does not validate a new
laptop's graphics drivers or replace the simulator and API trial above.
The local environment can also run simulator checks; paid provider stages need
credentials and separate validation.

The existing parent project's license is included unchanged in `LICENSE`.
External simulator, backend, and model dependencies retain their own licenses.
