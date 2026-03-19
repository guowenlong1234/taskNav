# DGNav Research Notes

## Scope

This document summarizes a full pass over the `habitat-lab/DGNav` subtree in the current clean33 worktree:

- Path: `habitat-lab/DGNav`
- Scan date: 2026-03-19
- File count under this subtree: 229 total files
- Non-`__pycache__` files: 147
- Text/source/doc/config files read and indexed here: 145
- Non-text resource files:
  - `bert_config/xlm-roberta-base/sentencepiece.bpe.model`
  - `vlnce_baselines/models/etp/.backup`

I treated `__pycache__/*.pyc` as generated artifacts and did not analyze them. I did not attempt token-by-token manual inspection of `bert_config/xlm-roberta-base/tokenizer.json` or `bert_config/bert-base-uncased/vocab.txt`; instead I documented their role, size, and where they are consumed.

## Top-Level Conclusions

1. `DGNav` is not a small patch on top of original ETPNav; it is a layered fork that now contains:
   - Habitat 3.3 compatibility shims
   - a DINOv2 + MLP RGB pipeline
   - dynamic graph edge weighting inside the global cross-modal transformer
   - optional node gating
   - dynamic `loc_noise` for graph growth
   - an Oracle experiment stack with cache, scoped ghost replacement, trace logs, and recent Oracle-FT fine-tuning support
   - a separate pretraining pipeline under `pretrain_src`

2. The true runtime path for current R2R/RxR experiments is:
   - bash entry (`run_r2r/main.bash`, `run_rxr/main.bash`, or Oracle wrappers)
   - `run.py`
   - config merge in `vlnce_baselines/config/default.py`
   - trainer lookup through Habitat Baselines registry
   - `vlnce_baselines/ss_trainer_ETP.py`
   - `PolicyViewSelectionETP` / `ETP`
   - `GlocalTextPathNavCMT` in `vlnce_baselines/models/etp/vilmodel_cmt.py`

3. The repository currently contains three generations of logic at once:
   - legacy CMA / Seq2Seq / DAgger support
   - the active DGNav / SS-ETP path
   - recent Oracle-FT experiments and clean33 migration work

4. The codebase is mid-migration rather than fully cleaned:
   - active files in the current git worktree are already modified around Oracle-FT
   - some scripts and YAMLs still hardcode paths from the older `DGNav_new` tree
   - several legacy modules remain present but are not part of the main DGNav training path

5. The most important implementation facts I found:
   - `run.py` forcibly guards imports so Habitat-Lab/Baselines come from the local clean worktree.
   - `ss_trainer_ETP.py` overrides the base trainer and is the real heart of training/eval/inference.
   - `GraphMap` is the center of graph state, ghost creation/merge, Oracle writeback, and relative feature construction.
   - Oracle query/writeback is no longer eval-only in code; it now supports `enable_in_train` / `enable_in_eval`.
   - Oracle-FT fusion is implemented in `Policy_ViewSelection_ETP.py` by passing `base/raw/mask` ghost features into navigation and fusing them through `OracleResidualAdapter`.
   - The dynamic graph mechanism is implemented inside `GraphLXRTXLayer` by adding semantic and instruction-conditioned edge corrections on top of geometric pair distances.
   - The waypoint predictor is frozen and loaded from a checkpoint, but its heatmap predictor currently uses only depth features (`vis_x = depth_x`) even though RGB features are extracted and still used for panorama/node embeddings.

## Current Worktree State

`git status --short` shows this subtree is already dirty in seven files:

- `run_r2r/eval_oracle_ft_base.yaml`
- `run_r2r/run_oracle_eval.bash`
- `run_r2r/run_oracle_experiment.bash`
- `run_r2r/run_oracle_ft_train.bash`
- `run_r2r/train_oracle_ft_base.yaml`
- `vlnce_baselines/config/default.py`
- `vlnce_baselines/ss_trainer_ETP.py`

This matches what the code shows: the repository is currently in an active Oracle-FT integration phase.

## Architecture Map

### 1. Launch and import control

`run.py` is the real Python entrypoint. Its responsibilities are broader than a normal experiment launcher:

- `_prepend_local_habitat_paths()` forces `habitat-lab` and `habitat-baselines` imports to come from the sibling clean worktree.
- `_assert_local_import_roots()` raises if `habitat_baselines` or `vlnce_baselines` came from the wrong path.
- `_patch_habitat_legacy_config_api()` patches old DGNav assumptions onto Habitat-Lab/Baselines 0.3.x:
  - reintroduces `habitat.Config`-like YACS symbols
  - patches missing utility names like `try_cv2_import`
  - patches `read_write()` so old YACS configs work against OmegaConf-based Habitat 3.3
  - creates `habitat_baselines.config.default._C` when old code expects it
- `run_exp()` merges config files, injects `EXP_NAME`, rewrites output directories, sets RNG seeds, and dispatches to the registered trainer.

Important detail: `run.py` is stricter than the bash wrappers. Even if the shell `PYTHONPATH` is wrong, `run.py` still rejects a mismatched `DGNAV_HABITAT_REPO_ROOT`.

### 2. Config stack

The active config system is still YACS-style, centered on `vlnce_baselines/config/default.py`.

Key design details:

- Flat legacy DGNav config structure is preserved.
- `get_config()` supports comma-separated config stacks.
- `_normalize_legacy_oracle_opts()` and `_normalize_oracle_config()` map old uppercase Oracle keys to canonical lowercase keys.
- `refresh_policy` is normalized into the boolean pair:
  - `query_only_new_or_changed`
  - `requery_on_realpos_update`
- `ORACLE.apply_mode` is normalized to `hard` or `soft`.
- `ORACLE.target_ghost_scope` is validated against:
  - `all`
  - `new_only`
  - `local_frontier`
  - `top1_shadow`
- `ORACLE.query_heading_strategy` accepts:
  - `face_frontier`
  - `travel_dir`
  - `multi_heading_pool`

Important finding: `multi_heading_pool` is accepted by config normalization, but `oracle_manager.py` explicitly raises `NotImplementedError` for it in the current provider path.

### 3. Main training/eval/infer trainer

The active trainer is `RLTrainer` registered as `SS-ETP` in `vlnce_baselines/ss_trainer_ETP.py`.

This file is the single most important file in the project. It owns:

- environment setup
- checkpoint save/load
- Oracle summary logs
- Oracle scope trace logs
- dynamic graph weight dumps
- Oracle-FT optimizer grouping
- `GraphMap` batch packing
- full rollout logic for train/eval/infer

The real rollout order in `rollout()` is:

1. reset/resume vector envs
2. tokenize instructions
3. encode text once for the current batch
4. at each step:
   - run waypoint predictor
   - build panorama token batch
   - run panorama encoder
   - pool panorama embedding into current node feature
   - query current position/orientation
   - convert candidate angle/distance to graph proposals
   - optionally fetch candidate real positions from env
   - optionally compute dynamic or random `loc_noise`
   - update each `GraphMap`
   - optionally run Oracle scope selection + query + writeback
   - pack global graph tensors via `_nav_gmap_variable()`
   - run navigation head
   - compute teacher action in graph space
   - compute CE loss during training
   - choose action by sample or argmax
   - translate graph action into environment control action
   - step envs
   - collect eval metrics or inference paths
   - pause completed envs

### 4. Model decomposition

`PolicyViewSelectionETP` wraps `ETP`, which internally exposes four modes:

- `language`
- `waypoint`
- `panorama`
- `navigation`

The active global planner backbone is `GlocalTextPathNavCMT` from `vlnce_baselines/models/etp/vilmodel_cmt.py`.

Its major pieces are:

- text embeddings and language encoder
- panorama/image embedding stack
- global graph encoder
- global action prediction head

### 5. Graph state

`GraphMap` in `vlnce_baselines/models/graph_utils.py` stores:

- real visited nodes: positions, embeddings, step ids
- ghost nodes: per-observation positions, mean position, accumulated embeddings, frontier parents
- optional real ghost landing positions when available
- shortest path / shortest distance maps over real nodes
- Oracle raw ghost embeddings and metadata
- book-keeping for newly created ghosts and Oracle scope/writeback

This file is where DGNav's graph semantics live.

### 6. Oracle subsystem

The Oracle path is split across:

- `vlnce_baselines/oracle/types.py`
- `vlnce_baselines/oracle/cache.py`
- `vlnce_baselines/oracle/providers.py`
- `vlnce_baselines/oracle/oracle_manager.py`

The provider does not invent a separate encoder. It reuses the live policy:

- peek panorama at the query pose through env API
- tokenize instruction
- run `policy.net(mode="waypoint")`
- run `policy.net(mode="panorama")`
- average pooled panorama tokens become the Oracle feature

This means Oracle features are "future-node average panorama embeddings in the model's own feature space", not a separate model family.

## Detailed Findings

### A. Clean33 / Habitat 3.3 migration layer

The codebase contains a full compatibility shell for running old DGNav logic on Habitat-Lab/Baselines 0.3.3:

- `run.py` patches config/runtime APIs before any DGNav imports.
- `habitat_extensions/__init__.py` patches `HabitatSimActions` lookup so legacy uppercase action names still resolve on newer Habitat.
- `vlnce_baselines/common/env_utils.py` translates legacy task config layout into fields expected by Habitat core.
- `habitat_extensions/habitat_simulator.py` reimplements simulator creation with compatibility for changed Habitat-Sim sensor APIs.

The migration is pragmatic rather than elegant: the code adapts the old DGNav assumptions instead of rewriting the project into modern Hydra/OmegaConf idioms.

### B. The actual R2R / RxR workflow

For current experiments:

- `run_r2r/main.bash` is the normal R2R entry and explicitly rejects a mismatched worktree root.
- `run_rxr/main.bash` is the RxR entry, but its root guarding is looser than `run_r2r/main.bash`.
- Oracle experiments are wrapped by:
  - `run_r2r/run_oracle_ft_train.bash`
  - `run_r2r/run_oracle_eval.bash`
  - `run_r2r/run_oracle_experiment.bash`

`run_oracle_experiment.bash` is not a generic runner; it is a preset dispatcher for named experiments like:

- `b0_train`
- `b1_train`
- `b1_eval_on_fixed500`
- `b2_train`
- `full_oracle_train`

Important findings inside this preset script:

- many eval presets still require manual filling of `ckpt_path=""`
- multiple presets are clearly intended as experiment notebooks encoded in bash
- `full_oracle_train` comments emphasize Oracle training on the baseline path, but it sets `IL_BACK_ALGO="teleport"` instead of `control`, which is inconsistent with the surrounding Oracle experiment narrative

### C. Active model: ETP + DGNav extensions

`vlnce_baselines/models/Policy_ViewSelection_ETP.py` is where most recent DGNav-specific model changes landed.

Important details:

1. `PolicyViewSelectionETP.from_config()` copies `TORCH_GPU_ID` into `config.MODEL` and also mirrors Oracle soft alpha into `MODEL.ORACLE_SOFT_ALPHA`.

2. `EffoNavDinoV2Encoder`:
   - loads a local DINOv2 repository through `torch.hub.load(..., source="local")`
   - freezes the DINO backbone
   - adds a 3-layer trainable projector from 384 to 512
   - can load projector weights from:
     - the current fine-tune checkpoint
     - a fallback pretrained checkpoint
   - prints a one-time RGB tensor stats debug line on first forward

3. `ETP.__init__()`:
   - constructs VLN-BERT CMT backbone
   - builds frozen depth encoder
   - chooses RGB extractor by config:
     - `clip`
     - `dino` with `EffoNav`
   - precomputes 12-view angle features
   - optionally creates `OracleResidualAdapter`

4. `ETP.forward(mode="navigation")`:
   - accepts not only `gmap_img_fts`
   - but also:
     - `gmap_base_img_fts`
     - `gmap_oracle_raw_fts`
     - `gmap_oracle_masks`
   - when Oracle-FT is enabled, fusion happens here, inside the policy, not inside `GraphMap`

This matches the design intent described in recent Oracle-FT docs.

### D. Waypoint prediction path

The waypoint path is easy to misunderstand because RGB and depth are both extracted.

What actually happens:

- `ETP.forward(mode="waypoint")` builds 12-view RGB and depth minibatches.
- It runs:
  - `self.rgb_encoder`
  - `self.depth_encoder`
- It passes both tensors to `BinaryDistPredictor_TRM`.

But inside `vlnce_baselines/waypoint_pred/TRM_net.py`, the actual heatmap predictor does:

- build `depth_x`
- ignore the RGB branch
- set `vis_x = depth_x`

So the waypoint heatmap is currently depth-driven only.

This is one of the most important concrete findings in the repository:

- RGB features are still extracted and later used as candidate/panorama appearance features.
- But the actual waypoint heatmap predictor currently ignores RGB for scoring.

### E. Dynamic graph implementation

The dynamic graph lives inside `GraphLXRTXLayer` in `vlnce_baselines/models/etp/vilmodel_cmt.py`.

It is not a separate graph module; it is injected into graph self-attention.

Per layer, when `use_dynamic_graph=True`, the code builds:

- `w1`: fixed geometric foundation weight, stored as a non-trainable parameter for logging
- `w2`: semantic edge weight
- `w3`: instruction edge weight
- `semantic_sim_mlp`
- `instruction_rel_mlp`
- optional distance embedding MLP for geometry-conditioned semantic edges

The fusion is:

- start from geometric pair distances
- optionally add semantic similarity correction
- optionally add instruction relevance correction
- then add the resulting tensor directly into the self-attention mask/bias

This means DGNav's "dynamic graph" is implemented as attention bias modulation rather than explicit graph rewiring in `networkx`.

### F. Node gating implementation

Node gating is also implemented inside `GraphLXRTXLayer`.

The gating logic:

- summarize instruction into a global vector
- concatenate each node feature with this instruction summary
- predict a scalar gate per node
- apply residual amplification: `V_new = V_old + V_old * gate`

This is a lightweight post-cross-attention filtering mechanism before graph self-attention.

### G. `GraphMap` semantics

`GraphMap.update_graph()` performs all graph growth and ghost merging:

- current real node is always inserted into `graph_nx`
- if `prev_vp` exists, connect the previous real node to the current one by Euclidean distance
- each candidate waypoint is localized against existing real nodes first
- if no real node matches, it is optionally localized against ghost mean positions
- otherwise a new ghost is created

Important field semantics:

- `ghost_pos[gid]`: list of observed candidate positions for that ghost
- `ghost_mean_pos[gid]`: mean of candidate positions
- `ghost_embeds[gid] = [sum_embed, count]`
- `ghost_fronts[gid]`: real nodes from which the ghost was observed
- `ghost_real_pos[gid]`: list of env-probed real landing positions, only when enabled
- `ghost_parent_real_node[gid]`: source real node of the ghost's first creation
- `ghost_aug_pos`: ghost positions optionally perturbed by `ghost_aug`

Oracle-related additions in the current code:

- `ghost_oracle_embeds`
- `ghost_oracle_meta`
- `last_added_ghost_ids`
- `step_added_ghost_ids`
- `oracle_last_scope_ids`
- `oracle_last_written_ids`
- `oracle_last_skipped_ids`
- `oracle_write_step`

This is a clear sign that the A3 scope-ablation design doc has already partially landed in code.

### H. Oracle scope and writeback behavior

Current Oracle scope selection is implemented in the trainer:

- `all`
- `new_only`
- `local_frontier`
- `top1_shadow`

Current behavior:

- for `top1_shadow`, the trainer first runs a baseline planner pass with Oracle disabled
- if top-1 is a ghost, it queries Oracle only for that shadow target
- it optionally reruns the planner using the scoped Oracle writeback

Writeback is performed by `GraphMap.apply_oracle_embeds()`, not directly by the manager.

### I. Oracle query logic

`OracleExperimentManager.query_ghosts()` is more mature than the older docs imply.

Per ghost it does:

1. validate the ghost is alive and has real positions
2. compute `real_pos_mean`
3. resolve query position using:
   - primary strategy `ghost_real_pos_mean`
   - fallback `nearest_real_pos` or `ghost_mean_pos`
4. resolve query heading from frontier relation
5. decide whether re-query is necessary using previous Oracle metadata
6. check spatial cache
7. if cache miss, call provider
8. if success, attach metadata and optionally cache the result
9. accumulate statistics and trace logs

Important details:

- training-mode Oracle is already supported via `enable_in_train`
- cache entries are keyed by scene + position + heading
- cache stats distinguish intra-episode and cross-episode hits
- the manager writes per-query trace JSONL if enabled

### J. Oracle-FT implementation status

The recent Oracle-FT design doc is not just aspirational; much of it is already implemented:

- `MODEL.ORACLE_FT` exists in config
- `_nav_gmap_variable()` now returns:
  - base graph image features
  - raw Oracle ghost features
  - Oracle masks
- `Policy_ViewSelection_ETP.py` defines `OracleResidualAdapter`
- navigation fuses base and Oracle projections
- trainer adds Oracle-FT optimizer groups
- trainer logs Oracle-FT feature statistics and gradient norms

Current Oracle-FT train scopes:

- `oracle_only`
- `baseline_plus_oracle_adapter`

`oracle_only` freezes almost everything except selected Oracle adapter/downstream parts.

### K. Dynamic `loc_noise`

This is a distinct DGNav addition and not just config clutter.

Inside `ss_trainer_ETP.py`, graph growth can use:

- fixed `loc_noise`
- random `loc_noise`
- dynamic `loc_noise`

Dynamic `loc_noise` is derived from the standard deviation of candidate waypoint angles, with selectable mapping:

- `linear`
- `sigmoid`
- `exponential`

This is a direct implementation of DGNav's "graph granularity should adapt to local ambiguity" idea.

### L. Environment and action execution

The environment stack in `vlnce_baselines/common/environments.py` and `habitat_extensions/nav.py` is custom.

The project does not simply issue Habitat primitive actions from the planner. Instead:

- planner chooses a ghost or stop target in graph space
- environment wrapper converts that into:
  - teleport or control-style backtracking to a frontier node
  - then low-level motion toward the target ghost

Three custom high-to-low actions exist:

- `MoveHighToLowAction`
- `MoveHighToLowActionEval`
- `MoveHighToLowActionInference`

These differ mainly in whether path/collision traces are retained.

### M. Evaluation metrics

The code supports the standard VLN metrics plus some DGNav-specific accounting:

- `success`
- `oracle_success`
- `spl`
- `ndtw`
- `sdtw`
- `steps_taken`
- `path_length`
- `collisions`
- `ghost_cnt`
- `episode_time`

During eval, `ss_trainer_ETP.py` computes many of these manually from recorded trajectories rather than trusting Habitat defaults alone.

### N. Pretraining branch

`pretrain_src` is effectively a parallel project embedded inside DGNav.

It has its own:

- parser
- data pipeline
- optimizer stack
- transformer/model definitions
- training loop
- feature extraction tools

Its default tasks are:

- `mlm`
- `sap`

`mrc` exists in code but is not enabled in the default JSON config.

Important relation to downstream DGNav:

- the pretraining model already supports an optional RGB projector for raw DINO features
- the downstream DINO projector loading logic is clearly meant to reuse or inherit weights from this pretraining path

### O. Legacy modules still present

The repository still ships legacy branches:

- DAgger trainer
- CMA / Seq2Seq config families
- PREVALENT-style VLN-BERT
- custom pytorch-transformers copies for waypoint prediction

These are useful for historical reference or fallback baselines, but they are not the active path for current DGNav R2R experiments.

## Concrete Risks / Inconsistencies

### 1. Several files still hardcode old `DGNav_new` paths

Notable examples:

- `run_r2r/eval_oracle_a3_base.yaml`
- `run_r2r/eval_oracle_o1.yaml`
- comments/examples in `auto_benchmark_num_env.py`
- comments/examples in `batch_eval_checkpoints.py`
- many path references inside `记录.md`

This does not necessarily break runtime if overridden from bash, but it makes the repo less self-consistent.

### 2. Many Oracle experiment presets require manual checkpoint filling

`run_r2r/run_oracle_experiment.bash` contains several presets with `ckpt_path=""`.

This means the script is acting as an experiment notebook template, not a fully ready one-click runner.

### 3. `multi_heading_pool` is config-valid but not runtime-supported

`vlnce_baselines/config/default.py` allows it.
`vlnce_baselines/oracle/oracle_manager.py` raises `NotImplementedError`.

### 4. The waypoint predictor ignores RGB in its scoring path

This is a subtle but major architectural fact:

- RGB is extracted
- RGB is used later for node/panorama features
- but waypoint heatmap logits are driven by depth only

This could be intentional, but it is worth calling out because it is easy to miss if reading only the high-level code.

### 5. `pretrain_src/pretrain_src/train_r2r.py` mixes `opts` and global `args`

Inside `main(opts)`, the loss scaling path uses:

- `args.gradient_accumulation_steps`

instead of:

- `opts.gradient_accumulation_steps`

This works when run through the current script entry because `args` is a module-global created in `__main__`, but it is brittle and not a clean function-local implementation.

### 6. `vlnce_baselines/models/policy.py` is not production-ready

`ILPolicy.act()` and `act2()` still contain `print(...)` plus `pdb.set_trace()`.

This is harmless for the active DGNav path because the custom trainer bypasses them, but it confirms this file is only a base wrapper, not a clean standalone policy interface.

### 7. DAgger path looks more legacy than active

`vlnce_baselines/dagger_trainer.py` still contains:

- explicit debug prints
- a terminal `pdb.set_trace()`

This strongly suggests the DAgger branch is not maintained at the same level as `SS-ETP`.

### 8. The codebase still mixes "documentation as design" and "documentation as logbook"

There are two different kinds of docs:

- forward-looking design docs:
  - `A3_Oracle_Scope_Code_Implementation_Guide.md`
  - `ETPNav_DGNav_ghost_experiment_plan_v2.md`
  - `ETPNav_DGNav_ghost_experiment_plan_v3_finetune.md`
  - `Oracle_Train_Finetune_DevDoc_v1.md`
- historical experiment notes:
  - `记录.md`
  - `CLEAN33_TRAIN_RUNBOOK_2026-03-15.md`

The implementation is already partly beyond older docs in some places and still behind them in others.

## How Main Data Moves Through the System

### Train / Eval / Infer (active SS-ETP path)

1. Instruction text
   - tokenized from env observations by `extract_instruction_tokens()`
   - encoded by `policy.net(mode="language")`

2. 12-view panorama
   - RGB and depth views are gathered from `rgb/rgb_30/...` and `depth/depth_30/...`
   - reordered to avoid Habitat 3.3 dict-key ordering bugs
   - encoded by frozen visual backbones

3. Waypoint heatmap
   - predicted by the frozen TRM waypoint predictor
   - NMS extracts up to 5 candidate waypoints

4. Candidate/panorama embeddings
   - candidate RGB/depth features are pooled
   - all 12 view tokens are passed through the panorama encoder
   - pooled panorama feature becomes the current real-node embedding

5. Graph update
   - current node is added to `GraphMap`
   - candidate waypoints become node edges or ghost updates
   - shortest paths/distances are recomputed over real nodes

6. Optional Oracle
   - select scope ids
   - query future panorama embeddings for selected ghosts
   - write raw Oracle embeddings back into `GraphMap`

7. Global graph packing
   - `_nav_gmap_variable()` turns graph state into padded tensors:
     - ids
     - step ids
     - visual features
     - relative position features
     - visited masks
     - pairwise distances
     - optional Oracle raw/base/mask tensors

8. Navigation
   - graph tokens are fused with text in `GlocalTextPathNavCMT`
   - dynamic graph edge terms alter graph self-attention
   - action logits over graph nodes are produced

9. Action translation
   - graph action becomes either stop or high-to-low control action
   - env wrapper performs backtracking / control execution

10. Metrics / paths
   - training accumulates CE loss
   - eval records per-episode metrics
   - inference writes R2R JSON or RxR JSONL submission format

## Directory-by-Directory Notes

### Top level docs and scripts

- `README.md`: public project description; presents DGNav as Dynamic Topology Awareness for VLN-CE, with installation, dataset, weight, and run instructions.
- `run.py`: strict launcher plus Habitat 3.3 compatibility layer.
- `requirements.txt`: old Python 3.7-era requirements; still references TensorFlow 1.13.1.
- `requirements-py39.txt`: practical runtime requirements for the clean Python 3.9 path; removes TensorFlow.
- `auto_benchmark_num_env.py`: benchmark script for `NUM_ENVIRONMENTS` throughput using train rollout timing logs.
- `batch_eval_checkpoints.py`: batch evaluator for downstream checkpoints, with plotting and best-checkpoint selection helpers.
- `CLEAN33_TRAIN_RUNBOOK_2026-03-15.md`: operational runbook for the clean33 worktree, with absolute commands and acceptance criteria.
- `Oracle_Train_Finetune_DevDoc_v1.md`: detailed design doc for Oracle-in-train plus Oracle-MLP fine-tuning.
- `A3_Oracle_Scope_Code_Implementation_Guide.md`: design doc for Oracle scope selection and shadow rerun experiments.
- `ETPNav_DGNav_ghost_experiment_plan_v2.md`: large research/design doc for Oracle diagnosis and streaming ghost work.
- `ETPNav_DGNav_ghost_experiment_plan_v3_finetune.md`: updated research/design doc elevating Oracle-aware fine-tuning to a main stage.
- `文档.md`: idea memo recommending a staged "trajectory-conditioned ghost enhancement" route rather than jumping directly to a world model.
- `记录.md`: long chronological experiment notebook with logs, metrics, and path references.

### `run_r2r`

- `iter_train.yaml`: main R2R training/eval config for active DGNav experiments.
- `r2r_vlnce.yaml`: task/simulator/dataset config for R2R continuous VLN.
- `main.bash`: standard R2R train/eval/infer launcher with strict clean-worktree guard.
- `eval_fixed500.yaml`: constrains eval to the 500-episode fixed subset.
- `episode_subsets/r2r_val_unseen_fixed500.txt`: the fixed500 episode id list; exactly 500 ids.
- `eval_oracle_a3_base.yaml`: full Oracle A3 evaluation base config; still hardcodes an old `DGNav_new` checkpoint path.
- `eval_oracle_a3_all.yaml`: A3 scope override for `all`.
- `eval_oracle_a3_new_only.yaml`: A3 scope override for `new_only`.
- `eval_oracle_a3_local_frontier.yaml`: A3 scope override for `local_frontier`.
- `eval_oracle_a3_top1_shadow.yaml`: A3 scope override for `top1_shadow` with planner rerun.
- `eval_oracle_o1.yaml`: older Oracle O1 evaluation config; also still hardcodes an old path.
- `train_oracle_ft_base.yaml`: base training override for Oracle-FT experiments.
- `eval_oracle_ft_base.yaml`: base evaluation override for Oracle-FT experiments.
- `oracle_ft_disabled.yaml`: disables Oracle-FT while keeping the config branch present.
- `oracle_ft_adapter_only.yaml`: enables only the Oracle adapter branch.
- `oracle_ft_xlayers.yaml`: enables Oracle adapter plus global encoder x-layer fine-tuning.
- `oracle_ft_eval_on.yaml`: keep Oracle on during eval.
- `oracle_ft_eval_off.yaml`: turn Oracle query off during eval while still loading Oracle-FT weights.
- `run_oracle_ft_train.bash`: low-level Oracle-FT training wrapper with env-var based overrides.
- `run_oracle_eval.bash`: low-level Oracle eval wrapper with env-var based overrides and optional CPU pinning.
- `run_oracle_experiment.bash`: preset dispatcher for named B0/B1/B2/full Oracle experiments.

### `run_rxr`

- `iter_train.yaml`: main RxR config using the same `SS-ETP` trainer.
- `rxr_vlnce.yaml`: RxR task/simulator/dataset config, including multilingual settings and HFOV 63.
- `main.bash`: standard RxR train/eval/infer launcher; less strict than the R2R launcher.

### `tools`

- `check_clean33_env.py`: verifies Habitat/Habitat Baselines/Habitat-Sim versions and import roots.
- `setup_clean33_local_assets.py`: symlinks shared non-tracked assets (`data`, `pretrained`, DINO repo copy, pretrain datasets/features, Habitat Baselines IL data) into the clean worktree.
- `select_batch_eval_checkpoint.py`: ranks batch-eval results by primary/secondary metrics and prints a JSON summary.

### `habitat_extensions`

- `__init__.py`: import registration plus HabitatSim action-name compatibility patch.
- `config/default.py`: lightweight YACS root for custom actions, sensors, measurements, and dataset extensions.
- `config/r2r_vlnce.yaml`: legacy/short-episode R2R task config for habitat_extensions path.
- `config/rxr_vlnce_en.yaml`: RxR English config.
- `config/rxr_vlnce_hi.yaml`: RxR Hindi config.
- `config/rxr_vlnce_te.yaml`: RxR Telugu config.
- `task.py`: custom dataset loaders for R2R and RxR, including role/language filtering and allowed-episode filtering.
- `sensors.py`: global GPS, orientation, shortest-path, Oracle progress, and RxR instruction sensors.
- `nav.py`: custom high-to-low actions for train/eval/inference.
- `measures.py`: custom metrics including position traces, path length, Oracle metrics, nDTW, sDTW, and top-down map.
- `maps.py`: top-down map drawing helpers and DGNav-specific map colors.
- `utils.py`: visualization, video generation, waypoint/map overlay, pose conversion helpers.
- `shortest_path_follower.py`: old Habitat shortest-path follower copied for compatibility.
- `obs_transformers.py`: per-sensor crop/resize and cube-map-to-equirect transformer wrappers.
- `habitat_simulator.py`: custom `Sim-v1` adapter for Habitat-Sim API compatibility.

### `vlnce_baselines/common`

- `aux_losses.py`: small auxiliary-loss registry singleton.
- `ops.py`: tensor padding, masks, and Transformer encoder factory.
- `transformer.py`: local Transformer implementation reused by common and pretrain code.
- `utils.py`: instruction token extraction plus a few geometry/helpers.
- `env_utils.py`: env construction and legacy config-to-Habitat-core synchronization.
- `environments.py`: custom RLEnv wrappers for training/eval/inference, including control-mode execution helpers.
- `base_il_trainer.py`: legacy base trainer; still useful for CMA / VLNBERT / inference paths, but `SS-ETP` overrides most active behavior.
- `recollection_dataset.py`: iterable teacher recollection dataset for DAgger/recollection-style training.

### `vlnce_baselines/config`

- `default.py`: main experiment config and Oracle normalization logic.
- `nonlearning.yaml`: random-agent evaluation/inference config.
- `r2r_configs/cma.yaml`: legacy CMA baseline config.
- `r2r_configs/cma_aug.yaml`: CMA augmentation variant.
- `r2r_configs/cma_aug_tune.yaml`: CMA augmentation fine-tune variant.
- `r2r_configs/cma_da.yaml`: CMA DAgger variant.
- `r2r_configs/cma_da_aug_tune.yaml`: CMA DAgger + aug tune variant.
- `r2r_configs/cma_pm.yaml`: CMA + progress monitor variant.
- `r2r_configs/cma_pm_aug.yaml`: CMA + progress monitor + augmentation variant.
- `r2r_configs/cma_pm_aug_tune.yaml`: CMA PM aug tune variant.
- `r2r_configs/cma_pm_da.yaml`: CMA PM + DAgger variant.
- `r2r_configs/cma_pm_da_aug_tune.yaml`: CMA PM DA aug tune variant.
- `r2r_configs/cma_sf.yaml`: CMA teacher-forcing / supervised-following style config.
- `r2r_configs/cma_ss.yaml`: CMA self-supervised / SS trainer config.
- `r2r_configs/seq2seq.yaml`: legacy Seq2Seq baseline config.
- `r2r_configs/seq2seq_aug.yaml`: Seq2Seq augmentation variant.
- `r2r_configs/seq2seq_aug_tune.yaml`: Seq2Seq aug tune variant.
- `r2r_configs/seq2seq_da.yaml`: Seq2Seq DAgger variant.
- `r2r_configs/seq2seq_pm.yaml`: Seq2Seq + progress monitor variant.
- `r2r_configs/seq2seq_pm_aug.yaml`: Seq2Seq PM + augmentation variant.
- `r2r_configs/seq2seq_pm_da_aug_tune.yaml`: Seq2Seq PM DA aug tune variant.
- `r2r_configs/test_set_inference.yaml`: legacy CMA test-set inference config.

### `vlnce_baselines`

- `__init__.py`: imports the registered trainers, envs, and policy so Habitat registry sees them.
- `utils.py`: miscellaneous allocation and distributed helper utilities, older than `common/utils.py`.
- `dagger_trainer.py`: legacy DAgger trainer path; present but not the active DGNav trainer.
- `ss_trainer_ETP.py`: active DGNav trainer with Oracle, dynamic loc-noise, Oracle-FT, and full rollout logic.

### `vlnce_baselines/models`

- `__init__.py`: empty marker.
- `policy.py`: minimal IL policy wrapper; contains legacy `pdb` stubs in `act()`.
- `utils.py`: angle-feature helpers for candidate direction encoding.
- `graph_utils.py`: graph geometry, Floyd graph, `GraphMap`, ghost state, Oracle writeback.
- `Policy_ViewSelection_ETP.py`: active DGNav policy, DINO encoder, Oracle adapter, waypoint/panorama/navigation forwarding.

### `vlnce_baselines/models/encoders`

- `instruction_encoder.py`: RNN instruction encoder for older/legacy policy paths.
- `resnet_encoders.py`: depth encoder, ResNet50 RGB encoder, and CLIP RGB encoder.

### `vlnce_baselines/models/etp`

- `vlnbert_init.py`: builds the active `GlocalTextPathNavCMT` backbone and injects dynamic-graph/node-gating config.
- `vilmodel_cmt.py`: the active cross-modal transformer implementation with dynamic graph and node gating.
- `.backup`: extensionless backup snapshot of older `vilmodel_cmt`-style code; not part of the active import path.

### `vlnce_baselines/models/vlnbert`

- `vlnbert_init.py`: legacy PREVALENT VLN-BERT loader.
- `vlnbert_PREVALENT.py`: legacy recurrent VLN-BERT implementation, not the active DGNav backbone.

### `vlnce_baselines/oracle`

- `types.py`: dataclasses for query specs, results, and trajectory observation buffer items.
- `cache.py`: radius/heading based scene-bucketed Oracle cache.
- `providers.py`: current Oracle provider that peeks future panorama obs and reuses the active policy to encode them.
- `oracle_manager.py`: query target resolution, cache handling, trace writing, per-step stats, and writeback coordination.

### `vlnce_baselines/waypoint_pred`

- `TRM_net.py`: frozen waypoint heatmap predictor.
- `utils.py`: NMS and attention-mask helpers for waypoint prediction.
- `transformer/waypoint_bert.py`: reduced visual Transformer for waypoint prediction.
- `transformer/pytorch_transformer/modeling_bert.py`: bundled old BERT implementation used by waypoint predictor.
- `transformer/pytorch_transformer/modeling_utils.py`: bundled old HuggingFace utility code.
- `transformer/pytorch_transformer/file_utils.py`: bundled cache/file utility code.

### `pretrain_src`

- `run_pt/run_r2r.bash`: simple pretraining launcher.
- `run_pt/r2r_model_config_dep.json`: pretraining model structure config; already includes RGB projector settings for DINO.
- `run_pt/r2r_pretrain_habitat.json`: pretraining task/data/optimizer config.
- `pretrain_src/parser.py`: pretraining CLI parser with JSON-config override.
- `pretrain_src/train_r2r.py`: main pretraining training loop.
- `pretrain_src/extract_dino_features.py`: render 36-view Matterport panoramas and extract DINOv2 features into HDF5.
- `pretrain_src/data/__init__.py`: package marker.
- `pretrain_src/data/common.py`: nav-graph, angle-feature, padding, and softmax helpers for pretraining data.
- `pretrain_src/data/dataset.py`: REVERIE/R2R text-path dataset and graph feature construction for pretraining tasks.
- `pretrain_src/data/loader.py`: meta-loader, prefetch-loader, and distributed data-loader construction.
- `pretrain_src/data/tasks.py`: MLM/MRC/SAP/OG datasets and collators.
- `pretrain_src/model/__init__.py`: package marker.
- `pretrain_src/model/ops.py`: same kind of tensor ops used in the downstream model.
- `pretrain_src/model/transformer.py`: local Transformer implementation used in pretraining.
- `pretrain_src/model/vilmodel.py`: pretraining glocal text-path model backbone; already supports RGB projector.
- `pretrain_src/model/pretrain_cmt.py`: heads and task forwarding for MLM/MRC/SAP pretraining.
- `pretrain_src/optim/__init__.py`: optimizer package marker.
- `pretrain_src/optim/adamw.py`: custom AdamW implementation.
- `pretrain_src/optim/lookahead.py`: Lookahead wrapper.
- `pretrain_src/optim/misc.py`: optimizer builder.
- `pretrain_src/optim/radam.py`: RAdam / PlainRAdam / AdamW implementations.
- `pretrain_src/optim/ralamb.py`: Ralamb optimizer.
- `pretrain_src/optim/rangerlars.py`: RangerLars helper.
- `pretrain_src/optim/sched.py`: LR schedules.
- `pretrain_src/utils/__init__.py`: package marker.
- `pretrain_src/utils/distributed.py`: distributed init and gather helpers.
- `pretrain_src/utils/logger.py`: logger, tensorboard logger, running meter.
- `pretrain_src/utils/misc.py`: RNG, dropout, CUDA, and model wrapping helpers.
- `pretrain_src/utils/save.py`: training metadata and checkpoint saver.

### `bert_config`

- `bert-base-uncased/config.json`: local BERT config used for R2R text backbone init.
- `bert-base-uncased/vocab.txt`: 30,522-token vocabulary file.
- `xlm-roberta-base/config.json`: local XLM-R config used for RxR text backbone init.
- `xlm-roberta-base/tokenizer.json`: very large tokenizer JSON for local XLM-R tokenizer loading.
- `xlm-roberta-base/sentencepiece.bpe.model`: sentencepiece model file paired with the tokenizer config.

## Final Assessment

This repository is coherent enough to run serious experiments, but it is not "fully productized". It is best understood as:

- a research codebase with a strong active path
- plus multiple historical branches and experiment notebooks preserved in-tree
- plus a substantial clean33 migration/compatibility layer
- plus a recent Oracle-FT branch that is already partially integrated, not merely planned

The active DGNav identity of the codebase is defined by four files more than any others:

- `run.py`
- `vlnce_baselines/ss_trainer_ETP.py`
- `vlnce_baselines/models/Policy_ViewSelection_ETP.py`
- `vlnce_baselines/models/etp/vilmodel_cmt.py`

And the most important supporting stateful module is:

- `vlnce_baselines/models/graph_utils.py`

If I had to summarize the project in one sentence:

`DGNav` is a Habitat-3.3-compatible, graph-based VLN system derived from ETP-style planning, extended with dynamic graph attention, adaptive ghost construction, DINO-based visual features, and an increasingly sophisticated Oracle experimentation and fine-tuning stack.
