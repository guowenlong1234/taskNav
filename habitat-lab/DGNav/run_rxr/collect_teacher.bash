#!/usr/bin/env bash
set -euo pipefail

export GLOG_minloglevel=2
export MAGNUM_LOG=quiet
export PYTHONUNBUFFERED=1

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
dgnav_dir="$(cd "${script_dir}/.." && pwd)"
preferred_habitat_repo_root="$(cd "${dgnav_dir}/.." && pwd -P)"

if [[ -n "${DGNAV_HABITAT_REPO_ROOT:-}" ]]; then
      requested_habitat_repo_root="$(cd "${DGNAV_HABITAT_REPO_ROOT}" && pwd -P)"
      if [[ "${requested_habitat_repo_root}" != "${preferred_habitat_repo_root}" ]]; then
            echo "[collect_teacher.bash] Refusing Habitat repo root ${requested_habitat_repo_root}" >&2
            echo "[collect_teacher.bash] Expected clean worktree root ${preferred_habitat_repo_root}" >&2
            exit 1
      fi
fi

habitat_repo_root="${preferred_habitat_repo_root}"
export DGNAV_HABITAT_REPO_ROOT="${habitat_repo_root}"
export PYTHONPATH="${habitat_repo_root}/habitat-lab:${habitat_repo_root}/habitat-baselines:${PYTHONPATH:-}"

conda_env="${CONDA_ENV:-py3-9}"
exp_name="${EXP_NAME:-rxr_teacher_collect}"
exp_config="${EXP_CONFIG:-run_rxr/collect_teacher.yaml}"
extra_opts=("$@")

if [[ -n "${NUM_ENVIRONMENTS:-}" ]]; then
      extra_opts=("NUM_ENVIRONMENTS" "${NUM_ENVIRONMENTS}" "${extra_opts[@]}")
fi
if [[ -n "${COLLECTOR_OUTPUT_DIR:-}" ]]; then
      extra_opts=("COLLECTOR.output_dir" "${COLLECTOR_OUTPUT_DIR}" "${extra_opts[@]}")
fi

cmd=(
      python
      collect_teacher.py
      --exp_name "${exp_name}"
      --exp-config "${exp_config}"
)
cmd+=("${extra_opts[@]}")

cd "${dgnav_dir}"
echo "[collect_teacher.bash] exp_name=${exp_name}"
echo "[collect_teacher.bash] exp_config=${exp_config}"
echo "[collect_teacher.bash] num_envs_override=${NUM_ENVIRONMENTS:-<none>}"

if [[ "${CONDA_DEFAULT_ENV:-}" == "${conda_env}" ]]; then
      "${cmd[@]}"
else
      conda run --no-capture-output -n "${conda_env}" "${cmd[@]}"
fi
