# Copyright (c) Meta Platforms, Inc. and affiliates.
# See file LICENSE.txt in the main directory.

"""SLURM job launcher.

Submits training and fine-tuning jobs, single- or multi-node.

Usage:
    python -m launchers.slurm run name=mmplus config=configs/mosaics/memory_mosaics+.yaml nodes=32
    python -m launchers.slurm run name=sft_metamath config=configs/sft/metamath.yaml
    python -m launchers.slurm cancel pattern=mmplus
"""

import getpass
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import yaml


JOB_ID_REGEX = re.compile(r"Submitted batch job (\d+)", flags=re.IGNORECASE)


SBATCH_TEMPLATE = """#!/bin/bash

{sbatch_extras}
#SBATCH --job-name={name}
#SBATCH --nodes={nodes}
#SBATCH --gres=gpu:{ngpus}
#SBATCH --ntasks-per-node={ngpus}
#SBATCH --cpus-per-task={ncpu}
#SBATCH --time={time}

#SBATCH --output={dump_dir}/%j/%j.stdout
#SBATCH --error={dump_dir}/%j/%j.stderr

#SBATCH --open-mode=append
#SBATCH --signal=USR2@120
#SBATCH --distribution=block

{env_setup}

cd {run_dir}

export LAUNCH_WITH="SBATCH"
export DUMP_DIR={dump_dir}
# Per-job JIT build dir under the dump folder, NOT the shared
# ~/.cache/torch_extensions: a stale FileBaton lock left by a previous or killed
# program in the shared cache makes new runs hang forever waiting on it. A fresh
# per-job dir has no stale lock. (torch.utils.cpp_extension.load creates it.)
export TORCH_EXTENSIONS_DIR=$DUMP_DIR/torch_extensions/$SLURM_JOB_ID
srun {log_output} --label -n {tasks} -N {nodes} python -u -m {script} config=$DUMP_DIR/base_config.yaml
EXIT_CODE=$?
# Requeue (auto-resume from the last checkpoint) only on a genuine failure.
# A run that reached args.steps writes $DUMP_DIR/training_complete; in that case
# do NOT requeue even if the process exits non-zero during distributed teardown
# (which would otherwise relaunch a finished job that immediately re-saves and exits).
if [ $EXIT_CODE -ne 0 ] && [ ! -f "$DUMP_DIR/training_complete" ]; then
    echo "Training exited $EXIT_CODE without completion marker -> requeueing $SLURM_JOB_ID"
    scontrol requeue $SLURM_JOB_ID
fi
"""


@dataclass
class LaunchArgs:
    name: str = ""
    config: str = ""
    root_dump_dir: str = ""
    script: str = "training.train"
    launcher: str = "sbatch"       # sbatch or bash (if in salloc)
    nodes: int = 1
    ngpus: int = 8
    ncpu: int = 16
    mem: str = ""                  # e.g. 1000G; empty = cluster default
    time: int = 4320               # 3 days in minutes
    partition: str = ""
    account: str = ""
    qos: str = ""
    exclude: str = ""
    constraint: str = ""
    dependency: str = ""
    reservation: str = ""
    conda_env: str = ""
    no_requeue: bool = False
    dirs_exists_ok: bool = False
    preview: bool = False
    override: bool = False
    stdout: bool = False


def _get_root_dump_dir() -> str:
    return os.environ.get("DUMP_ROOT", str(Path.cwd() / "dumps"))


def _copy_code(src: str, dst: str) -> None:
    print(f"Copying code: {src} -> {dst}")
    rsync_cmd = (
        f"rsync -ar --copy-links "
        f"--exclude='.git/' "
        f"--exclude='__pycache__' "
        f"--exclude='.pytest_cache' "
        f"--exclude='*.pyc' "
        f"--exclude='checkpoints' "
        f"--exclude='/data_out' "
        f"--exclude='/dumps' "
        f"--exclude='/vanilla_memory_mosaics' "   # the ICLR code (and its 400 MB dataset)
        f"--exclude='.ruff_cache' "
        f"--exclude='*.stderr' "
        f"--exclude='*.stdout' "
        f"--exclude='*.log' "
        f"--exclude='*.pth' "
        f"{src}/ {dst}"
    )
    subprocess.call(rsync_cmd, shell=True)


def _detect_conda_env() -> str:
    python_path = shutil.which("python")
    if python_path:
        return str(Path(python_path).parent.parent)
    return ""


def _build_env_setup(conda_env: str) -> str:
    lines = []
    mamba_exe = os.environ.get("MAMBA_EXE")
    if mamba_exe:
        lines.append(f'eval "$({mamba_exe} shell hook -s bash)"')
        lines.append(f"micromamba activate {conda_env}")
    else:
        conda_exe = os.environ.get("CONDA_EXE", "conda")
        lines.append(f'eval "$({conda_exe} shell.bash hook)"')
        lines.append(f"source activate {conda_env}")
    return "\n".join(lines)


def _check_config(config: dict) -> None:
    """Load the final config as the job will, so that an unknown key, a bad value or a
    missing pretrained checkpoint fails now rather than after the job waited in the queue."""
    import tempfile

    from training.train import _load_config
    with tempfile.NamedTemporaryFile("w", suffix=".yaml") as f:
        yaml.safe_dump(config, f)
        f.flush()
        _load_config(f.name)


def run_launch(args: LaunchArgs) -> str | None:
    assert args.name, "name is required"
    assert args.config, "config is required"

    root_dump_dir = args.root_dump_dir or _get_root_dump_dir()
    print(root_dump_dir)
    dump_dir = Path(root_dump_dir) / args.name
    launch_date = f"{datetime.now(tz=UTC):%Y_%m_%d_%H_%M_%S}"
    run_dir = dump_dir / "code" / launch_date

    if dump_dir.exists() and args.override:
        confirm = input(f"Delete {dump_dir}? (yes/no): ")
        if confirm.lower() == "yes":
            shutil.rmtree(str(dump_dir))
        else:
            print("Cancelled.")
            return None

    config = yaml.safe_load(open(args.config))
    config["dump_dir"] = str(dump_dir)
    # The job runs from the code snapshot in run_dir, not from here, so a relative data
    # path would resolve against the snapshot. Pin them to where they are now.
    data_cfg = config.get("data") or {}
    for key in ("cluster_data_dir", "root_dir", "sft_data_path"):
        if data_cfg.get(key) and not os.path.isabs(data_cfg[key]):
            data_cfg[key] = os.path.abspath(data_cfg[key])
    # A tokenizer is a HuggingFace repo id or a local directory; pin only the latter.
    tok = data_cfg.get("tokenizer_path")
    if tok and not os.path.isabs(tok) and os.path.isdir(tok):
        data_cfg["tokenizer_path"] = os.path.abspath(tok)
    # The pretrained checkpoint of a fine-tuning run (continue_from), likewise.
    cont = config.get("continue_from") or {}
    if cont.get("checkpoint_dir") and not os.path.isabs(cont["checkpoint_dir"]):
        cont["checkpoint_dir"] = os.path.abspath(cont["checkpoint_dir"])
    if args.script == "training.train":
        _check_config(config)

    dump_dir.mkdir(parents=True, exist_ok=args.dirs_exists_ok)
    run_dir.mkdir(parents=True, exist_ok=args.dirs_exists_ok)

    _copy_code(str(Path.cwd()), str(run_dir))

    with open(dump_dir / "base_config.yaml", "w") as f:
        yaml.dump(config, f)

    conda_env = args.conda_env or _detect_conda_env()
    assert conda_env, "Could not detect conda environment"

    sbatch_extras = []
    if args.account:
        sbatch_extras.append(f"#SBATCH --account={args.account}")
    if args.qos:
        sbatch_extras.append(f"#SBATCH --qos={args.qos}")
    if args.partition:
        sbatch_extras.append(f"#SBATCH --partition={args.partition}")
    if args.mem:
        sbatch_extras.append(f"#SBATCH --mem={args.mem}")
    if args.exclude:
        sbatch_extras.append(f"#SBATCH --exclude={args.exclude}")
    if args.constraint:
        sbatch_extras.append(f"#SBATCH --constraint={args.constraint}")
    if args.dependency:
        sbatch_extras.append(f"#SBATCH --dependency={args.dependency}")
    if args.no_requeue:
        sbatch_extras.append("#SBATCH --no-requeue")
    if args.reservation:
        sbatch_extras.append(f"#SBATCH --reservation={args.reservation}")

    log_output = (
        "-o $DUMP_DIR/%j/%j_%t_log.out -e $DUMP_DIR/%j/%j_%t_log.err"
        if not args.stdout else ""
    )

    sbatch = SBATCH_TEMPLATE.format(
        name=args.name,
        nodes=args.nodes,
        ngpus=args.ngpus,
        ncpu=args.ncpu,
        mem=args.mem,
        time=args.time,
        partition=args.partition,
        dump_dir=str(dump_dir),
        run_dir=str(run_dir),
        tasks=args.nodes * args.ngpus,
        script=args.script,
        sbatch_extras="\n".join(sbatch_extras),
        env_setup=_build_env_setup(conda_env),
        log_output=log_output,
    )

    if args.preview:
        print(sbatch)
        return None

    sbatch_dir = dump_dir / ".slurm" / launch_date
    sbatch_dir.mkdir(parents=True, exist_ok=True)
    sbatch_file = sbatch_dir / "submit.slurm"
    sbatch_file.write_text(sbatch)

    output = subprocess.check_output(f"sbatch {sbatch_file}", shell=True, encoding="utf-8")
    print(output.strip())
    match = JOB_ID_REGEX.search(output)
    if match:
        job_id = match.group(1)
        print(f"Job {job_id} submitted.")
        (dump_dir / job_id).mkdir(exist_ok=True)
        shutil.copytree(str(sbatch_dir), str(dump_dir / job_id / "slurm_info"), dirs_exist_ok=True)
        return job_id
    return None


def run_cancel(pattern: str, substr: bool = False) -> None:
    user = getpass.getuser()
    output = subprocess.check_output(
        ["squeue", "-a", "--noheader", "-u", user, "-o", "%j %i"],
        text=True,
    ).strip()

    kill_ids = []
    for line in output.split("\n"):
        if not line.strip():
            continue
        parts = line.split()
        job_name, job_id = parts[0], re.sub(r"_\[.*\]$", "", parts[1])
        if job_name == pattern or (substr and pattern in job_name):
            kill_ids.append(job_id)

    if not kill_ids:
        print(f"No jobs found matching '{pattern}'")
        return

    print(f"About to cancel {len(kill_ids)} jobs.")
    confirm = input('Type "yes" to confirm: ')
    if confirm == "yes":
        subprocess.call(f"scancel {' '.join(kill_ids)}", shell=True)
        print("Done.")
    else:
        print("Cancelled.")


def launch_eval_job(
    checkpoint_dir: str,
    tasks: str,
    dump_dir: str,
    job_name: str,
    n_gpus: int = 1,
    partition: str = "",
    account: str = "",
    qos: str = "",
) -> str | None:
    """Submit an eval job via SLURM."""
    args = LaunchArgs(
        name=job_name,
        config="",
        script="evaluation.eval",
        nodes=max(1, n_gpus // 8),
        ngpus=min(n_gpus, 8),
        partition=partition,
        account=account,
        qos=qos,
    )
    # For eval, we write a minimal config
    eval_config = {
        "checkpoint_dir": checkpoint_dir,
        "tasks": tasks,
        "dump_dir": dump_dir,
    }
    dump_path = Path(dump_dir)
    dump_path.mkdir(parents=True, exist_ok=True)
    with open(dump_path / "base_config.yaml", "w") as f:
        yaml.dump(eval_config, f)

    return run_launch(args)


if __name__ == "__main__":
    argv = sys.argv
    if len(argv) < 2 or argv[1] not in ("run", "cancel"):
        print("Usage: python -m launchers.slurm [run|cancel] key=value ...")
        sys.exit(1)

    command = argv[1]
    kwargs = {}
    for arg in argv[2:]:
        if "=" in arg:
            k, v = arg.split("=", 1)
            if v.isdigit():
                v = int(v)
            elif v.lower() in ("true", "false"):
                v = v.lower() == "true"
            kwargs[k] = v

    if command == "run":
        args = LaunchArgs(**kwargs)
        run_launch(args)
    elif command == "cancel":
        run_cancel(kwargs.get("pattern", ""), substr=kwargs.get("substr", False))
