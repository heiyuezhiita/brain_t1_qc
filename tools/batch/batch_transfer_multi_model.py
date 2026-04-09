import multiprocessing
import subprocess
import time
import os
import shutil

# =========================
# Configuration
# =========================
QC_TRAIN_PATH = "transfer_train.py"
CFG_POOL_DIR = "configs/transfer/batch_transfer/cohort_b"
GPU_LIST = [0, 0]
PYTHON_BIN = "python"
CONFIG_EXT = ".yaml"
TRAINING_DIR_NAME = "training_cfg"
END_DIR_NAME = "end_cfg"
SLEEP_SEC = 1
TASK_TAG = "multi_process_task_script"
VERBOSE_MKDIR = True
# =========================


def mkdir_all(paths, verbose=True):
    """Create one or more directories if missing."""
    if isinstance(paths, str):
        paths = [paths]
    elif not isinstance(paths, (list, tuple)):
        raise TypeError(f"Expected str/list/tuple, got {type(paths)}")

    for path in paths:
        if not os.path.exists(path):
            os.makedirs(path)
            if verbose:
                print(f"mkdir: {path}")


def run_script_in_process(training_dir, end_dir, script_path, cfg_tmp_path, gpu_id, gpu_q):
    """Move config, run training, move to done dir, then release GPU."""
    cf_name = os.path.basename(cfg_tmp_path)
    cf_training = os.path.join(training_dir, cf_name)

    shutil.copy(cfg_tmp_path, cf_training)
    os.remove(cfg_tmp_path)

    cmd = [PYTHON_BIN, script_path, "-c", cf_training, "-t", TASK_TAG, "-g", str(gpu_id)]
    print(f"[GPU {gpu_id}] RUN: {' '.join(cmd)}")
    subprocess.run(cmd)

    cf_end = os.path.join(end_dir, cf_name)
    shutil.copy(cf_training, cf_end)
    os.remove(cf_training)

    gpu_q.put(gpu_id)
    print(f"[GPU {gpu_id}] DONE: {cf_name}")


def task_with_multi_processes_qc(cfg_pool_dir, gpu_list):
    """Run configs in parallel with a simple GPU queue."""
    training_dir = os.path.join(cfg_pool_dir, TRAINING_DIR_NAME)
    end_dir = os.path.join(cfg_pool_dir, END_DIR_NAME)
    mkdir_all([training_dir, end_dir], verbose=VERBOSE_MKDIR)

    cfgs = sorted(
        e.path for e in os.scandir(cfg_pool_dir)
        if e.is_file() and e.name.endswith(CONFIG_EXT)
    )

    print(f"Found {len(cfgs)} configs:")
    for path in cfgs:
        print("  -", os.path.basename(path))

    gpu_q = multiprocessing.Queue()
    for gid in gpu_list:
        gpu_q.put(gid)
    print(f"Available GPUs: {gpu_list}")

    while cfgs:
        gid = None
        if not gpu_q.empty():
            gid = gpu_q.get()

        if gid is not None:
            cfg_path = cfgs.pop(0)
            p = multiprocessing.Process(
                target=run_script_in_process,
                args=(training_dir, end_dir, QC_TRAIN_PATH, cfg_path, gid, gpu_q),
            )
            p.start()
        else:
            time.sleep(SLEEP_SEC)

    for p in multiprocessing.active_children():
        p.join()

    print("All tasks finished.")


def main():
    task_with_multi_processes_qc(CFG_POOL_DIR, GPU_LIST)


if __name__ == "__main__":
    main()