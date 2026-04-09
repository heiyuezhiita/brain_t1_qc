import multiprocessing
import subprocess
import time
import os
import shutil


# =========================
# Configuration
# =========================
SCRIPT_PATH = "transfer_predict.py"
CFG_POOL_DIR = "configs/transfer/batch_predict/cohort_b"

GPU_LIST = [0, 0]
PYTHON_BIN = "python"
CONFIG_EXT = ".yaml"

RUNNING_DIR_NAME = "training_cfg"
DONE_DIR_NAME = "end_cfg"

SLEEP_SEC = 1
TASK_TAG = "multi_process_task_script"
VERBOSE_MKDIR = True
# =========================


def mkdir_all(paths, verbose=True):
    """Create one or more directories if needed."""
    if isinstance(paths, str):
        paths = [paths]
    elif not isinstance(paths, (list, tuple)):
        raise TypeError(f"Expected str, list, or tuple, got {type(paths)}")

    for path in paths:
        if not os.path.exists(path):
            os.makedirs(path)
            if verbose:
                print(f"mkdir: {path}")


def run_job(running_dir, done_dir, script_path, cfg_path, gpu_id, gpu_queue):
    """
    Run one job in a separate process.

    Steps:
    1. Move the config to the running directory.
    2. Run the script.
    3. Move the config to the done directory.
    4. Release the GPU back to the queue.
    """
    cfg_name = os.path.basename(cfg_path)
    running_cfg = os.path.join(running_dir, cfg_name)

    shutil.copy(cfg_path, running_cfg)
    os.remove(cfg_path)

    cmd = [PYTHON_BIN, script_path, "-c", running_cfg, "-t", TASK_TAG, "-g", str(gpu_id)]
    print(f"[GPU {gpu_id}] RUN: {' '.join(cmd)}")
    subprocess.run(cmd)

    done_cfg = os.path.join(done_dir, cfg_name)
    shutil.copy(running_cfg, done_cfg)
    os.remove(running_cfg)

    gpu_queue.put(gpu_id)
    print(f"[GPU {gpu_id}] DONE: {cfg_name}")


def run_jobs(cfg_pool_dir, gpu_list):
    """Run multiple jobs with simple GPU scheduling."""
    running_dir = os.path.join(cfg_pool_dir, RUNNING_DIR_NAME)
    done_dir = os.path.join(cfg_pool_dir, DONE_DIR_NAME)
    mkdir_all([running_dir, done_dir], verbose=VERBOSE_MKDIR)

    cfgs = sorted(
        entry.path for entry in os.scandir(cfg_pool_dir)
        if entry.is_file() and entry.name.endswith(CONFIG_EXT)
    )

    print(f"Found {len(cfgs)} config files:")
    for path in cfgs:
        print("  -", os.path.basename(path))

    gpu_queue = multiprocessing.Queue()
    for gpu_id in gpu_list:
        gpu_queue.put(gpu_id)

    print(f"Available GPUs: {gpu_list}")

    while cfgs:
        gpu_id = None
        if not gpu_queue.empty():
            gpu_id = gpu_queue.get()

        if gpu_id is not None:
            cfg_path = cfgs.pop(0)
            process = multiprocessing.Process(
                target=run_job,
                args=(running_dir, done_dir, SCRIPT_PATH, cfg_path, gpu_id, gpu_queue),
            )
            process.start()
        else:
            time.sleep(SLEEP_SEC)

    for process in multiprocessing.active_children():
        process.join()

    print("All jobs finished.")


def main():
    run_jobs(CFG_POOL_DIR, GPU_LIST)


if __name__ == "__main__":
    main()