"""
Parallel data generation runner.
Launches 12 generate_data.py processes simultaneously (6 normal + 6 wide bounds),
15k points each, seeds 101-112. Shows live tqdm progress per job + overall ETA.

Usage:
    python run_datagen.py
    python run_datagen.py --n 15000          # points per batch (default 15000)
    python run_datagen.py --dry-run          # print commands without running
"""

import argparse
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

from tqdm.auto import tqdm

JOBS = [
    # (seed, wide, out)
    (101, False, "data/batch01.csv"),
    (102, False, "data/batch02.csv"),
    (103, False, "data/batch03.csv"),
    (104, False, "data/batch04.csv"),
    (105, False, "data/batch05.csv"),
    (106, False, "data/batch06.csv"),
    (107, True,  "data/batch07.csv"),
    (108, True,  "data/batch08.csv"),
    (109, True,  "data/batch09.csv"),
    (110, True,  "data/batch10.csv"),
    (111, True,  "data/batch11.csv"),
    (112, True,  "data/batch12.csv"),
]

# Matches the tqdm.write() lines from generate_data.py:
#   [500/15000] valid=482 (96.4%) errors=0
PROGRESS_RE = re.compile(r"\[(\d+)/(\d+)\].*valid=(\d+)")


def _stream(proc, job_bar, n, log_path):
    """Read subprocess stdout line by line, update job_bar, write to log."""
    prev = 0
    with open(log_path, "w") as f:
        for line in proc.stdout:
            f.write(line)
            m = PROGRESS_RE.search(line)
            if m:
                current = int(m.group(1))
                valid   = int(m.group(3))
                delta   = current - prev
                if delta > 0:
                    job_bar.update(delta)
                    job_bar.set_postfix(valid=valid, refresh=False)
                    prev = current
    proc.wait()
    # fill any gap if last checkpoint < n
    remaining = n - prev
    if remaining > 0:
        job_bar.update(remaining)


def run_job(seed, wide, out, n, job_bar, overall_bar, overall_lock):
    cmd = [sys.executable, "-u", "generate_data.py",
           "--n", str(n), "--seed", str(seed), "--out", out]
    if wide:
        cmd.append("--wide")

    log_path = Path(out).with_suffix(".log")
    log_path.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    _stream(proc, job_bar, n, log_path)
    elapsed = time.time() - t0

    ok = proc.returncode == 0
    job_bar.set_description(
        job_bar.desc.split(":")[0] + (": DONE" if ok else ": FAIL")
    )
    job_bar.refresh()

    with overall_lock:
        overall_bar.update(1)

    return seed, ok, elapsed


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n",       type=int,  default=15000, help="Points per batch")
    p.add_argument("--dry-run", action="store_true",      help="Print commands only")
    args = p.parse_args()

    n_jobs    = len(JOBS)
    total_pts = n_jobs * args.n

    print(f"Launching {n_jobs} parallel jobs  |  {args.n:,} pts each  |  {total_pts:,} total")
    print(f"6 x core bounds (seeds 101-106)  +  6 x wide bounds (seeds 107-112)\n")

    if args.dry_run:
        for seed, wide, out in JOBS:
            flag = "--wide" if wide else "      "
            print(f"  python generate_data.py --n {args.n} --seed {seed} {flag} --out {out}")
        return

    # ── Build one tqdm bar per job (positions 0-11) + overall bar at bottom ──
    overall_lock = threading.Lock()
    job_bars = []
    for i, (seed, wide, out) in enumerate(JOBS):
        bounds = "wide" if wide else "core"
        bar = tqdm(
            total=args.n,
            desc=f"s{seed} {bounds}",
            position=i,
            leave=True,
            unit="pt",
            dynamic_ncols=True,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]{postfix}",
        )
        job_bars.append(bar)

    overall_bar = tqdm(
        total=n_jobs,
        desc="Overall",
        position=n_jobs,
        leave=True,
        unit="job",
        dynamic_ncols=True,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} jobs [{elapsed}<{remaining}]",
    )

    t_start = time.time()
    threads = []
    results = []

    def _run(seed, wide, out, job_bar, idx):
        res = run_job(seed, wide, out, args.n, job_bar, overall_bar, overall_lock)
        results.append(res)

    for i, (seed, wide, out) in enumerate(JOBS):
        t = threading.Thread(target=_run, args=(seed, wide, out, job_bars[i], i))
        t.start()
        threads.append(t)

    for t in threads:
        t.join()

    for bar in job_bars:
        bar.close()
    overall_bar.close()

    total_elapsed = time.time() - t_start
    failed = [seed for seed, ok, _ in results if not ok]

    print(f"\nFinished in {total_elapsed:.0f}s  ({total_elapsed/60:.1f} min)")

    if failed:
        print(f"FAILED seeds: {failed}")
        print("Check data/batchXX.log files for details.")
        sys.exit(1)
    else:
        valid_counts = []
        for _, _, out in JOBS:
            log = Path(out).with_suffix(".log")
            if log.exists():
                text = log.read_text()
                m = re.search(r"(\d+) valid / \d+ attempted", text)
                if m:
                    valid_counts.append(int(m.group(1)))
        total_valid = sum(valid_counts) if valid_counts else "?"
        print(f"All {n_jobs} batches done  |  ~{total_valid:,} valid points total")
        print(f"\nNext step:")
        print(f'  python merge_data.py --files "data/batch*.csv" --out data/merged_ccu.csv')


if __name__ == "__main__":
    main()
