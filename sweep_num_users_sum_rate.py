import argparse
import csv
import re
import subprocess
import sys
from pathlib import Path

import scipy.io as spio

from config import K, M


def parse_args():
    project_dir = Path(__file__).resolve().parent
    repo_root = project_dir.parent
    default_models_dir = repo_root / "result" / "save_model" / "UMi_UPA"
    default_output_dir = project_dir / "result" / "save_fig"

    parser = argparse.ArgumentParser(
        description="Run selfnomination_project/test_unified.py across multiple num_users values and aggregate sum-rate results."
    )
    parser.add_argument(
        "--models_dir",
        type=Path,
        default=default_models_dir,
        help="Directory containing trained .pth checkpoints.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=default_output_dir,
        help="Directory to save raw sweep outputs, summaries, and plots.",
    )
    parser.add_argument(
        "--channel_mode",
        type=str,
        default="UMi_UPA",
        choices=["RF", "UMi_UPA", "UMi_ULA", "Berlin_UPA", "RMa_UPA"],
        help="Channel mode used for evaluation.",
    )
    parser.add_argument(
        "--user_numbers",
        type=int,
        nargs="+",
        default=[20, 30, 40, 50, 60, 70],
        help="List of num_users values to sweep.",
    )
    parser.add_argument(
        "--snr_db",
        type=float,
        default=15.0,
        help="Fixed SNR used for all evaluations.",
    )
    parser.add_argument(
        "--num_test_samples",
        type=int,
        default=2000,
        help="Number of test samples passed to selfnomination_project/test_unified.py.",
    )
    parser.add_argument(
        "--gpu_id",
        type=str,
        default=None,
        help="CUDA device id to use during evaluation.",
    )
    parser.add_argument(
        "--without_baselines",
        action="store_true",
        help="Skip full-feedback and random-feedback baseline curves.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run evaluations even if raw result files already exist.",
    )
    return parser.parse_args()


def parse_model_metadata(model_path: Path):
    name = model_path.stem

    method_match = re.search(r"_(REINFORCE|DirectGrad)_", name)
    input_match = re.search(r"_(inH|inChg)_", name)
    scheduling_match = re.search(r"_(Random|Greedy)_", name)
    beamforming_match = re.search(r"_(ZF|RZF)_", name)
    user_match = re.search(r"_UE(\d+)_", name)

    if not all([method_match, input_match, scheduling_match, beamforming_match, user_match]):
        raise ValueError(f"Could not parse model metadata from filename: {model_path.name}")

    method = "reinforce" if method_match.group(1) == "REINFORCE" else "directgrad"
    input_type = "full" if input_match.group(1) == "inH" else "chg_input"
    scheduling = scheduling_match.group(1).lower()
    beamforming = beamforming_match.group(1).lower()
    user_count = int(user_match.group(1))

    return {
        "method": method,
        "input_type": input_type,
        "scheduling": scheduling,
        "beamforming": beamforming,
        "user_count": user_count,
        "label": name,
    }


def find_best_model(models_dir: Path, channel_mode: str, user_count: int, scheduling: str):
    pattern = f"*UE{user_count}_M{M}_K{K}_*_{channel_mode}_best.pth"
    best_match = None

    for path in sorted(models_dir.glob(pattern)):
        if "_ZS_" in path.name:
            continue
        meta = parse_model_metadata(path)
        if meta["method"] != "reinforce":
            continue
        if meta["input_type"] != "full":
            continue
        if meta["beamforming"] != "zf":
            continue
        if meta["scheduling"] != scheduling:
            continue

        epoch_match = re.search(r"Ne(\d+)", path.stem)
        epoch = int(epoch_match.group(1)) if epoch_match else -1
        if best_match is None or epoch > best_match[0]:
            best_match = (epoch, path)

    if best_match is None:
        raise FileNotFoundError(
            f"No matching checkpoint found for UE{user_count}, M{M}, K{K}, "
            f"{channel_mode}, reinforce/full/{scheduling}/zf in {models_dir}."
        )

    return best_match[1]


def scalar_from_mat(value):
    while hasattr(value, "shape") and value.shape == (1, 1):
        value = value[0, 0]
    if hasattr(value, "item"):
        return float(value.item())
    return float(value)


def maybe_find_precomputed_result(
    repo_root: Path,
    job: dict,
    channel_mode: str,
    user_count: int,
    num_test_samples: int,
    snr_db: float,
):
    if num_test_samples != 2000 or float(snr_db) != 15.0:
        return None

    root_test_dir = repo_root / "result" / "save_testresult"

    if job["method"] == "reinforce":
        sched_tag = job["scheduling"].upper()
        candidates = [
            root_test_dir / f"test_Nt32_UE{user_count}_M{M}_K{K}_reinforce_full_{sched_tag}_ZF_{channel_mode}_samples2000.mat",
            root_test_dir / "previous" / f"test_Nt32_UE{user_count}_M{M}_K{K}_reinforce_full_{sched_tag}_ZF_{channel_mode}_samples2000.mat",
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return None

    if job["method"] == "baseline" and job["nomination_mode"] == "all_fb":
        baseline_name = {
            "greedy": "baseline_greedy_zf",
            "random": "baseline_random_zf",
            "sus": "baseline_sus_zf",
        }.get(job["scheduling"])
        if baseline_name is None:
            return None
        candidates = [
            root_test_dir / f"{baseline_name}_{channel_mode}_UE20-70_Nt32_M30_K20_results.mat",
            root_test_dir / "previous" / f"{baseline_name}_{channel_mode}_UE20-70_Nt32_M30_K20_results.mat",
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return None

    return None


def load_precomputed_row(precomputed_path: Path, job: dict, channel_mode: str, user_count: int, snr_db: float):
    result = spio.loadmat(precomputed_path)

    if job["method"] == "reinforce":
        return {
            "model_label": job["label"],
            "method": job["method"],
            "input_type": job["input_type"],
            "scheduling": job["scheduling"],
            "beamforming": job["beamforming"],
            "nomination_mode": job["nomination_mode"],
            "random_fb_prob": job["random_fb_prob"],
            "channel_mode": channel_mode,
            "snr_db": float(snr_db),
            "num_users": int(user_count),
            "sum_rate_mean": scalar_from_mat(result["sum_rate_mean"]),
            "sum_rate_std": scalar_from_mat(result["sum_rate_std"]),
            "nominated_count_mean": scalar_from_mat(result["nominated_count_mean"]),
            "scheduled_count_mean": scalar_from_mat(result["scheduled_count_mean"]),
            "result_file": str(precomputed_path),
        }

    user_numbers = result["user_numbers"].flatten().astype(int)
    match_indices = [idx for idx, value in enumerate(user_numbers) if int(value) == int(user_count)]
    if not match_indices:
        raise ValueError(f"UE{user_count} not found in {precomputed_path}")
    idx = match_indices[0]

    return {
        "model_label": job["label"],
        "method": job["method"],
        "input_type": job["input_type"],
        "scheduling": job["scheduling"],
        "beamforming": job["beamforming"],
        "nomination_mode": job["nomination_mode"],
        "random_fb_prob": job["random_fb_prob"],
        "channel_mode": channel_mode,
        "snr_db": float(snr_db),
        "num_users": int(user_count),
        "sum_rate_mean": float(result["sum_rates_mean"].flatten()[idx]),
        "sum_rate_std": float(result["sum_rates_std"].flatten()[idx]),
        "nominated_count_mean": float(result["nominated_users_mean"].flatten()[idx]),
        "scheduled_count_mean": float(result["scheduled_users_mean"].flatten()[idx]),
        "result_file": str(precomputed_path),
    }


def run_single_evaluation(
    repo_root: Path,
    model_path: Path | None,
    job: dict,
    channel_mode: str,
    user_count: int,
    snr_db: float,
    num_test_samples: int,
    output_file: Path,
    gpu_id: str | None,
):
    cmd = [
        sys.executable,
        "selfnomination_project/run_test_unified_with_overrides.py",
        "--num_users_override",
        str(user_count),
        "--method",
        job["method"],
        "--scheduling",
        job["scheduling"],
        "--beamforming",
        job["beamforming"],
        "--channel_mode",
        channel_mode,
        "--num_test_samples",
        str(num_test_samples),
        "--snr_db",
        str(snr_db),
        "--output_file",
        str(output_file),
    ]
    if job["method"] != "baseline":
        cmd.extend(["--input_type", job["input_type"]])
        cmd.extend(["--model_path", str(model_path)])
    else:
        cmd.extend(["--nomination_mode", job["nomination_mode"]])
        if job["nomination_mode"] == "random_fb":
            cmd.extend(["--random_fb_prob", str(job["random_fb_prob"])])
    if gpu_id is not None:
        cmd.extend(["--gpu_id", gpu_id])

    subprocess.run(cmd, cwd=repo_root, check=True)


def build_jobs(include_baselines: bool):
    jobs = [
        {
            "model_path": "dynamic",
            "method": "reinforce",
            "input_type": "full",
            "scheduling": "greedy",
            "beamforming": "zf",
            "label": "Proposed PG SN+OS+ZF",
            "output_stem": "proposed_pg_sn_os_zf",
            "nomination_mode": "model",
            "random_fb_prob": None,
        },
        {
            "model_path": "dynamic",
            "method": "reinforce",
            "input_type": "full",
            "scheduling": "random",
            "beamforming": "zf",
            "label": "Proposed PG SN+RS+ZF",
            "output_stem": "proposed_pg_sn_rs_zf",
            "nomination_mode": "model",
            "random_fb_prob": None,
        },
    ]

    if include_baselines:
        jobs.extend(
            [
                {
                    "model_path": None,
                    "method": "baseline",
                    "input_type": "full",
                    "scheduling": "sus",
                    "beamforming": "zf",
                    "label": "All FB+SUS+ZF",
                    "output_stem": "all_fb_sus_zf",
                    "nomination_mode": "all_fb",
                    "random_fb_prob": None,
                },
                {
                    "model_path": None,
                    "method": "baseline",
                    "input_type": "full",
                    "scheduling": "greedy",
                    "beamforming": "zf",
                    "label": "All FB+OS+ZF",
                    "output_stem": "all_fb_os_zf",
                    "nomination_mode": "all_fb",
                    "random_fb_prob": None,
                },
                {
                    "model_path": None,
                    "method": "baseline",
                    "input_type": "full",
                    "scheduling": "random",
                    "beamforming": "zf",
                    "label": "All FB+RS+ZF",
                    "output_stem": "all_fb_rs_zf",
                    "nomination_mode": "all_fb",
                    "random_fb_prob": None,
                },
                {
                    "model_path": None,
                    "method": "baseline",
                    "input_type": "full",
                    "scheduling": "greedy",
                    "beamforming": "zf",
                    "label": "Random FB+OS+ZF",
                    "output_stem": "random_fb_os_zf",
                    "nomination_mode": "random_fb",
                    "random_fb_prob": 0.5,
                },
                {
                    "model_path": None,
                    "method": "baseline",
                    "input_type": "full",
                    "scheduling": "random",
                    "beamforming": "zf",
                    "label": "Random FB+RS+ZF",
                    "output_stem": "random_fb_rs_zf",
                    "nomination_mode": "random_fb",
                    "random_fb_prob": 0.5,
                },
            ]
        )

    return jobs


def save_summary_csv(rows, output_path: Path):
    fieldnames = [
        "model_label",
        "method",
        "input_type",
        "scheduling",
        "beamforming",
        "nomination_mode",
        "random_fb_prob",
        "channel_mode",
        "snr_db",
        "num_users",
        "sum_rate_mean",
        "sum_rate_std",
        "nominated_count_mean",
        "scheduled_count_mean",
        "result_file",
    ]
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_summary_mat(rows, output_path: Path):
    grouped = {}
    for row in rows:
        label = row["model_label"]
        grouped.setdefault(label, {"num_users": [], "sum_rate_mean": [], "sum_rate_std": []})
        grouped[label]["num_users"].append(row["num_users"])
        grouped[label]["sum_rate_mean"].append(row["sum_rate_mean"])
        grouped[label]["sum_rate_std"].append(row["sum_rate_std"])

    mat_payload = {}
    for label, values in grouped.items():
        key = re.sub(r"[^0-9A-Za-z_]", "_", label)
        mat_payload[f"{key}_num_users"] = values["num_users"]
        mat_payload[f"{key}_sum_rate_mean"] = values["sum_rate_mean"]
        mat_payload[f"{key}_sum_rate_std"] = values["sum_rate_std"]
    spio.savemat(output_path, mat_payload)


def maybe_save_plot(rows, output_path: Path):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"Skipping plot generation: {exc}")
        return

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 12,
            "axes.linewidth": 1.2,
            "axes.labelsize": 16,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 12,
        }
    )

    style_by_key = {
        "Proposed PG SN+OS+ZF": {"color": "#0047ff", "marker": "o", "linestyle": "-"},
        "Proposed PG SN+RS+ZF": {"color": "#ff0000", "marker": "s", "linestyle": "-"},
        "All FB+SUS+ZF": {"color": "#0a9510", "marker": "d", "linestyle": "--"},
        "All FB+OS+ZF": {"color": "#0a9510", "marker": "*", "linestyle": "--"},
        "All FB+RS+ZF": {"color": "#0a9510", "marker": "^", "linestyle": "--"},
        "Random FB+OS+ZF": {"color": "#7d2b8a", "marker": r"$*$", "linestyle": (0, (1, 1))},
        "Random FB+RS+ZF": {"color": "#7d2b8a", "marker": "v", "linestyle": (0, (1, 1))},
    }
    order = [
        "Proposed PG SN+OS+ZF",
        "Proposed PG SN+RS+ZF",
        "All FB+SUS+ZF",
        "All FB+OS+ZF",
        "All FB+RS+ZF",
        "Random FB+OS+ZF",
        "Random FB+RS+ZF",
    ]

    grouped = {}
    for row in rows:
        grouped.setdefault(row["model_label"], []).append(row)

    fig, ax = plt.subplots(figsize=(7.0, 5.25))
    for label in order:
        if label not in grouped:
            continue
        group_rows = sorted(grouped[label], key=lambda row: row["num_users"])
        x_vals = [row["num_users"] for row in group_rows]
        y_vals = [row["sum_rate_mean"] for row in group_rows]
        style = style_by_key[label]

        ax.plot(
            x_vals,
            y_vals,
            color=style["color"],
            marker=style["marker"],
            linestyle=style["linestyle"],
            linewidth=2.2,
            markersize=8.5,
            markerfacecolor="white" if label != "All FB+OS+ZF" else style["color"],
            markeredgecolor=style["color"],
            markeredgewidth=1.6,
            label=label,
            zorder=3,
        )

    x_values = [row["num_users"] for row in rows]
    ax.set_xlabel("Number of UEs")
    ax.set_ylabel("Sum-rate [bps/Hz]")
    ax.set_xlim(min(x_values) - 2, max(x_values) + 2)
    ax.set_ylim(bottom=0)
    ax.grid(True, which="major", linestyle="-", linewidth=0.6, color="#bdbdbd", alpha=0.65)
    ax.tick_params(direction="in", length=6, width=1.1)

    for spine in ax.spines.values():
        spine.set_color("black")
        spine.set_linewidth(1.2)

    legend = ax.legend(
        loc="upper left",
        frameon=True,
        fancybox=False,
        framealpha=1.0,
        edgecolor="black",
    )
    legend.get_frame().set_linewidth(1.0)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    project_dir = Path(__file__).resolve().parent
    repo_root = project_dir.parent
    output_dir = args.output_dir.resolve()
    raw_dir = output_dir / "num_users_sweep_raw"
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    jobs = build_jobs(include_baselines=not args.without_baselines)
    models_dir = args.models_dir.resolve()

    rows = []
    snr_tag = int(args.snr_db) if float(args.snr_db).is_integer() else str(args.snr_db).replace(".", "p")
    for user_count in args.user_numbers:
        for job in jobs:
            model_path = None
            precomputed_path = None
            if not args.force:
                precomputed_path = maybe_find_precomputed_result(
                    repo_root=repo_root,
                    job=job,
                    channel_mode=args.channel_mode,
                    user_count=user_count,
                    num_test_samples=args.num_test_samples,
                    snr_db=args.snr_db,
                )
            if precomputed_path is not None:
                print(f"Reusing root result for {job['label']} at UE={user_count}")
                rows.append(
                    load_precomputed_row(
                        precomputed_path=precomputed_path,
                        job=job,
                        channel_mode=args.channel_mode,
                        user_count=user_count,
                        snr_db=args.snr_db,
                    )
                )
                continue

            if job["method"] != "baseline":
                model_path = find_best_model(
                    models_dir=models_dir,
                    channel_mode=args.channel_mode,
                    user_count=user_count,
                    scheduling=job["scheduling"],
                )

            output_file = raw_dir / f"{job['output_stem']}_UE{user_count}_SNR{snr_tag}_results.mat"
            if output_file.exists() and not args.force:
                print(f"Reusing {job['label']} at UE={user_count}")
            else:
                print(f"Running {job['label']} at UE={user_count}")
                run_single_evaluation(
                    repo_root=repo_root,
                    model_path=model_path,
                    job=job,
                    channel_mode=args.channel_mode,
                    user_count=user_count,
                    snr_db=args.snr_db,
                    num_test_samples=args.num_test_samples,
                    output_file=output_file,
                    gpu_id=args.gpu_id,
                )

            result = spio.loadmat(output_file)
            rows.append(
                {
                    "model_label": job["label"],
                    "method": job["method"],
                    "input_type": job["input_type"],
                    "scheduling": job["scheduling"],
                    "beamforming": job["beamforming"],
                    "nomination_mode": job["nomination_mode"],
                    "random_fb_prob": job["random_fb_prob"],
                    "channel_mode": args.channel_mode,
                    "snr_db": float(args.snr_db),
                    "num_users": int(user_count),
                    "sum_rate_mean": scalar_from_mat(result["sum_rate_mean"]),
                    "sum_rate_std": scalar_from_mat(result["sum_rate_std"]),
                    "nominated_count_mean": scalar_from_mat(result["nominated_count_mean"]),
                    "scheduled_count_mean": scalar_from_mat(result["scheduled_count_mean"]),
                    "result_file": str(output_file),
                }
            )

    summary_prefix = f"{args.channel_mode}_num_users_sweep_M{M}_K{K}_SNR{snr_tag}"
    csv_path = output_dir / f"{summary_prefix}.csv"
    mat_path = output_dir / f"{summary_prefix}.mat"
    png_path = output_dir / f"{summary_prefix}.png"

    save_summary_csv(rows, csv_path)
    save_summary_mat(rows, mat_path)
    maybe_save_plot(rows, png_path)

    print(f"Saved CSV summary to: {csv_path}")
    print(f"Saved MAT summary to: {mat_path}")
    if png_path.exists():
        print(f"Saved plot to: {png_path}")


if __name__ == "__main__":
    main()
