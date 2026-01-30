#@markdown ## Save adversarial examples and log to CSV
import hashlib
import os
from datetime import time
import csv
from torch import nn
from utils.image_utilities import save_adv_examples

def save_run_results(combo_dir, summary, rows, tag, model_name, def_name, atk_name):
    """Utility di salvataggio per i risultati per-run CSV + Markdown + JPEG recovery."""
    os.makedirs(combo_dir, exist_ok=True)

    # --- CSV (normal) ---
    csv_path = os.path.join(combo_dir, "results.csv")
    fieldnames = ["batch_idx", "batch_size", "clean_correct", "adv_correct", "successful_on_correct"]
    save_csv(rows, fieldnames, csv_path)

    # --- NEW: JPEG CSV + Plot ---
    if "jpeg_recovery" in summary:
        jpeg_csv_path = os.path.join(combo_dir, "jpeg_recovery.csv")
        jpeg_plot_path = os.path.join(combo_dir, "jpeg_recovery.png")

        save_jpeg_recovery_csv(summary["jpeg_recovery"], jpeg_csv_path)
        plot_jpeg_recovery(summary["jpeg_recovery"], jpeg_plot_path)

    # --- Markdown Report ---
    md_path = os.path.join(combo_dir, "report.md")
    save_markdown_report(summary, rows, md_path)

    # Returned record
    record = {
        "tag": tag,
        "model": model_name,
        "defense": def_name,
        "attack": atk_name,
        "run_dir": combo_dir,
    }
    record.update(summary)
    return record


def plot_jpeg_recovery(jpeg_mean, out_path):
    """
    Plot JPEG recovery accuracy vs. compression quality.
    """
    import matplotlib.pyplot as plt

    qualities = list(jpeg_mean.keys())
    accs = [jpeg_mean[q] for q in qualities]

    plt.figure(figsize=(6, 4))
    plt.plot(qualities, accs, marker="o")
    plt.xlabel("JPEG Quality")
    plt.ylabel("Accuracy")
    plt.title("JPEG Recovery Accuracy vs Compression Quality")
    plt.grid(True)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"[INFO] JPEG recovery plot saved to {out_path}")


def save_jpeg_recovery_csv(jpeg_mean, out_path):
    """
    Save JPEG recovery accuracy (per quality) to CSV.
    """
    rows = [
        {"quality": q, "accuracy": jpeg_mean[q]}
        for q in jpeg_mean
    ]
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["quality", "accuracy"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"[INFO] JPEG recovery CSV saved to {out_path}")


SAVE_LIMIT = 5   # Save first 5 examples per model
def save_adv_samples(out_dir, model_cfg, attacked_images, inputs):
    saved = 0
    if saved < SAVE_LIMIT:
        model_adv_dir = os.path.join(out_dir, model_cfg.model_name, "adv_examples")
        os.makedirs(model_adv_dir, exist_ok=True)

        # loop through batch and save few images
        for i in range(min(inputs.size(0), SAVE_LIMIT - saved)):
            save_adv_examples(
                adv=attacked_images[i],
                orig=inputs[i],
                project_root=os.path.join(out_dir, model_cfg.model_name),
                prefix=f"example_{saved}",
                out_dirname="adv_examples",
                show_adv=False
            )
            saved += 1

def save_master_summary(out_dir, results_summary):
    """Save aggregated CSV + optional master markdown report."""
    if not results_summary:
        return None

    timestamp = time.strftime("%Y%m%d-%H%M%S")

    # CSV
    csv_path = os.path.join(out_dir, f"master_summary__{timestamp}.csv")
    keys = list(results_summary[0].keys())
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for r in results_summary:
            writer.writerow(r)

    # Markdown
    md_path = os.path.join(out_dir, f"master_report__{timestamp}.md")
    save_markdown_report(
        {"overall_runs": len(results_summary)},
        results_summary,
        md_path
    )

    return csv_path, md_path


def save_csv(rows, fieldnames, out_path):
    """Save raw results to CSV (batch-level)."""
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    print(f"[INFO] CSV saved to {out_path}")


def save_markdown_report(summary, rows, out_path, max_rows=20):
    """Save a Markdown report with explanations and per-batch table."""
    lines = []
    lines.append("# Evaluation Report\n")
    lines.append("## Summary\n")
    lines.append(f"- **Total samples:** {summary['total_samples']}")
    lines.append(f"- **Clean accuracy:** {100*summary['clean_accuracy']:.2f}%")
    if summary["adversarial_accuracy"] is not None:
        lines.append(f"- **Adversarial accuracy:** {100*summary['adversarial_accuracy']:.2f}%")
        lines.append(f"- **Attack success rate:** {100*summary['attack_success_rate_on_correct']:.2f}%")
    lines.append("\n---\n")
    lines.append("## Metric explanations\n")
    lines.append("- **Clean accuracy**: fraction of inputs correctly classified without attack.")
    lines.append("- **Adversarial accuracy**: fraction of inputs correctly classified under attack.")
    lines.append("- **Attack success rate**: fraction of originally correct predictions that were flipped by the attack.\n")

    lines.append("## Per-batch results\n")
    lines.append("| Batch | Size | Clean correct | Adv correct | Success on correct |")
    lines.append("|-------|------|---------------|-------------|--------------------|")
    for r in rows[:max_rows]:
        lines.append(f"| {r['batch_idx']} | {r['batch_size']} | {r['clean_correct']} | {r['adv_correct']} | {r['successful_on_correct']} |")

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        f.write("\n".join(lines))
    print(f"[INFO] Markdown report saved to {out_path}")


def save_master_report(results_summary, out_path):
    """
    Save a human-readable Markdown report summarizing all experiments in the grid.
    results_summary: list of dicts (from run_grid)
    out_path: path to .md file
    """
    if not results_summary:
        print("[WARN] No results to save in master report")
        return

    lines = []
    lines.append("# Master Evaluation Report\n")
    lines.append(f"Total runs: {len(results_summary)}\n")
    lines.append("| Model | Defense | Attack | Clean Acc | Adv Acc | Success Rate | Run Dir |")
    lines.append("|-------|---------|--------|-----------|---------|--------------|---------|")

    for r in results_summary:
        clean_acc = f"{100*r['clean_accuracy']:.2f}%"
        adv_acc = f"{100*r['adversarial_accuracy']:.2f}%" if r['adversarial_accuracy'] is not None else "N/A"
        success_rate = f"{100*r['attack_success_rate_on_correct']:.2f}%" if r['attack_success_rate_on_correct'] is not None else "N/A"

        lines.append(
            f"| {r['model']} | {r['defense']} | {r['attack']} | "
            f"{clean_acc} | {adv_acc} | {success_rate} | {r['run_dir']} |"
        )

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        f.write("\n".join(lines))
    print(f"[INFO] Master Markdown report saved to {out_path}")


def save_and_report_adv_examples(adv, orig, project_root, prefix="adv", show_adv=False):
    """
    Wrapper to save adversarial examples and print paths.
    adv: tensor [B,C,H,W] (normalized or unnormalized depending on save_adv_examples impl)
    orig: original tensor [B,C,H,W]
    """
    paths = save_adv_examples(
        adv=adv,
        orig=orig,
        project_root=project_root,
        prefix=prefix,
        show_adv=show_adv,
    )
    print("Saved images:")
    print(" original:", paths["orig"])
    print(" adversarial:", paths["adv"])
    print(" diff (amplified):", paths["diff"])


def state_dict_md5(model: nn.Module):
    # compute a stable checksum across parameters (move to cpu to be safe)
    m = hashlib.md5()
    for k, v in sorted(model.state_dict().items()):
        m.update(k.encode())
        m.update(v.detach().cpu().numpy().tobytes())
    return m.hexdigest()


def print_model_summary(name: str, model: nn.Module):
    print(f"--- MODEL SUMMARY: {name} ---")
    print("Type:", type(model))
    print("Device(s):", {p.device for p in model.parameters()})
    print("Params count:", sum(p.numel() for p in model.parameters()))
    try:
        print("State-dict md5:", state_dict_md5(model))
    except Exception as e:
        print("Checksum failed:", e)
    # show top-level modules to confirm structure
    print("Children modules:", [c for c in model.named_children()])
    print("-----------------------------")


def postprocess_results(out_dir, models, attacks, defenses):
    """Optional post-run analysis (e.g., ε-sweep plots)."""
    try:
        from utils.image_utilities import plot_eps_sweep

        for model_cfg in models:
            model_name = model_cfg.model_name
            for atk_cfg in attacks:
                if not hasattr(atk_cfg, "eps_list") or atk_cfg.eps_list is None:
                    continue
                for def_cfg in defenses:
                    plot_path = os.path.join(
                        out_dir,
                        f"eps_sweep_{model_name}_{atk_cfg.name}_{def_cfg.name}.png"
                    )
                    print(f"[INFO] Plotting ε-sweep for {atk_cfg.name} ({def_cfg.name})…")
                    plot_eps_sweep(
                        results_root=out_dir,
                        attack_name=atk_cfg.name,
                        defense_name=def_cfg.name,
                        model_name=model_name,
                        save_path=plot_path
                    )
    except Exception as e:
        print(f"[WARN] Postprocess skipped: {e}")