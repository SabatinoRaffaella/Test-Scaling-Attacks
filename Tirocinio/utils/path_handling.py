from pathlib import Path

def prepare_checkpoint_dir(out_dir=None, model_name="model"):
    """
    Resolves and ensures a valid checkpoint directory, returning both
    the directory path and full checkpoint file path.

    Args:
        out_dir (str | None): Directory where checkpoints should go.
                              If None → defaults to <project_root>/attack_results/checkpoints
        model_name (str): Model name for checkpoint filename.

    Returns:
        (Path, Path): Tuple (directory_path, checkpoint_path)
    """
    project_root = Path(__file__).resolve().parents[1]  # project root (e.g. /content/Tirocinio)
    if out_dir is None:
        out_dir = project_root / "attack_results" / "checkpoints"
    else:
        out_dir = Path(out_dir).expanduser().resolve()

    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / f"{model_name}.pth"

    print(f"[INFO] Checkpoint directory ready: {out_dir}")
    return out_dir, ckpt_path