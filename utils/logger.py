from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
import time

def make_writer(project: str, run_name: str | None = None):
    if run_name is None:
        run_name = time.strftime("%Y%m%d-%H%M%S")
    log_dir = Path("runs") / project / run_name
    # Use:
    # project = logs for logs
    return SummaryWriter(str(log_dir)), str(log_dir)

def log_scalars(writer: SummaryWriter, step: int, prefix: str, metrics: dict):
    for k, v in metrics.items():
        writer.add_scalar(f"{prefix}/{k}", float(v), step)