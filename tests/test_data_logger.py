import pandas as pd
from pathlib import Path
from src.simulation.data_logger import TrajectoryLogger

def print_trajectory_data(logger, path=None):
    if path is None:
        path = Path(logger.output_dir)
    for file in path.glob("*.parquet"):
        df = pd.read_parquet(file)
        print(f"Data from {file.name}:")
        print(df)

if __name__ == "__main__":
    path = Path(__file__).resolve().parent.parent / "src" / "simulation" / "trajectory_data"
    logger = TrajectoryLogger(output_dir=path)
    print_trajectory_data(logger, path=path)