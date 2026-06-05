import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ann" / "python"))

from algorithms import create_index


def main() -> None:
    index = create_index("brute_force")
    index.build([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [5.0, 5.0],
    ])

    query = [0.9, 0.1]

    for neighbor in index.query(query, 2):
        print(f"index={neighbor.index} distance={neighbor.distance}")


if __name__ == "__main__":
    main()
