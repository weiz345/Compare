from nearest_neighbors import BruteForceNN


def main() -> None:
    index = BruteForceNN()
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
