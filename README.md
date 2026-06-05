# Brute-force nearest neighbors (Python + C++)

Minimal **exact** k-nearest-neighbors implementations in Python and C++, with a shared test script to verify both return the same top-k indices.

## API

Both languages define a common abstract interface with two methods:

```python
class NearestNeighbors(ABC):
    def build(self, points): ...
    def query(self, point, k) -> list[Neighbor]: ...
```

```cpp
class NearestNeighbors {
  virtual void build(const std::vector<std::vector<float>>& points) = 0;
  virtual std::vector<Neighbor> query(const std::vector<float>& point, int k) const = 0;
};
```

Implementations are created by name via `create_index("brute_force")`. Currently available: **`brute_force`**.

Verification compares **indices only** (not distances).

## Quick start

### Prerequisites

- Python 3.10+
- C++17 compiler and CMake 3.16+

### Build C++

```bash
cmake -B build
cmake --build build
```

### Run demos

```bash
./build/demo      # C++
python3 main.py   # Python
```

### Verify equivalence

```bash
python3 verify.py                      # default: --algo brute_force
python3 verify.py --algo brute_force
```

Expected output:

```
PASS: Python and C++ agree on all 15 query result(s) using algo='brute_force'.
```

## Project layout

```
nearest_neighbors.hpp   C++ implementation (header-only)
nearest_neighbors.py    Python implementation
script.cases            shared test script (single source of truth)
script_loader.hpp       C++ parser for script.cases
script_loader.py        Python parser for script.cases
run_script.cpp          C++ test runner
run_script.py           Python test runner
verify.py               runs both runners and diffs output
main.cpp / main.py      small demos
```

## Test script format (`script.cases`)

One file, read directly by both languages. Operations run top-to-bottom; each `build` replaces the index.

```
build
0 0
1 0
0 1
query 0.9 0.1 k 2
```

- **`build`** — following lines are points (zero lines = empty index)
- **`query <coords...> k <int>`** — query point, then `k`
- Lines starting with `#` are comments

### Example

```
# basic 2-NN
build
0 0
1 0
0 1
5 5
query 0.9 0.1 k 2
```

Both runners print:

```
build ok
query k 2
1
0
```

## How verification works

```
script.cases
     ├── run_script.py  ──► stdout
     └── run_script.cpp ──► stdout
                │
            verify.py (diff)
```

1. Python and C++ each parse `script.cases` with their own loader.
2. Each executes the same `build` / `query` sequence.
3. Each prints the same canonical text (indices only).
4. `verify.py` diffs the two outputs.

No test data is duplicated in source code.

## Adding tests

Edit `script.cases` only. Append new `build` / `query` blocks, then run:

```bash
python3 verify.py
```

See `test_plan.txt` for the intended cases and expected indices.

## Run test runners individually

```bash
python3 run_script.py script.cases --algo brute_force
./build/run_script script.cases --algo brute_force
```
