# ANN cross-language verifier (Python + C++)

k-nearest-neighbors implementations in Python and C++ sharing a common interface and test script. Use `harness/verify.py` to diff stdout and confirm both sides return the same top-k indices.

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

Implementations are created by name via `create_index(...)`. Currently available:

- **`brute_force`** — exact L2 scan (Python and C++)
- **`multiprobe`** — multi-probe grid ANN (Python and C++, ported from MultiProbeANN)

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
./build/demo
PYTHONPATH=ann/python python3 examples/main.py
```

### Verify equivalence

`harness/verify.py` rebuilds C++ by default when any side uses `cpp`. Use `--no-build` to skip.

```bash
.venv/bin/python3 harness/verify.py                                          # python:bf vs cpp:bf
.venv/bin/python3 harness/verify.py --python-algo brute_force                # python:bf vs python:bf
.venv/bin/python3 harness/verify.py --cpp-algo brute_force                   # cpp:bf vs cpp:bf
.venv/bin/python3 harness/verify.py --left python:brute_force --right cpp:brute_force
.venv/bin/python3 harness/verify.py --no-build                               # skip cmake rebuild
```

Expected output:

```
PASS: python:brute_force and cpp:brute_force agree on all 15 query result(s).
```

## Project layout

```
ann/
  python/               ann_interface.py, algorithms.py
  cpp/                  ann_interface.hpp, algorithms.hpp
harness/
  script.cases          shared test script (single source of truth)
  python/               script_loader.py, run_script.py
  cpp/                  script_loader.hpp, run_script.cpp
  verify.py             runs both runners and diffs output
examples/
  main.py / main.cpp    small demos
```

## Test script format (`harness/script.cases`)

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
harness/script.cases
     ├── harness/python/run_script.py  ──► stdout
     └── build/run_script (C++)        ──► stdout
                    │
            harness/verify.py (diff)
```

1. Python and C++ each parse `script.cases` with their own loader.
2. Each executes the same `build` / `query` sequence.
3. Each prints the same canonical text (indices only).
4. `verify.py` diffs the two outputs.

No test data is duplicated in source code.

## Adding tests

Edit `harness/script.cases` only. Append new `build` / `query` blocks, then run:

```bash
.venv/bin/python3 harness/verify.py
```

## Run test runners individually

```bash
PYTHONPATH=ann/python python3 harness/python/run_script.py --algo brute_force
./build/run_script harness/script.cases --algo brute_force
```
