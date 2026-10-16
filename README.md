# K-Means

An interactive command-line implementation of the K-Means clustering algorithm
(Lloyd's algorithm) in pure Java — no dependencies, no build system.

It reads a numeric dataset from a CSV file, partitions it into *k* clusters, and
writes the grouped result to `new.csv`.

## Requirements

- JDK 8 or later

## Input format

The input is a plain CSV file with **no header row**:

- **Column 1** — an identifier for the row (a row number or any unique string).
  It is carried through to the output but never used in calculations.
- **Columns 2…n** — numeric features. Every row must have the same number of
  columns, and there must be at least one feature column.

Blank lines and a UTF-8 BOM are ignored. Non-finite values (`NaN`, `Infinity`)
and ragged rows are rejected with a line-numbered error message.

Example `data.csv`:

```csv
alice,1.0,1.2
bob,1.5,0.9
carol,9.1,8.8
dave,8.7,9.2
```

## Build and run

Compile once:

```sh
javac -d out src/KMeans.java src/DataPoint.java
```

Run from the directory containing your CSV file:

```sh
java -cp /path/to/out KMeans
```

The program then prompts for:

1. **CSV file name** — without the `.csv` extension. Path traversal
   (`..`, `/`, `\`) is rejected for safety.
2. **Number of clusters** — between 2 and the number of data rows.
3. **Maximum iterations** — an upper bound; the algorithm may stop earlier.
4. **Initial centroid rows** — one distinct row number per cluster. Picking
   rows from visibly different regions of the data gives better results;
   identical or poorly spread seeds can converge to a weaker local optimum
   (an inherent property of K-Means, not a bug).

## Output

Console:

- Total execution time (seconds)
- Number of iterations actually run
- The final centroid of each cluster

`new.csv` (written to the current directory, overwriting any existing file):

- One row per data point: `cluster,id,feature1,feature2,...`
- Each cluster's rows are grouped together, separated by blank lines

## How it works

- **Algorithm** — classic Lloyd's algorithm: assign each point to its nearest
  centroid, recompute each centroid as the mean of its members, repeat.
- **Flat data layout** — all features are stored in a single row-major
  `double[]`, so the hot loop reads memory sequentially instead of chasing
  per-point object references.
- **Squared distances** — distances are compared in squared-Euclidean form,
  which preserves ordering while avoiding a square root per evaluation.
- **Single-pass iterations** — cluster sums, counts, and assignments are all
  accumulated in one sweep; no cluster lists are rebuilt per iteration.
- **Empty-cluster reseeding** — if a cluster loses all its points, it is
  re-seeded with the point farthest from its current centroid rather than
  being dropped or crashed on.
- **Convergence** — stops early when every centroid coordinate moves less than
  a relative tolerance of `1e-6`, or when the iteration cap is hit.

## Files

| File               | Purpose                                                        |
|--------------------|----------------------------------------------------------------|
| `src/KMeans.java`  | CLI prompts, CSV I/O, validation, and the clustering `Engine`  |
| `src/DataPoint.java` | Immutable value type: row identifier + feature vector        |

## License

Apache License 2.0 — see `LICENSE`.
