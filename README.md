# K-Means

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

An interactive command-line implementation of the K-Means clustering algorithm
(Lloyd's algorithm) in pure Java — no dependencies, no build system.

It reads a numeric dataset from a CSV file, partitions it into *k* clusters, and
writes the grouped result to `new.csv`.

## Requirements

- JDK 8 or later

## Input format

The input is a plain CSV file with **no header row**:

- **Column 1** — an identifier for the row (a row number or any unique string).
  It must be unique, is carried through to the output, and is never used in
  calculations.
- **Columns 2…n** — numeric features. Every row must have the same number of
  columns, and there must be at least one feature column.

Blank lines and a UTF-8 BOM are ignored. Only plain decimal and scientific
notation is accepted — non-finite values (`NaN`, `Infinity`), hex floats, type
suffixes (`1.5d`, `2f`), and ragged rows are rejected with a line-numbered
error message.

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
javac -d out src/*.java
```

Run from the directory containing your CSV file:

```sh
java -cp /path/to/out KMeans
```

A ready-to-use dataset ships in [`examples/data.csv`](examples/data.csv)
(15 points in 3 obvious clusters). To try it:

```sh
cd examples
java -cp ../out KMeans
# file name: data · clusters: 3 · iterations: 100 · seed rows: 1, 6, 11
```

The program then prompts for:

1. **CSV file name** — the `.csv` extension is optional. Path traversal and
   filenames that are unsafe on any supported platform (`..`, `/`, `\`,
   `:`, `?`, `*`, `"`, `<`, `>`, `|`, control characters, Windows device
   names such as `CON`/`NUL`/`COM1`, trailing dots or spaces) are rejected
   for safety.
2. **Number of clusters** — between 2 and the number of data rows.
3. **Maximum iterations** — an upper bound; the algorithm may stop earlier.
4. **Initial centroid rows** — one distinct row number per cluster. Picking
   rows from visibly different regions of the data gives better results;
   identical or poorly spread seeds can converge to a weaker local optimum
   (an inherent property of K-Means, not a bug).

## Output

Console:

- Clustering execution time (seconds)
- Number of iterations actually run
- The final centroid of each cluster

`new.csv` (written to the current directory; if it already exists — including
when it is the input file — you are asked to confirm before it is overwritten):

- One row per data point: `cluster,id,feature1,feature2,...`
- Identifiers are RFC 4180-quoted when they contain characters needing it,
  and an identifier whose first non-whitespace character is `=`, `+`, `-`,
  or `@` is prefixed with `'` so spreadsheet applications treat it as text
  instead of a formula.
- Each cluster's rows are grouped together, separated by blank lines
- `new.csv` is a report, not re-ingestable input — the writer quotes and
  escapes identifiers, while the reader expects plain unquoted CSV

## How it works

- **Algorithm** — classic Lloyd's algorithm: assign each point to its nearest
  centroid, recompute each centroid as the mean of its members, repeat.
- **Flat data layout** — `Dataset` stores all features in a single row-major
  `double[]` (one source of truth, no per-point copies), so the hot loop reads
  memory sequentially instead of chasing object references.
- **Overflow-safe distances** — Euclidean distances are evaluated with a
  fast squared-sum path that falls back to a max-scaled form when the sum
  would overflow, so ordering stays correct for finite input of any
  magnitude.
- **Overflow-safe centroids** — each centroid is refined as a running mean
  over its members (`mean·(n−1)/n + x/n`), a convex combination that stays
  finite for any finite input where a plain sum could overflow.
- **Empty-cluster reseeding** — if a cluster loses all its points, it is
  re-seeded with the point farthest from its current centroid rather than
  being dropped or crashed on.
- **Convergence** — stops early when every centroid coordinate moves less than
  a relative tolerance of `1e-6`, or when the iteration cap is hit.

## Project layout

| File                   | Purpose                                                        |
|------------------------|----------------------------------------------------------------|
| `src/KMeans.java`      | CLI prompts, filename validation, and run orchestration        |
| `src/DatasetIO.java`   | CSV reading/writing and output escaping                        |
| `src/Engine.java`      | Lloyd's algorithm engine and result holder                     |
| `src/Dataset.java`     | Immutable flat row-major feature matrix + row identifiers      |
| `examples/data.csv`    | Ready-to-use sample dataset (15 points, 3 clusters)            |
| `CONTRIBUTING.md`      | How to report issues and submit pull requests                  |
| `CODE_OF_CONDUCT.md`   | Community expectations for participation                       |
| `LICENSE`              | Apache License 2.0                                             |

## Contributing

Bug reports and pull requests are welcome — see
[CONTRIBUTING.md](CONTRIBUTING.md). Please follow the
[Code of Conduct](CODE_OF_CONDUCT.md).

## License

Apache License 2.0 — see [LICENSE](LICENSE).
