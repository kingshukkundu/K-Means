# Contributing

Thanks for your interest in improving this project! Contributions of all
kinds are welcome — bug reports, feature requests, documentation, and code.

## Reporting bugs and requesting features

- Search the existing issues first to avoid duplicates.
- Open a new issue using the **bug report** or **feature request** template
  under the Issues tab. Include your JDK version, the input CSV (or a
  minimal reproducer), and the expected vs. actual behavior.

## Development setup

Requirements: JDK 8 or later. There are no external dependencies.

```sh
# Compile
javac -d out src/*.java

# Run (from a directory containing your CSV file)
cd examples
java -cp ../out KMeans
# file name: data · clusters: 3 · iterations: 100 · seed rows: 1, 6, 11
```

## Code style

- Plain Java — please do not add external dependencies.
- Match the existing style: 4-space indentation, Javadoc on public API.
- Keep `Dataset` immutable — it is the single source of truth for the feature
  matrix — and keep the `Engine` free of I/O; all file and console
  interaction stays in `KMeans.main` and its helpers.
- The project must compile cleanly:

  ```sh
  javac -Xlint:all -d out src/*.java
  ```

## Pull requests

1. Fork the repository and create a branch off `master`.
2. Make your change, compile with `-Xlint:all`, and run the program against
   `examples/data.csv` to verify the output.
3. Open a pull request describing **what** changed and **why**. Include
   sample input/output when the behavior changes.

## Code of conduct

By participating in this project you agree to abide by the
[Code of Conduct](CODE_OF_CONDUCT.md).

## License

By contributing, you agree that your contributions will be licensed under the
[Apache License 2.0](LICENSE).
