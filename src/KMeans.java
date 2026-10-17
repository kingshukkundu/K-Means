// ============================================================================
//   Copyright 2019 Kingshuk Kundu
//
//   Licensed under the Apache License, Version 2.0 (the "License");
//   you may not use this file except in compliance with the License.
//   You may obtain a copy of the License at
//
//       http://www.apache.org/licenses/LICENSE-2.0
//
//   Unless required by applicable law or agreed to in writing, software
//   distributed under the License is distributed on an "AS IS" BASIS,
//   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//   See the License for the specific language governing permissions and
//   limitations under the License.
// ============================================================================

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.FileAlreadyExistsException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Locale;
import java.util.Scanner;
import java.util.function.Function;
import java.util.regex.Pattern;

/**
 * K-Means clustering application entry point: CLI prompts, input-file
 * validation, and orchestration of {@link DatasetIO} and {@link Engine}.
 */
public class KMeans {

    private static final Pattern WINDOWS_DEVICE_NAME =
        Pattern.compile("(?i)con|prn|aux|nul|com[1-9]|lpt[1-9]");

    private static final int MAX_FILE_NAME_BYTES =
        255 - ".clustered.csv".length() - (2 + 36 + ".tmp".length());

    public static void main(String[] args) {
        try (Scanner sc = new Scanner(System.in)) {
            String fileName;
            while (true) {
                fileName = promptLine(sc, "Enter the name of the CSV file (with or without the .csv extension): ");
                try {
                    validateFileName(fileName);
                    break;
                } catch (IllegalArgumentException e) {
                    System.out.println(e.getMessage());
                }
            }
            // An explicit .csv extension is kept as typed; otherwise appended.
            boolean hasCsvExtension = fileName.toLowerCase(Locale.ROOT).endsWith(".csv");
            Path inputPath = Paths.get(hasCsvExtension ? fileName : fileName + ".csv");
            Dataset dataset = DatasetIO.readDataset(inputPath);
            if (dataset.size() < 2) {
                throw new IllegalArgumentException(
                    "Dataset must contain at least 2 rows to form clusters.");
            }

            int numClusters = promptInt(sc, "Choose the number of clusters (>= 2): ",
                k -> k >= 2 && k <= dataset.size() ? null
                    : "Number of clusters must be between 2 and " + dataset.size() + ".");

            int maxIterations = promptInt(sc, "Please choose the maximum number of iterations: ",
                it -> it >= 1 ? null : "Maximum iterations must be >= 1.");

            // Farthest-point seeding spreads initial centroids across the
            // data deterministically, replacing manual seed selection.
            Engine.KMeansResult result = new Engine(dataset,
                Engine.farthestFirstSeeds(dataset, numClusters)).run(maxIterations);

            System.out.printf(Locale.ROOT, "%nClustering execution time: %.6f seconds%n", result.executionTimeSeconds());
            System.out.println("Total number of iterations: " + result.iterations());

            System.out.println("\nFinal centroids:");
            double[][] centroids = result.centroids();
            for (int i = 0; i < centroids.length; i++) {
                System.out.print("Cluster " + (i + 1) + ": ");
                printFeatures(centroids[i]);
            }

            // Derived from the input name, so the output can never be the
            // input file and results from different datasets don't collide.
            String outputName = (hasCsvExtension
                ? fileName.substring(0, fileName.length() - ".csv".length())
                : fileName) + ".clustered.csv";
            Path outputPath = Paths.get(outputName);
            try {
                DatasetIO.writeClusters(outputPath, dataset, result.clusters(), false);
            } catch (FileAlreadyExistsException e) {
                if (!confirm(sc, "\nAn existing " + outputName
                        + " will be overwritten. Overwrite? [y/N]: ")) {
                    System.out.println("Output not written.");
                    // Distinct from error exits: a caller can tell "ran but
                    // wrote nothing" apart from success (0) or failure (1).
                    System.exit(2);
                }
                DatasetIO.writeClusters(outputPath, dataset, result.clusters(), true);
            }
            System.out.println("\nClustered data written to " + outputName);

        } catch (IllegalArgumentException e) {
            System.err.println("Error: " + e.getMessage());
            System.exit(1);
        } catch (IOException e) {
            System.err.println("Error reading/writing file: " + e.getMessage());
            System.exit(1);
        } catch (Exception e) {
            System.err.println("Unexpected error: " + e.getMessage());
            e.printStackTrace();
            System.exit(1);
        }
    }

    // -------------------------------------------------------------------------
    // Input helpers
    // -------------------------------------------------------------------------

    private static String promptLine(Scanner sc, String message) {
        // print() leaves the prompt buffered; flush or the terminal shows
        // nothing while hasNextLine() blocks.
        System.out.print(message);
        System.out.flush();
        if (!sc.hasNextLine()) {
            throw new IllegalArgumentException("Unexpected end of input.");
        }
        return sc.nextLine().trim();
    }

    // Reprompts until the input parses and the validator accepts it; the
    // validator returns the rejection message to show, or null when valid.
    private static int promptInt(Scanner sc, String message, Function<Integer, String> validator) {
        while (true) {
            String input = promptLine(sc, message);
            int value;
            try {
                value = Integer.parseInt(input);
            } catch (NumberFormatException e) {
                System.out.println("Invalid input '" + input + "'. Please enter a valid integer.");
                continue;
            }
            String error = validator.apply(value);
            if (error == null) {
                return value;
            }
            System.out.println(error);
        }
    }

    private static boolean confirm(Scanner sc, String message) {
        System.out.print(message);
        System.out.flush();
        // EOF must decline — never auto-confirm an overwrite on absent input.
        if (!sc.hasNextLine()) {
            return false;
        }
        String answer = sc.nextLine().trim();
        return answer.equalsIgnoreCase("y") || answer.equalsIgnoreCase("yes");
    }

    // Rejects traversal and names unsafe on Windows: \/:*?"<>|, control
    // chars, device names, whitespace around the base name, and leading or
    // trailing dots (which produce hidden files or an empty stem, e.g.
    // ".csv" names no file).
    private static void validateFileName(String fileName) {
        if (fileName.isEmpty()) {
            throw new IllegalArgumentException("File name cannot be empty.");
        }
        if (fileName.getBytes(StandardCharsets.UTF_8).length > MAX_FILE_NAME_BYTES) {
            throw new IllegalArgumentException(
                "File name is too long (maximum " + MAX_FILE_NAME_BYTES + " bytes).");
        }
        if (fileName.contains("..") || fileName.chars().anyMatch(KMeans::isIllegalFileNameChar)) {
            throw new IllegalArgumentException(
                "Invalid file name. Path separators and other unsafe characters are not allowed.");
        }
        int dot = fileName.indexOf('.');
        String baseName = dot >= 0 ? fileName.substring(0, dot) : fileName;
        String trimmedBase = baseName.trim();
        if (fileName.startsWith(".") || fileName.endsWith(".")
                || trimmedBase.isEmpty() || WINDOWS_DEVICE_NAME.matcher(trimmedBase).matches()
                || !baseName.equals(trimmedBase)) {
            throw new IllegalArgumentException(
                "Invalid file name. Reserved or platform-incompatible names are not allowed.");
        }
    }

    private static boolean isIllegalFileNameChar(int c) {
        return c <= 0x1F || c == 0x7F || "\\/:*?\"<>|".indexOf(c) >= 0;
    }

    private static void printFeatures(double[] features) {
        StringBuilder sb = new StringBuilder();
        for (double f : features) {
            if (sb.length() > 0) {
                sb.append(' ');
            }
            sb.append(String.format(Locale.ROOT, "%.6g", f));
        }
        System.out.println(sb);
    }
}
