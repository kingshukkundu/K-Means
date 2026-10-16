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
import java.nio.file.FileAlreadyExistsException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.Scanner;
import java.util.regex.Pattern;

/**
 * K-Means clustering application entry point.
 * Handles CLI interaction and input-file validation and orchestrates the
 * run; CSV I/O lives in {@link DatasetIO} and the clustering algorithm in
 * {@link Engine}.
 */
public class KMeans {

    private static final Pattern WINDOWS_DEVICE_NAME =
        Pattern.compile("(?i)con|prn|aux|nul|com[1-9]|lpt[1-9]");

    public static void main(String[] args) {
        try (Scanner sc = new Scanner(System.in)) {
            String fileName = promptLine(sc, "Enter the name of the CSV file (with or without the .csv extension): ");
            validateFileName(fileName);
            // An explicit .csv extension is kept exactly as typed so the name
            // resolves correctly on case-sensitive filesystems; otherwise it
            // is appended.
            Path inputPath = Paths.get(fileName.toLowerCase(Locale.ROOT).endsWith(".csv")
                ? fileName
                : fileName + ".csv");
            Dataset dataset = DatasetIO.readDataset(inputPath);

            int numClusters = promptInt(sc, "Choose the number of clusters (>= 2): ");
            if (numClusters < 2) {
                fail("Number of clusters must be >= 2.");
            }
            if (numClusters > dataset.size()) {
                fail("Number of clusters cannot exceed dataset size (" + dataset.size() + ").");
            }

            int maxIterations = promptInt(sc, "Please choose the maximum number of iterations: ");
            if (maxIterations < 1) {
                fail("Maximum iterations must be >= 1.");
            }

            System.out.println("\nPlease pick a row number between 1 and " + dataset.size() + " for each centroid.\n");

            List<Integer> seedIndices = new ArrayList<>();
            for (int i = 1; i <= numClusters; i++) {
                int row = promptInt(sc, "Centroid " + i + ": Pick the row number: ");
                if (row < 1 || row > dataset.size()) {
                    fail("Row number must be between 1 and " + dataset.size() + ".");
                }
                if (seedIndices.contains(row - 1)) {
                    fail("Duplicate row number selected.");
                }
                seedIndices.add(row - 1);
            }

            Engine.KMeansResult result = new Engine(dataset, seedIndices).run(maxIterations);

            System.out.printf(Locale.ROOT, "%nClustering execution time: %.6f seconds%n", result.executionTimeSeconds);
            System.out.println("Total number of iterations: " + result.iterations);

            System.out.println("\nFinal centroids:");
            for (int i = 0; i < result.centroids.length; i++) {
                System.out.print("Cluster " + (i + 1) + ": ");
                printFeatures(result.centroids[i]);
            }

            Path outputPath = Paths.get("new.csv");
            try {
                DatasetIO.writeClusters(outputPath, dataset, result.clusters, false);
            } catch (FileAlreadyExistsException e) {
                String warning = Files.isSameFile(inputPath, outputPath)
                    ? "\nnew.csv is the input file; writing output will overwrite it."
                    : "\nAn existing new.csv will be overwritten.";
                if (!confirm(sc, warning + " Overwrite? [y/N]: ")) {
                    System.out.println("Output not written.");
                    return;
                }
                DatasetIO.writeClusters(outputPath, dataset, result.clusters, true);
            }
            System.out.println("\nClustered data written to new.csv");

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
        System.out.print(message);
        if (!sc.hasNextLine()) {
            throw new IllegalArgumentException("Unexpected end of input.");
        }
        return sc.nextLine().trim();
    }

    private static int promptInt(Scanner sc, String message) {
        while (true) {
            String input = promptLine(sc, message);
            try {
                return Integer.parseInt(input);
            } catch (NumberFormatException e) {
                System.out.println("Invalid input '" + input + "'. Please enter a valid integer.");
            }
        }
    }

    private static boolean confirm(Scanner sc, String message) {
        String answer = promptLine(sc, message);
        return answer.equalsIgnoreCase("y") || answer.equalsIgnoreCase("yes");
    }

    private static void fail(String message) {
        System.err.println("Error: " + message);
        System.exit(1);
    }

    // Rejects directory traversal plus every name that is unsafe on at
    // least one supported platform: \/:*?"<>| (Windows file/drive syntax),
    // control characters, Windows device names (matched on the portion
    // before the first dot, since an extension may be appended), and names
    // ending in a dot or space. Everything else — including non-ASCII
    // names — is left to the filesystem to validate.
    private static void validateFileName(String fileName) {
        if (fileName.isEmpty()) {
            throw new IllegalArgumentException("File name cannot be empty.");
        }
        if (fileName.contains("..") || fileName.chars().anyMatch(KMeans::isIllegalFileNameChar)) {
            throw new IllegalArgumentException(
                "Invalid file name. Path separators and other unsafe characters are not allowed.");
        }
        int dot = fileName.indexOf('.');
        String baseName = dot >= 0 ? fileName.substring(0, dot) : fileName;
        if (WINDOWS_DEVICE_NAME.matcher(baseName).matches()
                || fileName.endsWith(".") || fileName.endsWith(" ")) {
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
