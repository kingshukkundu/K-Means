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

import java.io.BufferedWriter;
import java.io.IOException;
import java.math.BigDecimal;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Locale;
import java.util.Scanner;
import java.util.Set;

/**
 * K-Means clustering application.
 * Handles CLI interaction, CSV I/O, and orchestrates the K-Means algorithm.
 */
public class KMeans {

    private static final double CONVERGENCE_TOLERANCE = 1e-6;

    public static void main(String[] args) {
        try (Scanner sc = new Scanner(System.in)) {
            String fileName = promptLine(sc, "Enter the name of the CSV file (with or without the .csv extension): ");
            if (fileName.toLowerCase(Locale.ROOT).endsWith(".csv")) {
                fileName = fileName.substring(0, fileName.length() - ".csv".length());
            }
            validateFileName(fileName);
            Path inputPath = Paths.get(fileName + ".csv");
            Dataset dataset = readDataset(inputPath);

            int numClusters = promptInt(sc, "Choose the number of clusters (>= 2): ");
            if (numClusters < 2) {
                System.err.println("Error: Number of clusters must be >= 2.");
                return;
            }
            if (numClusters > dataset.size()) {
                System.err.println("Error: Number of clusters cannot exceed dataset size (" + dataset.size() + ").");
                return;
            }

            int maxIterations = promptInt(sc, "Please choose the maximum number of iterations: ");
            if (maxIterations < 1) {
                System.err.println("Error: Maximum iterations must be >= 1.");
                return;
            }

            System.out.println("\nPlease pick a row number between 1 and " + dataset.size() + " for each centroid.\n");

            List<Integer> seedIndices = new ArrayList<>();
            for (int i = 1; i <= numClusters; i++) {
                int row = promptInt(sc, "Centroid " + i + ": Pick the row number: ");
                if (row < 1 || row > dataset.size()) {
                    System.err.println("Error: Row number must be between 1 and " + dataset.size() + ".");
                    return;
                }
                if (seedIndices.contains(row - 1)) {
                    System.err.println("Error: Duplicate row number selected.");
                    return;
                }
                seedIndices.add(row - 1);
            }

            KMeansResult result = new Engine(dataset, seedIndices).run(maxIterations);

            System.out.printf("%nClustering execution time: %.6f seconds%n", result.executionTimeSeconds);
            System.out.println("Total number of iterations: " + result.iterations);

            System.out.println("\nFinal centroids:");
            for (int i = 0; i < result.centroids.length; i++) {
                System.out.print("Cluster " + (i + 1) + ": ");
                printFeatures(result.centroids[i]);
            }

            Path outputPath = Paths.get("new.csv");
            if (Files.exists(outputPath)) {
                boolean isInput = Files.isSameFile(inputPath, outputPath);
                String warning = isInput
                    ? "\nnew.csv is the input file; writing output will overwrite it."
                    : "\nAn existing new.csv will be overwritten.";
                if (!confirm(sc, warning + " Overwrite? [y/N]: ")) {
                    System.out.println("Output not written.");
                    return;
                }
            }
            writeClusters(outputPath, dataset, result.clusters);
            System.out.println("\nClustered data written to new.csv");

        } catch (IllegalArgumentException e) {
            System.err.println("Error: " + e.getMessage());
        } catch (IOException e) {
            System.err.println("Error reading/writing file: " + e.getMessage());
        } catch (Exception e) {
            System.err.println("Unexpected error: " + e.getMessage());
            e.printStackTrace();
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

    // Rejects directory traversal plus every character that is unsafe on at
    // least one supported platform: \/:*?"<>| (Windows file/drive syntax) and
    // control characters. Everything else — including non-ASCII names — is
    // left to the filesystem to validate.
    private static void validateFileName(String fileName) {
        if (fileName.isEmpty()) {
            throw new IllegalArgumentException("File name cannot be empty.");
        }
        if (fileName.contains("..") || fileName.chars().anyMatch(KMeans::isIllegalFileNameChar)) {
            throw new IllegalArgumentException(
                "Invalid file name. Path separators and other unsafe characters are not allowed.");
        }
    }

    private static boolean isIllegalFileNameChar(int c) {
        return c <= 0x1F || c == 0x7F || "\\/:*?\"<>|".indexOf(c) >= 0;
    }

    // -------------------------------------------------------------------------
    // CSV I/O
    // -------------------------------------------------------------------------

    private static Dataset readDataset(Path path) throws IOException {
        List<String> ids = new ArrayList<>();
        List<double[]> rows = new ArrayList<>();
        List<String> lines = Files.readAllLines(path, StandardCharsets.UTF_8);

        if (lines.isEmpty()) {
            throw new IllegalArgumentException("CSV file is empty.");
        }

        int expectedFeatureCount = -1;
        int lineNumber = 0;
        Set<String> seenIds = new HashSet<>();

        for (String rawLine : lines) {
            lineNumber++;
            String line = rawLine.trim();
            if (lineNumber == 1 && line.startsWith("\uFEFF")) {
                line = line.substring(1).trim();
            }
            if (line.isEmpty()) {
                continue;
            }

            String[] tokens = line.split(",", -1);
            boolean firstDataRow = expectedFeatureCount == -1;
            if (firstDataRow) {
                if (tokens.length < 2) {
                    throw new IllegalArgumentException(
                        "Line " + lineNumber + ": Expected at least 2 columns (ID + at least one feature), found " + tokens.length);
                }
                expectedFeatureCount = tokens.length - 1;
            } else if (tokens.length != expectedFeatureCount + 1) {
                throw new IllegalArgumentException(
                    "Line " + lineNumber + ": Expected " + (expectedFeatureCount + 1)
                        + " columns, found " + tokens.length + ". All rows must have the same number of columns.");
            }

            String id = tokens[0].trim();
            if (id.isEmpty()) {
                throw new IllegalArgumentException(
                    "Line " + lineNumber + ": Missing identifier in the first column.");
            }
            if (!seenIds.add(id)) {
                throw new IllegalArgumentException(
                    "Line " + lineNumber + ": Duplicate identifier '" + id + "'. Identifiers must be unique.");
            }

            double[] features = new double[expectedFeatureCount];
            for (int i = 1; i < tokens.length; i++) {
                try {
                    // BigDecimal accepts plain decimal/scientific notation only —
                    // it rejects hex floats, NaN/Infinity, and d/f suffixes that
                    // Double.parseDouble would silently accept.
                    features[i - 1] = new BigDecimal(tokens[i].trim()).doubleValue();
                } catch (NumberFormatException e) {
                    throw new IllegalArgumentException(
                        "Line " + lineNumber + ": Invalid numeric value"
                            + (firstDataRow ? ". If the file has a header row, remove it first." : ""), e);
                }
                if (!Double.isFinite(features[i - 1])) {
                    throw new IllegalArgumentException(
                        "Line " + lineNumber + ": Non-finite numeric value '" + tokens[i].trim() + "'");
                }
            }

            ids.add(id);
            rows.add(features);
        }

        if (ids.isEmpty()) {
            throw new IllegalArgumentException("No valid data rows found in CSV file.");
        }

        return Dataset.of(ids, rows);
    }

    private static void writeClusters(Path path, Dataset dataset, List<List<Integer>> clusters) throws IOException {
        try (BufferedWriter writer = Files.newBufferedWriter(path, StandardCharsets.UTF_8)) {
            for (int clusterNum = 1; clusterNum <= clusters.size(); clusterNum++) {
                if (clusterNum > 1) {
                    writer.write("\n\n\n\n");
                }
                for (int point : clusters.get(clusterNum - 1)) {
                    StringBuilder sb = new StringBuilder();
                    sb.append(clusterNum).append(',');
                    sb.append(dataset.id(point));
                    for (int d = 0; d < dataset.dimensions(); d++) {
                        sb.append(',').append(dataset.feature(point, d));
                    }
                    sb.append('\n');
                    writer.write(sb.toString());
                }
            }
        }
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

    // -------------------------------------------------------------------------
    // K-Means engine: Lloyd's algorithm with farthest-point re-seeding.
    //
    // The Dataset owns the feature matrix in flat row-major form, so the hot
    // loop reads memory sequentially instead of chasing object references.
    // Distances are compared in squared form, avoiding a square root per
    // evaluation, and each centroid is refined as a running mean over its
    // members — a convex combination that can never overflow, so finite
    // input features guarantee finite centroids.
    // -------------------------------------------------------------------------

    private static final class Engine {

        private final Dataset dataset;
        private final int pointCount;
        private final int dimensions;
        private final double[][] centroids;
        private final int[] counts;
        private final int[] assignments;

        Engine(Dataset dataset, List<Integer> seedIndices) {
            this.dataset = dataset;
            this.pointCount = dataset.size();
            this.dimensions = dataset.dimensions();

            int k = seedIndices.size();
            centroids = new double[k][dimensions];
            for (int i = 0; i < k; i++) {
                dataset.copyFeaturesTo(seedIndices.get(i), centroids[i], 0);
            }
            counts = new int[k];
            assignments = new int[pointCount];
        }

        KMeansResult run(int maxIterations) {
            long startTime = System.nanoTime();

            assignAll();
            reseedEmptyClusters();
            recomputeCentroids();
            int iterations = 1;

            while (iterations < maxIterations) {
                double[][] previous = copyCentroids();
                assignAll();
                reseedEmptyClusters();
                recomputeCentroids();
                iterations++;

                if (hasConverged(previous)) {
                    break;
                }
            }

            double executionTime = (System.nanoTime() - startTime) / 1_000_000_000.0;
            return new KMeansResult(buildClusters(), centroids, iterations, executionTime);
        }

        private void assignAll() {
            Arrays.fill(counts, 0);
            for (int p = 0; p < pointCount; p++) {
                int nearest = nearestCentroid(p);
                assignments[p] = nearest;
                counts[nearest]++;
            }
        }

        private int nearestCentroid(int point) {
            int nearest = 0;
            double minDist = squaredDistance(point, centroids[0]);
            for (int i = 1; i < centroids.length; i++) {
                double dist = squaredDistance(point, centroids[i]);
                if (dist < minDist) {
                    minDist = dist;
                    nearest = i;
                }
            }
            return nearest;
        }

        private double squaredDistance(int point, double[] centroid) {
            double sum = 0.0;
            for (int d = 0; d < dimensions; d++) {
                double diff = dataset.feature(point, d) - centroid[d];
                sum += diff * diff;
            }
            return sum;
        }

        // Re-seed empty clusters with the point farthest from its current centroid.
        // Donor clusters must have at least 2 members so no donor is left empty;
        // since k never exceeds the dataset size, a valid donor always exists.
        private void reseedEmptyClusters() {
            for (int i = 0; i < centroids.length; i++) {
                if (counts[i] != 0) {
                    continue;
                }
                int farthest = -1;
                double maxDist = -1;
                for (int p = 0; p < pointCount; p++) {
                    int source = assignments[p];
                    if (counts[source] < 2) {
                        continue;
                    }
                    double dist = squaredDistance(p, centroids[source]);
                    if (dist > maxDist) {
                        maxDist = dist;
                        farthest = p;
                    }
                }
                if (farthest < 0) {
                    continue;
                }
                int source = assignments[farthest];
                assignments[farthest] = i;
                counts[source]--;
                counts[i]++;
            }
        }

        // Recomputes each centroid as a running mean over its assigned points:
        // mean * (n-1)/n + x/n is a convex combination, so it stays finite for
        // any finite input where a plain sum could overflow. counts is refilled
        // here and ends up holding the same totals produced above — an empty
        // cluster simply keeps its previous centroid.
        private void recomputeCentroids() {
            Arrays.fill(counts, 0);
            for (int p = 0; p < pointCount; p++) {
                int c = assignments[p];
                int n = ++counts[c];
                double[] centroid = centroids[c];
                if (n == 1) {
                    dataset.copyFeaturesTo(p, centroid, 0);
                    continue;
                }
                double keep = (n - 1.0) / n;
                double add = 1.0 / n;
                for (int d = 0; d < dimensions; d++) {
                    centroid[d] = centroid[d] * keep + dataset.feature(p, d) * add;
                }
            }
        }

        private List<List<Integer>> buildClusters() {
            List<List<Integer>> clusters = new ArrayList<>(centroids.length);
            for (int i = 0; i < centroids.length; i++) {
                clusters.add(new ArrayList<>());
            }
            for (int p = 0; p < pointCount; p++) {
                clusters.get(assignments[p]).add(p);
            }
            return clusters;
        }

        private double[][] copyCentroids() {
            double[][] copy = new double[centroids.length][];
            for (int i = 0; i < centroids.length; i++) {
                copy[i] = centroids[i].clone();
            }
            return copy;
        }

        private boolean hasConverged(double[][] previous) {
            for (int i = 0; i < centroids.length; i++) {
                for (int d = 0; d < dimensions; d++) {
                    double scale = Math.max(1.0, Math.abs(centroids[i][d]));
                    if (Math.abs(centroids[i][d] - previous[i][d]) > CONVERGENCE_TOLERANCE * scale) {
                        return false;
                    }
                }
            }
            return true;
        }
    }

    // -------------------------------------------------------------------------
    // Result holder
    // -------------------------------------------------------------------------

    private static final class KMeansResult {
        final List<List<Integer>> clusters;
        final double[][] centroids;
        final int iterations;
        final double executionTimeSeconds;

        KMeansResult(List<List<Integer>> clusters, double[][] centroids, int iterations, double executionTimeSeconds) {
            this.clusters = clusters;
            this.centroids = centroids;
            this.iterations = iterations;
            this.executionTimeSeconds = executionTimeSeconds;
        }
    }
}
