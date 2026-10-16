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
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
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
            String fileName = promptLine(sc, "Enter the name of the CSV file (without .csv extension): ");
            validateFileName(fileName);
            List<DataPoint> dataset = readDataset(fileName + ".csv");

            int numClusters = promptInt(sc, "Choose the number of clusters (>= 2): ");
            if (numClusters < 2) {
                System.out.println("Error: Number of clusters must be >= 2.");
                return;
            }
            if (numClusters > dataset.size()) {
                System.out.println("Error: Number of clusters cannot exceed dataset size (" + dataset.size() + ").");
                return;
            }

            int maxIterations = promptInt(sc, "Please choose the maximum number of iterations: ");
            if (maxIterations < 1) {
                System.out.println("Error: Maximum iterations must be >= 1.");
                return;
            }

            System.out.println("\nPlease pick a row number between 1 and " + dataset.size() + " for each centroid.\n");

            List<Integer> seedIndices = new ArrayList<>();
            Set<Integer> chosenRows = new HashSet<>();
            for (int i = 1; i <= numClusters; i++) {
                int row = promptInt(sc, "Centroid " + i + ": Pick the row number: ");
                if (row < 1 || row > dataset.size()) {
                    System.out.println("Error: Row number must be between 1 and " + dataset.size() + ".");
                    return;
                }
                if (!chosenRows.add(row)) {
                    System.out.println("Error: Duplicate row number selected.");
                    return;
                }
                seedIndices.add(row - 1);
            }

            KMeansResult result = new Engine(dataset, seedIndices).run(maxIterations);

            System.out.printf("%nTotal execution time: %.6f seconds%n", result.executionTimeSeconds);
            System.out.println("Total number of iterations: " + result.iterations);

            System.out.println("\nFinal centroids:");
            for (int i = 0; i < result.centroids.length; i++) {
                System.out.print("Cluster " + (i + 1) + ": ");
                printFeatures(result.centroids[i]);
            }

            Path outputPath = Paths.get("new.csv");
            if (Files.exists(outputPath)) {
                System.out.println("\nNote: an existing new.csv will be overwritten.");
            }
            writeClusters("new.csv", result.clusters);
            System.out.println("\nClustered data written to new.csv");

        } catch (IllegalArgumentException e) {
            System.out.println("Error: " + e.getMessage());
        } catch (IOException e) {
            System.out.println("Error reading/writing file: " + e.getMessage());
        } catch (Exception e) {
            System.out.println("Unexpected error: " + e.getMessage());
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
        System.out.print(message);
        while (!sc.hasNextInt()) {
            if (!sc.hasNext()) {
                throw new IllegalArgumentException("Unexpected end of input.");
            }
            String invalid = sc.next();
            System.out.println("Invalid input '" + invalid + "'. Please enter a valid integer.");
            System.out.print(message);
        }
        int value = sc.nextInt();
        if (sc.hasNextLine()) {
            sc.nextLine(); // consume remaining newline
        }
        return value;
    }

    private static void validateFileName(String fileName) {
        if (fileName.isEmpty()) {
            throw new IllegalArgumentException("File name cannot be empty.");
        }
        if (fileName.contains("..") || fileName.contains("/") || fileName.contains("\\")) {
            throw new IllegalArgumentException("Invalid file name. Directory traversal is not allowed.");
        }
    }

    // -------------------------------------------------------------------------
    // CSV I/O
    // -------------------------------------------------------------------------

    private static List<DataPoint> readDataset(String path) throws IOException {
        List<DataPoint> points = new ArrayList<>();
        List<String> lines = Files.readAllLines(Paths.get(path), StandardCharsets.UTF_8);

        if (lines.isEmpty()) {
            throw new IllegalArgumentException("CSV file is empty.");
        }

        int expectedFeatureCount = -1;
        int lineNumber = 0;

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
            if (expectedFeatureCount == -1) {
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

            double[] features = new double[expectedFeatureCount];
            for (int i = 1; i < tokens.length; i++) {
                try {
                    features[i - 1] = Double.parseDouble(tokens[i].trim());
                } catch (NumberFormatException e) {
                    throw new IllegalArgumentException(
                        "Line " + lineNumber + ": Invalid numeric value"
                            + (lineNumber == 1 ? ". If the file has a header row, remove it first." : ""), e);
                }
                if (!Double.isFinite(features[i - 1])) {
                    throw new IllegalArgumentException(
                        "Line " + lineNumber + ": Non-finite numeric value '" + tokens[i].trim() + "'");
                }
            }

            points.add(new DataPoint(id, features));
        }

        if (points.isEmpty()) {
            throw new IllegalArgumentException("No valid data rows found in CSV file.");
        }

        return points;
    }

    private static void writeClusters(String path, List<List<DataPoint>> clusters) throws IOException {
        try (BufferedWriter writer = Files.newBufferedWriter(Paths.get(path), StandardCharsets.UTF_8)) {
            int clusterNum = 1;
            for (List<DataPoint> cluster : clusters) {
                for (DataPoint point : cluster) {
                    StringBuilder sb = new StringBuilder();
                    sb.append(clusterNum).append(',');
                    sb.append(point.getId());
                    for (double feature : point.getFeatures()) {
                        sb.append(',').append(feature);
                    }
                    sb.append('\n');
                    writer.write(sb.toString());
                }
                writer.write("\n\n\n\n");
                clusterNum++;
            }
        }
    }

    private static void printFeatures(double[] features) {
        for (double f : features) {
            System.out.print(f + " ");
        }
        System.out.println();
    }

    // -------------------------------------------------------------------------
    // K-Means engine: Lloyd's algorithm with farthest-point re-seeding.
    //
    // Features live in a single flat row-major array rather than per-point
    // objects, so the hot loop reads memory sequentially instead of chasing
    // references. Cluster sums and counts are accumulated during the
    // assignment pass, so each iteration is a single sweep over the dataset.
    // Distances are compared in squared form, avoiding a square root per
    // distance evaluation.
    // -------------------------------------------------------------------------

    private static final class Engine {

        private final List<DataPoint> dataset;
        private final double[] data;
        private final int pointCount;
        private final int dimensions;
        private final double[][] centroids;
        private final double[][] sums;
        private final int[] counts;
        private final int[] assignments;

        Engine(List<DataPoint> dataset, List<Integer> seedIndices) {
            this.dataset = dataset;
            this.pointCount = dataset.size();
            this.dimensions = dataset.get(0).getDimensions();

            data = new double[pointCount * dimensions];
            for (int p = 0; p < pointCount; p++) {
                dataset.get(p).copyFeaturesTo(data, p * dimensions);
            }

            int k = seedIndices.size();
            centroids = new double[k][dimensions];
            for (int i = 0; i < k; i++) {
                System.arraycopy(data, seedIndices.get(i) * dimensions, centroids[i], 0, dimensions);
            }
            sums = new double[k][dimensions];
            counts = new int[k];
            assignments = new int[pointCount];
        }

        KMeansResult run(int maxIterations) {
            long startTime = System.nanoTime();

            assignAndAccumulate();
            reseedEmptyClusters();
            recomputeCentroids();
            int iterations = 1;

            while (iterations < maxIterations) {
                double[][] previous = copyCentroids();
                resetAccumulators();
                assignAndAccumulate();
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

        private void assignAndAccumulate() {
            for (int p = 0, offset = 0; p < pointCount; p++, offset += dimensions) {
                int nearest = nearestCentroid(offset);
                assignments[p] = nearest;
                counts[nearest]++;
                double[] sum = sums[nearest];
                for (int d = 0; d < dimensions; d++) {
                    sum[d] += data[offset + d];
                }
            }
        }

        private int nearestCentroid(int offset) {
            int nearest = 0;
            double minDist = squaredDistance(offset, centroids[0]);
            for (int i = 1; i < centroids.length; i++) {
                double dist = squaredDistance(offset, centroids[i]);
                if (dist < minDist) {
                    minDist = dist;
                    nearest = i;
                }
            }
            return nearest;
        }

        private double squaredDistance(int offset, double[] centroid) {
            double sum = 0.0;
            for (int d = 0; d < dimensions; d++) {
                double diff = data[offset + d] - centroid[d];
                sum += diff * diff;
            }
            return sum;
        }

        private void resetAccumulators() {
            Arrays.fill(counts, 0);
            for (double[] sum : sums) {
                Arrays.fill(sum, 0.0);
            }
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
                for (int p = 0, offset = 0; p < pointCount; p++, offset += dimensions) {
                    int source = assignments[p];
                    if (counts[source] < 2) {
                        continue;
                    }
                    double dist = squaredDistance(offset, centroids[source]);
                    if (dist > maxDist) {
                        maxDist = dist;
                        farthest = p;
                    }
                }
                if (farthest < 0) {
                    continue;
                }
                int source = assignments[farthest];
                int offset = farthest * dimensions;
                assignments[farthest] = i;
                counts[source]--;
                counts[i]++;
                for (int d = 0; d < dimensions; d++) {
                    double value = data[offset + d];
                    sums[source][d] -= value;
                    sums[i][d] += value;
                }
            }
        }

        private void recomputeCentroids() {
            for (int i = 0; i < centroids.length; i++) {
                if (counts[i] == 0) {
                    continue;
                }
                double inverse = 1.0 / counts[i];
                double[] centroid = new double[dimensions];
                for (int d = 0; d < dimensions; d++) {
                    centroid[d] = sums[i][d] * inverse;
                }
                centroids[i] = centroid;
            }
        }

        private List<List<DataPoint>> buildClusters() {
            List<List<DataPoint>> clusters = new ArrayList<>(centroids.length);
            for (int i = 0; i < centroids.length; i++) {
                clusters.add(new ArrayList<>());
            }
            for (int p = 0; p < pointCount; p++) {
                clusters.get(assignments[p]).add(dataset.get(p));
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
        final List<List<DataPoint>> clusters;
        final double[][] centroids;
        final int iterations;
        final double executionTimeSeconds;

        KMeansResult(List<List<DataPoint>> clusters, double[][] centroids, int iterations, double executionTimeSeconds) {
            this.clusters = clusters;
            this.centroids = centroids;
            this.iterations = iterations;
            this.executionTimeSeconds = executionTimeSeconds;
        }
    }
}
