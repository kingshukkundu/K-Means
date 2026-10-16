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

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

/**
 * K-Means engine: Lloyd's algorithm with farthest-point re-seeding.
 * Performs no I/O; {@link #run(int)} is deterministic and repeatable.
 */
final class Engine {

    private static final double CONVERGENCE_TOLERANCE = 1e-6;
    // Floor under the relative tolerance so a centroid at exactly zero is
    // not held to a bitwise-exact match.
    private static final double MIN_CONVERGENCE_SCALE = 1e-12;

    private final Dataset dataset;
    private final int pointCount;
    private final int dimensions;
    private final int[] seeds;
    private final double[][] centroids;
    private final double[][] previousCentroids;
    private final int[] counts;
    private final int[] assignments;

    Engine(Dataset dataset, List<Integer> seedIndices) {
        if (dataset == null || seedIndices == null || seedIndices.isEmpty()) {
            throw new IllegalArgumentException(
                "Dataset must be non-null and at least one seed index is required");
        }
        this.dataset = dataset;
        this.pointCount = dataset.size();
        this.dimensions = dataset.dimensions();

        // Distinct in-range seeds imply k <= pointCount, which the
        // empty-cluster reseeding step relies on.
        boolean[] used = new boolean[pointCount];
        seeds = new int[seedIndices.size()];
        for (int i = 0; i < seeds.length; i++) {
            Integer row = seedIndices.get(i);
            if (row == null || row < 0 || row >= pointCount) {
                throw new IllegalArgumentException(
                    "Seed row index " + row + " is out of range for a dataset of " + pointCount + " rows");
            }
            if (used[row]) {
                throw new IllegalArgumentException("Duplicate seed row index " + row);
            }
            used[row] = true;
            seeds[i] = row;
        }
        centroids = new double[seeds.length][dimensions];
        previousCentroids = new double[seeds.length][dimensions];
        counts = new int[seeds.length];
        assignments = new int[pointCount];
    }

    KMeansResult run(int maxIterations) {
        if (maxIterations < 1) {
            throw new IllegalArgumentException("Maximum iterations must be >= 1");
        }
        long startTime = System.nanoTime();

        // Iteration 1 starts from the seed vectors, so an already-optimal
        // seed set converges in a single pass.
        seedCentroids();
        int iterations = 0;
        while (iterations < maxIterations) {
            snapshotCentroids();
            assignAll();
            reseedEmptyClusters();
            recomputeCentroids();
            iterations++;

            if (hasConverged()) {
                break;
            }
        }

        double executionTime = (System.nanoTime() - startTime) / 1_000_000_000.0;
        return new KMeansResult(buildClusters(), centroids, iterations, executionTime);
    }

    // Restores seed vectors so a repeated run() starts from the same state.
    private void seedCentroids() {
        for (int i = 0; i < seeds.length; i++) {
            dataset.copyFeaturesTo(seeds[i], centroids[i], 0);
        }
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
        double minDist = distanceTo(point, centroids[0]);
        for (int i = 1; i < centroids.length; i++) {
            double dist = distanceTo(point, centroids[i]);
            if (dist < minDist) {
                minDist = dist;
                nearest = i;
            }
        }
        return nearest;
    }

    // Euclidean distance: overflows fall back to a max-scaled form so
    // ordering stays correct; unrepresentable distances saturate at +Inf.
    private double distanceTo(int point, double[] centroid) {
        double sum = 0.0;
        for (int d = 0; d < dimensions; d++) {
            double diff = dataset.feature(point, d) - centroid[d];
            sum += diff * diff;
        }
        if (sum < Double.POSITIVE_INFINITY) {
            return Math.sqrt(sum);
        }
        double maxAbs = 0.0;
        for (int d = 0; d < dimensions; d++) {
            double abs = Math.abs(dataset.feature(point, d) - centroid[d]);
            if (abs > maxAbs) {
                maxAbs = abs;
            }
        }
        if (maxAbs == Double.POSITIVE_INFINITY) {
            return Double.POSITIVE_INFINITY;
        }
        double scaled = 0.0;
        for (int d = 0; d < dimensions; d++) {
            double ratio = (dataset.feature(point, d) - centroid[d]) / maxAbs;
            scaled += ratio * ratio;
        }
        return maxAbs * Math.sqrt(scaled);
    }

    // Re-seeds each empty cluster with the point farthest from its current
    // centroid; donor clusters must keep at least 1 member.
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
                double dist = distanceTo(p, centroids[source]);
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

    // Recomputes each centroid as a running mean — a convex combination
    // that cannot overflow. An empty cluster keeps its previous centroid.
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

    // Persistent buffer refilled each iteration — no per-pass allocation
    // and no stale values across repeated run() calls.
    private void snapshotCentroids() {
        for (int i = 0; i < centroids.length; i++) {
            System.arraycopy(centroids[i], 0, previousCentroids[i], 0, dimensions);
        }
    }

    private boolean hasConverged() {
        for (int i = 0; i < centroids.length; i++) {
            for (int d = 0; d < dimensions; d++) {
                double scale = Math.max(MIN_CONVERGENCE_SCALE, Math.abs(centroids[i][d]));
                if (Math.abs(centroids[i][d] - previousCentroids[i][d])
                        > CONVERGENCE_TOLERANCE * scale) {
                    return false;
                }
            }
        }
        return true;
    }

    /**
     * Immutable snapshot of one clustering run: cluster assignments, final
     * centroids, iteration count, and wall-clock time. Defensively copied.
     */
    static final class KMeansResult {

        private final List<List<Integer>> clusters;
        private final double[][] centroids;
        private final int iterations;
        private final double executionTimeSeconds;

        KMeansResult(List<List<Integer>> clusters, double[][] centroids, int iterations, double executionTimeSeconds) {
            List<List<Integer>> clusterCopy = new ArrayList<>(clusters.size());
            for (List<Integer> cluster : clusters) {
                clusterCopy.add(Collections.unmodifiableList(new ArrayList<>(cluster)));
            }
            this.clusters = Collections.unmodifiableList(clusterCopy);

            double[][] centroidCopy = new double[centroids.length][];
            for (int i = 0; i < centroids.length; i++) {
                centroidCopy[i] = centroids[i].clone();
            }
            this.centroids = centroidCopy;
            this.iterations = iterations;
            this.executionTimeSeconds = executionTimeSeconds;
        }

        List<List<Integer>> clusters() {
            return clusters;
        }

        // Fresh copy: double[][] is inherently mutable.
        double[][] centroids() {
            double[][] copy = new double[centroids.length][];
            for (int i = 0; i < centroids.length; i++) {
                copy[i] = centroids[i].clone();
            }
            return copy;
        }

        int iterations() {
            return iterations;
        }

        double executionTimeSeconds() {
            return executionTimeSeconds;
        }
    }
}
