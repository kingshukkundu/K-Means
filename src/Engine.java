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
 *
 * The Dataset owns the feature matrix in flat row-major form, so the hot
 * loop reads memory sequentially instead of chasing object references.
 * Distances are compared in a single linear metric computed
 * overflow-safely, and each centroid is refined as a running mean over
 * its members — a convex combination that can never overflow — so
 * finite input features guarantee finite centroids and correctly
 * ordered distances.
 *
 * The engine performs no I/O: it consumes an immutable {@link Dataset}
 * and seed row indices, and returns an immutable {@link KMeansResult}.
 * {@link #run(int)} is deterministic and may be invoked repeatedly.
 */
final class Engine {

    private static final double CONVERGENCE_TOLERANCE = 1e-6;

    private final Dataset dataset;
    private final int pointCount;
    private final int dimensions;
    private final int[] seeds;
    private final double[][] centroids;
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

        seeds = new int[seedIndices.size()];
        for (int i = 0; i < seeds.length; i++) {
            Integer row = seedIndices.get(i);
            if (row == null || row < 0 || row >= pointCount) {
                throw new IllegalArgumentException(
                    "Seed row index " + row + " is out of range for a dataset of " + pointCount + " rows");
            }
            seeds[i] = row;
        }
        centroids = new double[seeds.length][dimensions];
        counts = new int[seeds.length];
        assignments = new int[pointCount];
    }

    KMeansResult run(int maxIterations) {
        if (maxIterations < 1) {
            throw new IllegalArgumentException("Maximum iterations must be >= 1");
        }
        long startTime = System.nanoTime();

        seedCentroids();
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

    // Restores every centroid to its seed row so a repeated run() produces
    // the same result instead of continuing from the previous run's state.
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

    // Euclidean distance, computed overflow-safely. The common case is a
    // single pass of squared differences plus one square root; if that
    // sum overflows to +Inf, the distance is recomputed as
    // max|d|·sqrt(Σ(d/max|d|)²), which keeps ordering correct for
    // magnitudes beyond sqrt(Double.MAX_VALUE) ≈ 1.3e154. When even a
    // coordinate difference exceeds the largest finite double, the true
    // distance is itself unrepresentable and the result saturates at
    // +Inf — the only representable answer, since no double could order
    // two such distances anyway. A single consistent linear metric is
    // required — mixing squared and scaled values across a point's
    // candidates would corrupt comparisons.
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

    /**
     * Immutable snapshot of one clustering run: point-to-cluster
     * assignments (as index lists into the {@link Dataset}), the final
     * centroid vectors, the number of iterations actually run, and
     * wall-clock execution time. Defensive copies keep the snapshot
     * stable even if the engine is re-run.
     */
    static final class KMeansResult {

        final List<List<Integer>> clusters;
        final double[][] centroids;
        final int iterations;
        final double executionTimeSeconds;

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
    }
}
