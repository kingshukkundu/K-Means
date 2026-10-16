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

import java.util.HashSet;
import java.util.List;
import java.util.Set;

/**
 * Immutable dataset: one identifier per row plus a flat row-major feature
 * matrix. Invariants (unique non-empty ids, uniform feature counts, finite
 * values) are enforced at construction.
 */
public final class Dataset {

    private final String[] ids;
    private final double[] data;
    private final int dimensions;

    private Dataset(String[] ids, double[] data, int dimensions) {
        this.ids = ids;
        this.data = data;
        this.dimensions = dimensions;
    }

    /** Creates a dataset from per-row identifiers and feature vectors. */
    public static Dataset of(List<String> ids, List<double[]> rows) {
        return of(ids, rows, null);
    }

    // Variant for the CSV reader: lineNumbers maps each row to its source
    // line so errors cite file lines, not data-row indexes.
    static Dataset of(List<String> ids, List<double[]> rows, int[] lineNumbers) {
        if (ids == null || rows == null || ids.isEmpty() || ids.size() != rows.size()
                || (lineNumbers != null && lineNumbers.length != ids.size())) {
            throw new IllegalArgumentException(
                "IDs and rows must be non-null, non-empty, and equal in size; "
                    + "line numbers, when given, must match that size");
        }

        String[] idArray = new String[ids.size()];
        Set<String> seenIds = new HashSet<>();
        for (int p = 0; p < ids.size(); p++) {
            String rawId = ids.get(p);
            // Trim before validating so "a" and " a" can never coexist.
            String id = rawId == null ? null : rawId.trim();
            if (id == null || id.isEmpty()) {
                throw new IllegalArgumentException("Missing identifier at " + rowRef(lineNumbers, p));
            }
            if (!seenIds.add(id)) {
                throw new IllegalArgumentException(
                    "Duplicate identifier '" + id + "' at " + rowRef(lineNumbers, p));
            }
            idArray[p] = id;
        }

        double[] firstRow = rows.get(0);
        if (firstRow == null || firstRow.length == 0) {
            throw new IllegalArgumentException(
                "Feature vector at " + rowRef(lineNumbers, 0) + " is missing or empty");
        }
        int dimensions = firstRow.length;

        // Integer.MAX_VALUE - 2 is the largest array length the JVM allows
        // and keeps all flat indexes in int range.
        long cellCount = (long) rows.size() * dimensions;
        if (cellCount > Integer.MAX_VALUE - 2L) {
            throw new IllegalArgumentException(
                "Dataset is too large: " + cellCount + " feature values exceeds the supported maximum");
        }
        double[] data = new double[(int) cellCount];
        for (int p = 0; p < rows.size(); p++) {
            double[] row = rows.get(p);
            if (row == null || row.length != dimensions) {
                throw new IllegalArgumentException(
                    "Inconsistent feature count at " + rowRef(lineNumbers, p));
            }
            // NaN/Infinity would corrupt distance comparisons and centroids.
            for (int d = 0; d < dimensions; d++) {
                double value = row[d];
                if (!Double.isFinite(value)) {
                    throw new IllegalArgumentException(
                        "Non-finite feature value at " + rowRef(lineNumbers, p)
                            + ", feature " + (d + 1));
                }
                data[p * dimensions + d] = value;
            }
        }

        return new Dataset(idArray, data, dimensions);
    }

    private static String rowRef(int[] lineNumbers, int p) {
        return lineNumbers == null ? "row " + (p + 1) : "line " + lineNumbers[p];
    }

    /**
     * Number of data points.
     */
    public int size() {
        return ids.length;
    }

    /**
     * Number of features per point.
     */
    public int dimensions() {
        return dimensions;
    }

    /**
     * Identifier of the given point.
     */
    public String id(int point) {
        return ids[point];
    }

    /**
     * Feature {@code d} of the given point.
     */
    public double feature(int point, int d) {
        return data[point * dimensions + d];
    }

    /**
     * Copies one point's feature vector into the destination array,
     * avoiding an intermediate allocation.
     */
    void copyFeaturesTo(int point, double[] destination, int offset) {
        System.arraycopy(data, point * dimensions, destination, offset, dimensions);
    }
}
