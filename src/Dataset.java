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

import java.util.List;

/**
 * Immutable storage for the input dataset: one identifier per row plus a
 * single flat row-major feature matrix.
 *
 * Keeping all features contiguous is the single source of truth for both the
 * clustering engine and output — the hot loops read memory sequentially
 * instead of chasing per-point object references, and no feature value is
 * ever stored twice.
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

    /**
     * Creates a dataset from per-row identifiers and feature vectors.
     * Every row must have the same non-zero number of features.
     */
    public static Dataset of(List<String> ids, List<double[]> rows) {
        if (ids == null || rows == null || ids.isEmpty() || ids.size() != rows.size()) {
            throw new IllegalArgumentException(
                "IDs and rows must be non-null, non-empty, and equal in size");
        }

        String[] idArray = new String[ids.size()];
        for (int p = 0; p < ids.size(); p++) {
            String id = ids.get(p);
            if (id == null || id.isEmpty()) {
                throw new IllegalArgumentException("Missing identifier at row " + (p + 1));
            }
            idArray[p] = id;
        }

        int dimensions = rows.get(0) == null ? 0 : rows.get(0).length;
        if (dimensions == 0) {
            throw new IllegalArgumentException("Feature vectors cannot be empty");
        }

        // Integer.MAX_VALUE - 2 is the largest array length the JVM allows;
        // the bound also keeps every point * dimensions + d index below in
        // feature()/copyFeaturesTo() inside int range.
        long cellCount = (long) rows.size() * dimensions;
        if (cellCount > Integer.MAX_VALUE - 2L) {
            throw new IllegalArgumentException(
                "Dataset is too large: " + cellCount + " feature values exceeds the supported maximum");
        }
        double[] data = new double[(int) cellCount];
        for (int p = 0; p < rows.size(); p++) {
            double[] row = rows.get(p);
            if (row == null || row.length != dimensions) {
                throw new IllegalArgumentException("Inconsistent feature count at row " + (p + 1));
            }
            System.arraycopy(row, 0, data, p * dimensions, dimensions);
        }

        return new Dataset(idArray, data, dimensions);
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
