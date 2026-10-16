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

/**
 * Immutable representation of a data point with an identifier and feature vector.
 * Separates the ID (column 0 in the source CSV) from the numeric features
 * used by the K-Means algorithm.
 */
public final class DataPoint {

    private final String id;
    private final double[] features;

    public DataPoint(String id, double[] features) {
        if (id == null || id.isEmpty()) {
            throw new IllegalArgumentException("ID cannot be null or empty");
        }
        if (features == null || features.length == 0) {
            throw new IllegalArgumentException("Features cannot be null or empty");
        }
        this.id = id;
        this.features = features.clone();
    }

    public String getId() {
        return id;
    }

    public int getDimensions() {
        return features.length;
    }

    /**
     * Returns a defensive copy of the feature vector.
     */
    public double[] getFeatures() {
        return features.clone();
    }

    /**
     * Copies this point's features into the destination array at the given
     * offset, avoiding an intermediate allocation.
     */
    void copyFeaturesTo(double[] destination, int offset) {
        System.arraycopy(features, 0, destination, offset, features.length);
    }
}
