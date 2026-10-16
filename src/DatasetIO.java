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

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.IOException;
import java.math.BigDecimal;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.OpenOption;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

/**
 * CSV serialization for {@link Dataset}: reads the plain comma-separated
 * input format (identifier column plus numeric feature columns, no header
 * row, no quoting support) and writes cluster-grouped results. Output
 * escapes identifiers per RFC 4180 and defuses spreadsheet formula
 * injection; input is streamed line by line so file size is not bounded
 * by a whole-file buffer.
 */
final class DatasetIO {

    private DatasetIO() {
    }

    static Dataset readDataset(Path path) throws IOException {
        List<String> ids = new ArrayList<>();
        List<double[]> rows = new ArrayList<>();
        int expectedFeatureCount = -1;
        int lineNumber = 0;
        Set<String> seenIds = new HashSet<>();

        try (BufferedReader reader = Files.newBufferedReader(path, StandardCharsets.UTF_8)) {
            String rawLine;
            while ((rawLine = reader.readLine()) != null) {
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
        }

        if (lineNumber == 0) {
            throw new IllegalArgumentException("CSV file is empty.");
        }
        if (ids.isEmpty()) {
            throw new IllegalArgumentException("No valid data rows found in CSV file.");
        }

        return Dataset.of(ids, rows);
    }

    // With overwrite=false the file is opened CREATE_NEW, so the existence
    // check is atomic with the write — a file created after the prompt can
    // never be silently overwritten.
    static void writeClusters(Path path, Dataset dataset, List<List<Integer>> clusters,
                              boolean overwrite) throws IOException {
        OpenOption[] options = overwrite
            ? new OpenOption[] { StandardOpenOption.CREATE, StandardOpenOption.TRUNCATE_EXISTING,
                                 StandardOpenOption.WRITE }
            : new OpenOption[] { StandardOpenOption.CREATE_NEW, StandardOpenOption.WRITE };
        try (BufferedWriter writer = Files.newBufferedWriter(path, StandardCharsets.UTF_8, options)) {
            for (int clusterNum = 1; clusterNum <= clusters.size(); clusterNum++) {
                if (clusterNum > 1) {
                    writer.write("\n\n\n\n");
                }
                for (int point : clusters.get(clusterNum - 1)) {
                    StringBuilder sb = new StringBuilder();
                    sb.append(clusterNum).append(',');
                    sb.append(csvField(dataset.id(point)));
                    for (int d = 0; d < dataset.dimensions(); d++) {
                        sb.append(',').append(dataset.feature(point, d));
                    }
                    sb.append('\n');
                    writer.write(sb.toString());
                }
            }
        }
    }

    // Escapes a value for RFC 4180 CSV output (quote-doubling plus quoting)
    // and defuses spreadsheet formula injection: a value starting with =,
    // +, -, or @ gets a leading single quote so the cell is treated as
    // text rather than a formula.
    private static String csvField(String value) {
        if (!value.isEmpty() && "=+-@".indexOf(value.charAt(0)) >= 0) {
            value = "'" + value;
        }
        if (value.indexOf('"') >= 0 || value.indexOf(',') >= 0
                || value.indexOf('\n') >= 0 || value.indexOf('\r') >= 0) {
            return '"' + value.replace("\"", "\"\"") + '"';
        }
        return value;
    }
}
