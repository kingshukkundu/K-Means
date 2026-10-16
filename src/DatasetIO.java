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
import java.io.InputStreamReader;
import java.math.BigDecimal;
import java.nio.charset.CharacterCodingException;
import java.nio.charset.CharsetDecoder;
import java.nio.charset.CodingErrorAction;
import java.nio.charset.StandardCharsets;
import java.nio.file.AtomicMoveNotSupportedException;
import java.nio.file.FileAlreadyExistsException;
import java.nio.file.Files;
import java.nio.file.LinkOption;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.util.ArrayList;
import java.util.List;
import java.util.UUID;

/**
 * CSV serialization for {@link Dataset}: reads the plain comma-separated
 * input format and writes cluster-grouped results with RFC 4180 escaping.
 */
final class DatasetIO {

    private DatasetIO() {
    }

    static Dataset readDataset(Path path) throws IOException {
        List<String> ids = new ArrayList<>();
        List<double[]> rows = new ArrayList<>();
        List<Integer> lineNumbers = new ArrayList<>();
        int expectedFeatureCount = -1;
        int lineNumber = 0;

        // Strict decoding: corrupt bytes must fail, not become U+FFFD in ids.
        CharsetDecoder decoder = StandardCharsets.UTF_8.newDecoder()
            .onMalformedInput(CodingErrorAction.REPORT)
            .onUnmappableCharacter(CodingErrorAction.REPORT);
        try (BufferedReader reader = new BufferedReader(
                new InputStreamReader(Files.newInputStream(path), decoder))) {
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

                double[] features = new double[expectedFeatureCount];
                for (int i = 1; i < tokens.length; i++) {
                    try {
                        // BigDecimal rejects hex floats, NaN/Infinity, and
                        // d/f suffixes that Double.parseDouble would accept.
                        features[i - 1] = new BigDecimal(tokens[i].trim()).doubleValue();
                    } catch (NumberFormatException e) {
                        throw new IllegalArgumentException(
                            "Line " + lineNumber + ": Invalid numeric value"
                                + (firstDataRow ? ". If the file has a header row, remove it first." : ""), e);
                    }
                }

                ids.add(tokens[0]);
                rows.add(features);
                lineNumbers.add(lineNumber);
            }
        } catch (CharacterCodingException e) {
            throw new IOException("File is not valid UTF-8: " + path, e);
        }

        if (lineNumber == 0) {
            throw new IllegalArgumentException("CSV file is empty.");
        }
        if (ids.isEmpty()) {
            throw new IllegalArgumentException("No valid data rows found in CSV file.");
        }

        // Dataset.of enforces all data invariants in one place; the source
        // line of each row lets its errors cite real file lines.
        return Dataset.of(ids, rows,
            lineNumbers.stream().mapToInt(Integer::intValue).toArray());
    }

    // Writes via a sibling temp file moved into place once complete, so a
    // mid-write failure never leaves a truncated file.
    static void writeClusters(Path path, Dataset dataset, List<List<Integer>> clusters,
                              boolean overwrite) throws IOException {
        // Early check so the common rerun case fails before serialization;
        // moveIntoPlace still performs the real atomic create-if-absent.
        if (!overwrite && Files.exists(path, LinkOption.NOFOLLOW_LINKS)) {
            throw new FileAlreadyExistsException(path.toString());
        }
        Path tmp = createSiblingTempFile(path);
        try {
            writeRows(tmp, dataset, clusters);
            moveIntoPlace(tmp, path, overwrite);
        } finally {
            try {
                Files.deleteIfExists(tmp);
            } catch (IOException | SecurityException ignored) {
                // A failed cleanup must not mask the real error.
            }
        }
    }

    private static void writeRows(Path tmp, Dataset dataset, List<List<Integer>> clusters)
            throws IOException {
        try (BufferedWriter writer = Files.newBufferedWriter(tmp, StandardCharsets.UTF_8)) {
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

    // createFile (not createTempFile) keeps normal umask permissions, and
    // same-directory placement keeps the move on one filesystem.
    private static Path createSiblingTempFile(Path target) throws IOException {
        // Both are null when the target is a filesystem root.
        Path dir = target.toAbsolutePath().getParent();
        Path fileName = target.getFileName();
        if (dir == null || fileName == null) {
            throw new IllegalArgumentException(
                "Output path must name a file inside a directory: " + target);
        }
        String prefix = "." + fileName + ".";
        for (int attempt = 0; attempt < 100; attempt++) {
            Path candidate = dir.resolve(prefix + UUID.randomUUID() + ".tmp");
            try {
                return Files.createFile(candidate);
            } catch (FileAlreadyExistsException e) {
                // Generated name collided — try another.
            }
        }
        throw new IOException("Unable to create a temporary file in " + dir);
    }

    // Commits the temp file under the final name; overwrite=false needs an
    // atomic create-if-absent (hard link, else a check-then-move fallback).
    private static void moveIntoPlace(Path tmp, Path target, boolean overwrite)
            throws IOException {
        if (!overwrite) {
            if (tryLinkIntoPlace(tmp, target)) {
                return;
            }
            if (Files.exists(target, LinkOption.NOFOLLOW_LINKS)) {
                throw new FileAlreadyExistsException(target.toString());
            }
            Files.move(tmp, target);
            return;
        }
        try {
            Files.move(tmp, target, StandardCopyOption.ATOMIC_MOVE,
                    StandardCopyOption.REPLACE_EXISTING);
        } catch (AtomicMoveNotSupportedException | UnsupportedOperationException e) {
            // Fall back to a plain move when atomic rename is unsupported.
            Files.move(tmp, target, StandardCopyOption.REPLACE_EXISTING);
        }
    }

    // Files.createLink is the only atomic create-if-absent primitive the
    // filesystem API offers. Returns false when hard links are unsupported.
    private static boolean tryLinkIntoPlace(Path tmp, Path target) throws IOException {
        try {
            Files.createLink(target, tmp);
        } catch (FileAlreadyExistsException e) {
            throw e;
        } catch (IOException | UnsupportedOperationException | SecurityException e) {
            return false;
        }
        // The output is already committed, so delete failure is not a
        // write failure; the finally in writeClusters retries cleanup.
        try {
            Files.delete(tmp);
        } catch (IOException | SecurityException ignored) {
        }
        return true;
    }

    // RFC 4180 escaping plus formula-injection defusal: a value starting
    // with =, +, -, or @ (after whitespace) gets a leading single quote.
    private static String csvField(String value) {
        int first = 0;
        while (first < value.length() && Character.isWhitespace(value.charAt(first))) {
            first++;
        }
        if (first < value.length() && "=+-@".indexOf(value.charAt(first)) >= 0) {
            value = "'" + value;
        }
        if (value.indexOf('"') >= 0 || value.indexOf(',') >= 0
                || value.indexOf('\n') >= 0 || value.indexOf('\r') >= 0) {
            return '"' + value.replace("\"", "\"\"") + '"';
        }
        return value;
    }
}
