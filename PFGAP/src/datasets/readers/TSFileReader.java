package datasets.readers;

import ch.randelshofer.fastdoubleparser.JavaDoubleParser;
import ch.randelshofer.fastdoubleparser.JavaFloatParser;
import core.AppContext;
import datasets.ListObjectDataset;
import datasets.NumericStorageType;
import org.apache.commons.lang3.time.DurationFormatUtils;

import java.io.BufferedReader;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Locale;
import java.util.Objects;
import java.util.Set;

/**
 * Reader for UEA/UCR-style .ts time-series files.
 *
 * <p>The reader supports univariate and multivariate observations, equal and
 * unequal lengths, internally ragged multivariate observations, optional
 * timestamp tuples, embedded labels, and separate label files.</p>
 *
 * <p>Numeric observations always use primitive arrays:</p>
 * <ul>
 *     <li>FLOAT32: {@code float[]} or {@code float[][]}</li>
 *     <li>FLOAT64: {@code double[]} or {@code double[][]}</li>
 * </ul>
 * Missing numeric values use primitive NaN. Generic observations use
 * {@code Object[]} or {@code Object[][]}, with missing values represented by
 * {@code null}. Boxed numeric arrays are never used as numeric dataset storage.
 * AUTO resolves to FLOAT64 because .ts is an untyped text format.</p>
 *
 * <p>Timestamped values of the form {@code (timestamp,value)} are supported.
 * Timestamp coordinates are parsed structurally but are not retained; source
 * tuple order determines observation order.</p>
 *
 * <p>If {@code @classLabel true} or {@code @targetLabel true} is declared,
 * the final top-level colon-separated field is read as the label or target.
 * A separate label file may be used instead, but the two mechanisms cannot be
 * combined.</p>
 */
public class TSFileReader implements DatasetReader {
    private static final char DIMENSION_SEPARATOR = ':';
    private static final char VALUE_SEPARATOR = ',';

    private final String dataFileName;
    private final String labelFileName;
    private final boolean isNumeric;
    private final boolean hasMissingValues;
    private final boolean isRegression;
    private final NumericStorageType numericStorageType;
    private final Set<String> missingIndicators;

    public TSFileReader(
            String dataFileName,
            boolean isNumeric,
            boolean hasMissingValues,
            boolean isRegression
    ) {
        this(dataFileName, null, isNumeric, hasMissingValues,
                isRegression, NumericStorageType.AUTO);
    }

    public TSFileReader(
            String dataFileName,
            String labelFileName,
            boolean isNumeric,
            boolean hasMissingValues,
            boolean isRegression
    ) {
        this(dataFileName, labelFileName, isNumeric, hasMissingValues,
                isRegression, NumericStorageType.AUTO);
    }

    public TSFileReader(
            String dataFileName,
            String labelFileName,
            boolean isNumeric,
            boolean hasMissingValues,
            boolean isRegression,
            NumericStorageType numericStorageType
    ) {
        this.dataFileName = requireNonblank(dataFileName, "dataFileName");
        this.labelFileName = normalizeNullableString(labelFileName);
        this.isNumeric = isNumeric;
        this.hasMissingValues = hasMissingValues;
        this.isRegression = isRegression;
        NumericStorageType requested = Objects.requireNonNull(
                numericStorageType, "numericStorageType cannot be null.");
        this.numericStorageType = requested == NumericStorageType.AUTO
                ? NumericStorageType.FLOAT64 : requested;
        this.missingIndicators = snapshotMissingIndicators();
    }

    public TSFileReader(ReaderOptions options) {
        this(requireOptions(options).getDataPath(),
                options.getLabelPath(), options.isNumeric(),
                options.hasMissingValues(), options.isRegression(),
                options.getNumericStorageType());
    }

    @Override
    public ListObjectDataset read() throws IOException {
        Path dataPath = validateFile(dataFileName, "TS data");
        TSMetadata metadata = inspectMetadata(dataPath);
        if (!metadata.hasDataSection) {
            throw new IOException("TS file does not contain an @data section: "
                    + dataPath);
        }
        if (metadata.hasEmbeddedLabels && labelFileName != null) {
            throw new IOException(
                    "Ambiguous label source: the TS file declares embedded "
                            + "labels or targets, but a separate label file "
                            + "was also supplied.");
        }

        List<Object> separateLabels = labelFileName == null
                ? List.of()
                : DelimitedFileReader.readGenericLabels(
                        labelFileName, false, isRegression);
        ListObjectDataset dataset = new ListObjectDataset();
        int instanceIndex = 0;
        int commonLength = -1;
        boolean unequalOrRagged = false;
        boolean inDataSection = false;
        long start = System.nanoTime();

        try (BufferedReader reader = Files.newBufferedReader(
                dataPath, StandardCharsets.UTF_8)) {
            String line;
            while ((line = reader.readLine()) != null) {
                String trimmed = line.trim();
                if (trimmed.isEmpty() || trimmed.startsWith("#")) {
                    continue;
                }
                if (!inDataSection) {
                    if (trimmed.equalsIgnoreCase("@data")) {
                        inDataSection = true;
                    }
                    continue;
                }

                ParsedTSRow parsed = parseTSRow(
                        trimmed, metadata, separateLabels,
                        instanceIndex, dataPath);
                dataset.add(parsed.label, parsed.data, instanceIndex);

                ShapeSummary shape = summarizeShape(parsed.data);
                if (shape.ragged) {
                    unequalOrRagged = true;
                }
                if (commonLength < 0) {
                    commonLength = shape.length;
                } else if (shape.length != commonLength) {
                    unequalOrRagged = true;
                }
                ProgressLogger.logProgress(instanceIndex);
                instanceIndex++;
            }
        }

        if (instanceIndex == 0) {
            throw new IOException(
                    "TS file contains no data observations: " + dataPath);
        }
        if (labelFileName != null
                && separateLabels.size() != instanceIndex) {
            throw new IOException(
                    "Separate label count " + separateLabels.size()
                            + " does not match TS observation count "
                            + instanceIndex + ".");
        }

        int datasetLength = unequalOrRagged
                ? 0 : Math.max(commonLength, 0);
        dataset.setLength(datasetLength);
        AppContext.length = datasetLength;
        ProgressLogger.logDuration(start, System.nanoTime());
        return dataset;
    }

    private ParsedTSRow parseTSRow(
            String line,
            TSMetadata metadata,
            List<Object> separateLabels,
            int rowIndex,
            Path dataPath
    ) throws IOException {
        List<String> parts = splitTopLevel(
                line, DIMENSION_SEPARATOR);
        if (parts.isEmpty()) {
            throw new IOException("Invalid empty TS data row at observation "
                    + rowIndex + " in " + dataPath + ".");
        }

        Object label = null;
        int dimensionCount = parts.size();
        if (metadata.hasEmbeddedLabels) {
            if (parts.size() < 2) {
                throw new IOException(
                        "Embedded label or target expected at observation "
                                + rowIndex + " in " + dataPath + ".");
            }
            label = parseLabel(parts.get(parts.size() - 1));
            dimensionCount--;
        } else if (labelFileName != null) {
            if (rowIndex >= separateLabels.size()) {
                throw new IOException(
                        "Separate label file contains fewer labels than TS "
                                + "observations; missing label for observation "
                                + rowIndex + ".");
            }
            label = separateLabels.get(rowIndex);
        }

        if (metadata.isUnivariate && dimensionCount != 1) {
            throw new IOException(
                    "TS metadata declares @univariate true, but observation "
                            + rowIndex + " contains " + dimensionCount
                            + " dimensions.");
        }
        if (dimensionCount == 1) {
            return new ParsedTSRow(label,
                    parseUnivariateDimension(parts.get(0), rowIndex));
        }
        return new ParsedTSRow(label,
                parseMultivariateDimensions(parts, dimensionCount, rowIndex));
    }

    private Object parseUnivariateDimension(
            String dimension,
            int rowIndex
    ) throws IOException {
        List<String> tokens = splitValues(dimension);
        if (isNumeric) {
            if (numericStorageType == NumericStorageType.FLOAT32) {
                float[] values = new float[tokens.size()];
                for (int i = 0; i < values.length; i++) {
                    values[i] = parseFloatToken(
                            extractValue(tokens.get(i)), rowIndex, 0, i);
                }
                return values;
            }
            double[] values = new double[tokens.size()];
            for (int i = 0; i < values.length; i++) {
                values[i] = parseDoubleToken(
                        extractValue(tokens.get(i)), rowIndex, 0, i);
            }
            return values;
        }

        Object[] values = new Object[tokens.size()];
        for (int i = 0; i < values.length; i++) {
            values[i] = parseGenericToken(
                    extractValue(tokens.get(i)), rowIndex, 0, i);
        }
        return values;
    }

    private Object parseMultivariateDimensions(
            List<String> parts,
            int dimensionCount,
            int rowIndex
    ) throws IOException {
        if (isNumeric) {
            if (numericStorageType == NumericStorageType.FLOAT32) {
                float[][] matrix = new float[dimensionCount][];
                for (int d = 0; d < dimensionCount; d++) {
                    List<String> tokens = splitValues(parts.get(d));
                    matrix[d] = new float[tokens.size()];
                    for (int t = 0; t < tokens.size(); t++) {
                        matrix[d][t] = parseFloatToken(
                                extractValue(tokens.get(t)),
                                rowIndex, d, t);
                    }
                }
                return matrix;
            }
            double[][] matrix = new double[dimensionCount][];
            for (int d = 0; d < dimensionCount; d++) {
                List<String> tokens = splitValues(parts.get(d));
                matrix[d] = new double[tokens.size()];
                for (int t = 0; t < tokens.size(); t++) {
                    matrix[d][t] = parseDoubleToken(
                            extractValue(tokens.get(t)),
                            rowIndex, d, t);
                }
            }
            return matrix;
        }

        Object[][] matrix = new Object[dimensionCount][];
        for (int d = 0; d < dimensionCount; d++) {
            List<String> tokens = splitValues(parts.get(d));
            matrix[d] = new Object[tokens.size()];
            for (int t = 0; t < tokens.size(); t++) {
                matrix[d][t] = parseGenericToken(
                        extractValue(tokens.get(t)), rowIndex, d, t);
            }
        }
        return matrix;
    }

    private float parseFloatToken(
            String token,
            int row,
            int dimension,
            int position
    ) throws IOException {
        String value = normalizeToken(token);
        if (isMissing(value)) {
            if (hasMissingValues) {
                return Float.NaN;
            }
            throw missingValueError(row, dimension, position);
        }
        try {
            return JavaFloatParser.parseFloat(value);
        } catch (NumberFormatException e) {
            throw numericValueError(value, row, dimension, position, e);
        }
    }

    private double parseDoubleToken(
            String token,
            int row,
            int dimension,
            int position
    ) throws IOException {
        String value = normalizeToken(token);
        if (isMissing(value)) {
            if (hasMissingValues) {
                return Double.NaN;
            }
            throw missingValueError(row, dimension, position);
        }
        try {
            return JavaDoubleParser.parseDouble(value);
        } catch (NumberFormatException e) {
            throw numericValueError(value, row, dimension, position, e);
        }
    }

    private Object parseGenericToken(
            String token,
            int row,
            int dimension,
            int position
    ) throws IOException {
        String value = normalizeToken(token);
        if (isMissing(value)) {
            if (hasMissingValues) {
                return null;
            }
            throw missingValueError(row, dimension, position);
        }
        try {
            return JavaDoubleParser.parseDouble(value);
        } catch (NumberFormatException ignored) {
            if (value.equalsIgnoreCase("true")
                    || value.equalsIgnoreCase("false")) {
                return Boolean.parseBoolean(value);
            }
            return value;
        }
    }

    private Object parseLabel(String token) throws IOException {
        String value = normalizeToken(token);
        if (isMissing(value)) {
            throw new IOException(
                    "TS labels and targets cannot be missing.");
        }
        if (isRegression) {
            try {
                return JavaDoubleParser.parseDouble(value);
            } catch (NumberFormatException e) {
                throw new IOException(
                        "Could not parse regression target: " + value, e);
            }
        }
        try {
            return Integer.parseInt(value);
        } catch (NumberFormatException ignored) {
            try {
                double numeric = JavaDoubleParser.parseDouble(value);
                if (numeric == Math.rint(numeric)
                        && numeric >= Integer.MIN_VALUE
                        && numeric <= Integer.MAX_VALUE) {
                    return (int) numeric;
                }
                return numeric;
            } catch (NumberFormatException ignoredAgain) {
                return value;
            }
        }
    }

    /** Splits on a separator only when it occurs outside parentheses. */
    private static List<String> splitTopLevel(
            String value,
            char separator
    ) throws IOException {
        List<String> parts = new ArrayList<>();
        int depth = 0;
        int start = 0;
        for (int i = 0; i < value.length(); i++) {
            char current = value.charAt(i);
            if (current == '(') {
                depth++;
            } else if (current == ')') {
                depth--;
                if (depth < 0) {
                    throw new IOException(
                            "Unbalanced parentheses in TS data row.");
                }
            } else if (current == separator && depth == 0) {
                parts.add(value.substring(start, i).trim());
                start = i + 1;
            }
        }
        if (depth != 0) {
            throw new IOException(
                    "Unbalanced parentheses in TS data row.");
        }
        parts.add(value.substring(start).trim());
        return parts;
    }

    private static List<String> splitValues(String dimension)
            throws IOException {
        if (dimension == null || dimension.trim().isEmpty()) {
            return List.of();
        }
        return splitTopLevel(dimension, VALUE_SEPARATOR);
    }

    /** Returns the value component of a raw value or timestamp tuple. */
    private static String extractValue(String token) throws IOException {
        if (token == null) {
            return null;
        }
        String trimmed = token.trim();
        if (!trimmed.startsWith("(")) {
            return trimmed;
        }
        if (!trimmed.endsWith(")")) {
            throw new IOException("Malformed TS timestamp tuple: " + token);
        }
        String inner = trimmed.substring(1, trimmed.length() - 1);
        List<String> pair = splitTopLevel(inner, VALUE_SEPARATOR);
        if (pair.size() != 2) {
            throw new IOException(
                    "TS timestamp tuple must contain timestamp and value: "
                            + token);
        }
        return pair.get(1).trim();
    }

    private boolean isMissing(String token) {
        return token == null || token.isEmpty()
                || token.equals("?")
                || missingIndicators.contains(
                        token.toUpperCase(Locale.ROOT));
    }

    private static String normalizeToken(String token) {
        return token == null ? "" : token.trim();
    }

    private static IOException missingValueError(
            int row,
            int dimension,
            int position
    ) {
        return new IOException(
                "Missing TS value at observation " + row
                        + ", dimension " + dimension
                        + ", position " + position
                        + " while hasMissingValues=false.");
    }

    private static IOException numericValueError(
            String value,
            int row,
            int dimension,
            int position,
            Exception cause
    ) {
        return new IOException(
                "Could not parse TS numeric value '" + value
                        + "' at observation " + row
                        + ", dimension " + dimension
                        + ", position " + position + ".", cause);
    }

    private static ShapeSummary summarizeShape(Object data) {
        if (data instanceof double[] values) {
            return new ShapeSummary(values.length, false);
        }
        if (data instanceof float[] values) {
            return new ShapeSummary(values.length, false);
        }
        if (data instanceof Object[] values
                && !(data instanceof Object[][])) {
            return new ShapeSummary(values.length, false);
        }
        if (data instanceof double[][] matrix) {
            return summarizeMatrix(matrix);
        }
        if (data instanceof float[][] matrix) {
            return summarizeMatrix(matrix);
        }
        if (data instanceof Object[][] matrix) {
            return summarizeMatrix(matrix);
        }
        throw new IllegalArgumentException(
                "Unsupported TS observation representation: "
                        + data.getClass().getName());
    }

    private static ShapeSummary summarizeMatrix(double[][] matrix) {
        if (matrix.length == 0) {
            return new ShapeSummary(0, false);
        }
        int length = matrix[0].length;
        for (double[] dimension : matrix) {
            if (dimension.length != length) {
                return new ShapeSummary(length, true);
            }
        }
        return new ShapeSummary(length, false);
    }

    private static ShapeSummary summarizeMatrix(float[][] matrix) {
        if (matrix.length == 0) {
            return new ShapeSummary(0, false);
        }
        int length = matrix[0].length;
        for (float[] dimension : matrix) {
            if (dimension.length != length) {
                return new ShapeSummary(length, true);
            }
        }
        return new ShapeSummary(length, false);
    }

    private static ShapeSummary summarizeMatrix(Object[][] matrix) {
        if (matrix.length == 0) {
            return new ShapeSummary(0, false);
        }
        int length = matrix[0].length;
        for (Object[] dimension : matrix) {
            if (dimension.length != length) {
                return new ShapeSummary(length, true);
            }
        }
        return new ShapeSummary(length, false);
    }

    private static TSMetadata inspectMetadata(Path dataPath)
            throws IOException {
        TSMetadata metadata = new TSMetadata();
        try (BufferedReader reader = Files.newBufferedReader(
                dataPath, StandardCharsets.UTF_8)) {
            String line;
            while ((line = reader.readLine()) != null) {
                String trimmed = line.trim();
                if (trimmed.isEmpty() || trimmed.startsWith("#")) {
                    continue;
                }
                String lower = trimmed.toLowerCase(Locale.ROOT);
                if (lower.equals("@data")) {
                    metadata.hasDataSection = true;
                    break;
                }
                if (lower.startsWith("@classlabel")
                        || lower.startsWith("@targetlabel")) {
                    metadata.hasEmbeddedLabels =
                            parseBooleanMetadataLine(trimmed);
                } else if (lower.startsWith("@univariate")) {
                    metadata.isUnivariate =
                            parseBooleanMetadataLine(trimmed);
                } else if (lower.startsWith("@timestamps")) {
                    metadata.hasTimestamps =
                            parseBooleanMetadataLine(trimmed);
                }
            }
        }
        return metadata;
    }

    private static boolean parseBooleanMetadataLine(String line) {
        String[] tokens = line.trim().split("\\s+");
        return tokens.length >= 2 && Boolean.parseBoolean(tokens[1]);
    }

    private static Path validateFile(String value, String role)
            throws IOException {
        Path path = Path.of(requireNonblank(value, role));
        if (!Files.isRegularFile(path) || !Files.isReadable(path)) {
            throw new IOException(role
                    + " file is not a readable regular file: " + path);
        }
        return path;
    }

    private static ReaderOptions requireOptions(ReaderOptions options) {
        if (options == null) {
            throw new IllegalArgumentException(
                    "TSFileReader requires non-null ReaderOptions.");
        }
        return options;
    }

    private static String requireNonblank(String value, String role) {
        if (value == null || value.isBlank()) {
            throw new IllegalArgumentException(role + " cannot be blank.");
        }
        return value.trim();
    }

    private static String normalizeNullableString(String value) {
        if (value == null) {
            return null;
        }
        String trimmed = value.trim();
        return trimmed.isEmpty() || trimmed.equalsIgnoreCase("None")
                ? null : trimmed;
    }

    private static Set<String> snapshotMissingIndicators() {
        if (AppContext.MissingStrings == null
                || AppContext.MissingStrings.isEmpty()) {
            return Set.of();
        }
        Set<String> values = new HashSet<>();
        for (String indicator : AppContext.MissingStrings) {
            if (indicator != null && !indicator.isBlank()) {
                values.add(indicator.trim().toUpperCase(Locale.ROOT));
            }
        }
        return values.isEmpty()
                ? Set.of() : Collections.unmodifiableSet(values);
    }

    private static final class TSMetadata {
        private boolean hasEmbeddedLabels;
        private boolean isUnivariate;
        private boolean hasTimestamps;
        private boolean hasDataSection;
    }

    private record ParsedTSRow(Object label, Object data) {
    }

    private record ShapeSummary(int length, boolean ragged) {
    }

    public static class ProgressLogger {
        public static void logProgress(int index) {
            if (index % 1000 != 0) {
                return;
            }
            if (index % 100000 == 0) {
                System.out.print("\n");
                if (index % 1000000 == 0) {
                    long used = AppContext.runtime.totalMemory()
                            - AppContext.runtime.freeMemory();
                    System.out.print(index + ":" + used / 1024 / 1024
                            + "mb\n");
                }
            } else {
                System.out.print(".");
            }
        }

        public static void logDuration(long start, long end) {
            String duration = DurationFormatUtils.formatDuration(
                    (long) ((end - start) / 1e6), "H:m:s.SSS");
            System.out.println("finished in " + duration);
        }
    }
}
