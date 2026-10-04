package datasets.readers;

import ch.randelshofer.fastdoubleparser.JavaDoubleParser;
import ch.randelshofer.fastdoubleparser.JavaFloatParser;
import core.AppContext;
import datasets.ListObjectDataset;
import datasets.NumericStorageType;
import de.siegmar.fastcsv.reader.CsvReader;
import de.siegmar.fastcsv.reader.CsvRecord;
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
 * General reader for delimited one-dimensional observations and row-encoded
 * two-dimensional observations.
 *
 * <p>One-dimensional observations may be tabular records, univariate time
 * series, or other ordered feature vectors. Two-dimensional observations are
 * encoded with {@code arraySeparator} between dimensions and
 * {@code entrySeparator} within each dimension.</p>
 *
 * <p>Numeric output is always primitive:</p>
 * <ul>
 *     <li>FLOAT64: {@code double[]} or {@code double[][]}</li>
 *     <li>FLOAT32: {@code float[]} or {@code float[][]}</li>
 *     <li>Generic: {@code Object[]} or {@code Object[][]}</li>
 * </ul>
 *
 * <p>Embedded labels are supported only for one-dimensional observations.
 * Row-encoded two-dimensional observations require a separate label file or
 * no labels. Missing numeric values are represented by primitive NaN values;
 * missing generic values are represented by {@code null}.</p>
 */
public class DelimitedFileReader implements DatasetReader {

    private final String dataFileName;
    private final String labelFileName;
    private final String entrySeparator;
    private final String arraySeparator;
    private final char outerSeparator;
    private final char innerSeparator;
    private final boolean hasHeader;
    private final boolean is2D;
    private final boolean isNumeric;
    private final boolean hasMissingValues;
    private final boolean targetColumnIsFirst;
    private final boolean isTest;
    private final boolean isRegression;
    private final NumericStorageType numericStorageType;

    public DelimitedFileReader(
            String dataFileName,
            String labelFileName,
            String entrySeparator,
            String arraySeparator,
            boolean hasHeader,
            boolean is2D,
            boolean isNumeric,
            boolean hasMissingValues,
            boolean targetColumnIsFirst,
            boolean isTest,
            boolean isRegression
    ) {
        this(
                dataFileName, labelFileName, entrySeparator, arraySeparator,
                hasHeader, is2D, isNumeric, hasMissingValues,
                targetColumnIsFirst, isTest, isRegression,
                NumericStorageType.AUTO
        );
    }

    public DelimitedFileReader(
            String dataFileName,
            String labelFileName,
            String entrySeparator,
            String arraySeparator,
            boolean hasHeader,
            boolean is2D,
            boolean isNumeric,
            boolean hasMissingValues,
            boolean targetColumnIsFirst,
            boolean isTest,
            boolean isRegression,
            NumericStorageType numericStorageType
    ) {
        this.dataFileName = requireNonblank(dataFileName, "dataFileName");
        this.labelFileName = normalizeNullableString(labelFileName);
        this.entrySeparator = normalizeSeparator(entrySeparator);
        this.arraySeparator = normalizeSeparator(arraySeparator);
        this.hasHeader = hasHeader;
        this.is2D = is2D;
        this.isNumeric = isNumeric;
        this.hasMissingValues = hasMissingValues;
        this.targetColumnIsFirst = targetColumnIsFirst;
        this.isTest = isTest;
        this.isRegression = isRegression;
        this.numericStorageType = resolveStorageType(numericStorageType);
        this.innerSeparator = requireSingleCharacterSeparator(
                this.entrySeparator, "entrySeparator");
        this.outerSeparator = is2D
                ? requireSingleCharacterSeparator(this.arraySeparator, "arraySeparator")
                : this.innerSeparator;

        if (is2D && shouldParseEmbeddedLabel()) {
            throw new IllegalArgumentException(
                    "DelimitedFileReader does not support embedded labels for "
                            + "row-encoded 2D observations. Supply a separate "
                            + "label file or configure unlabeled input.");
        }
    }

    @Override
    public ListObjectDataset read() throws IOException {
        long start = System.nanoTime();
        List<Object> labels = labelFileName == null
                ? List.of()
                : readGenericLabels(labelFileName, hasHeader, isRegression);

        Path dataPath = Path.of(dataFileName);
        validateDataFile(dataPath);

        ListObjectDataset dataset = new ListObjectDataset();
        int instanceIndex = 0;
        int commonLength = -1;
        boolean unequalLengths = false;

        try (CsvReader<CsvRecord> csvReader = CsvReader.builder()
                .fieldSeparator(outerSeparator)
                .skipEmptyLines(false)
                .detectBomHeader(true)
                .ofCsvRecord(dataPath)) {
            var iterator = csvReader.iterator();
            if (hasHeader) {
                if (!iterator.hasNext()) {
                    throw new IOException(
                            "Delimited file contains no header or data: " + dataPath);
                }
                iterator.next();
            }

            while (iterator.hasNext()) {
                CsvRecord record = iterator.next();
                List<String> fields = record.getFields();
                if (fields.isEmpty()) {
                    throw new IllegalArgumentException(
                            "Encountered a delimited record with no fields in file "
                                    + dataPath + ".");
                }

                ParsedInstance parsed = parseRecord(
                        fields, labels, instanceIndex, dataPath);
                dataset.add(parsed.label, parsed.data, instanceIndex);

                if (commonLength < 0) {
                    commonLength = parsed.length;
                } else if (parsed.length != commonLength) {
                    unequalLengths = true;
                }

                ProgressLogger.logProgress(instanceIndex);
                instanceIndex++;
            }
        } catch (IllegalArgumentException e) {
            throw e;
        } catch (RuntimeException e) {
            throw new IOException(
                    "Failed while parsing delimited dataset: " + dataPath, e);
        }

        if (instanceIndex == 0) {
            throw new IOException(
                    "Delimited dataset contains no data records: " + dataPath);
        }
        validateSeparateLabelCount(labels, instanceIndex);

        int datasetLength = unequalLengths ? 0 : Math.max(commonLength, 0);
        dataset.setLength(datasetLength);
        AppContext.length = datasetLength;

        ProgressLogger.logDuration(start, System.nanoTime());
        return dataset;
    }

    private ParsedInstance parseRecord(
            List<String> fields,
            List<Object> labels,
            int instanceIndex,
            Path dataPath
    ) {
        if (is2D) {
            Object label = getSeparateLabel(labels, instanceIndex);
            if (isNumeric) {
                Object data = parsePrimitiveNumericMatrix(
                        fields, dataPath, instanceIndex);
                return new ParsedInstance(
                        label, data, numericMatrixLength(data, instanceIndex));
            }
            Object[][] data = parseObjectMatrix(fields, dataPath, instanceIndex);
            return new ParsedInstance(
                    label, data, data.length == 0 ? 0 : data[0].length);
        }

        if (isNumeric) {
            return parsePrimitiveNumericVector(
                    fields, labels, instanceIndex, dataPath);
        }
        return parseObjectVector(fields, labels, instanceIndex, dataPath);
    }

    private ParsedInstance parsePrimitiveNumericVector(
            List<String> fields,
            List<Object> labels,
            int instanceIndex,
            Path dataPath
    ) {
        boolean embedded = shouldParseEmbeddedLabel();
        if (embedded) {
            validateEmbeddedLabelRecord(fields, dataPath, instanceIndex);
        }
        int labelIndex = embedded
                ? (targetColumnIsFirst ? 0 : fields.size() - 1)
                : -1;
        int dataLength = fields.size() - (embedded ? 1 : 0);
        Object label = embedded
                ? parseLabel(fields.get(labelIndex))
                : getSeparateLabel(labels, instanceIndex);

        if (numericStorageType == NumericStorageType.FLOAT32) {
            float[] data = new float[dataLength];
            int output = 0;
            for (int field = 0; field < fields.size(); field++) {
                if (field != labelIndex) {
                    data[output++] = parsePrimitiveFloatToken(
                            fields.get(field), dataPath, instanceIndex, field);
                }
            }
            return new ParsedInstance(label, data, data.length);
        }

        double[] data = new double[dataLength];
        int output = 0;
        for (int field = 0; field < fields.size(); field++) {
            if (field != labelIndex) {
                data[output++] = parsePrimitiveDoubleToken(
                        fields.get(field), dataPath, instanceIndex, field);
            }
        }
        return new ParsedInstance(label, data, data.length);
    }

    private ParsedInstance parseObjectVector(
            List<String> fields,
            List<Object> labels,
            int instanceIndex,
            Path dataPath
    ) {
        boolean embedded = shouldParseEmbeddedLabel();
        if (embedded) {
            validateEmbeddedLabelRecord(fields, dataPath, instanceIndex);
        }
        int labelIndex = embedded
                ? (targetColumnIsFirst ? 0 : fields.size() - 1)
                : -1;
        Object label = embedded
                ? parseLabel(fields.get(labelIndex))
                : getSeparateLabel(labels, instanceIndex);
        Object[] data = new Object[fields.size() - (embedded ? 1 : 0)];
        int output = 0;
        for (int field = 0; field < fields.size(); field++) {
            if (field == labelIndex) {
                continue;
            }
            data[output++] = parseGenericToken(
                    fields.get(field), dataPath, instanceIndex, field);
        }
        return new ParsedInstance(label, data, data.length);
    }

    private Object parsePrimitiveNumericMatrix(
            List<String> dimensions,
            Path dataPath,
            int instanceIndex
    ) {
        if (numericStorageType == NumericStorageType.FLOAT32) {
            float[][] data = new float[dimensions.size()][];
            int expectedLength = -1;
            for (int dimension = 0; dimension < dimensions.size(); dimension++) {
                String[] tokens = splitLiteral(
                        dimensions.get(dimension), innerSeparator);
                float[] values = new float[tokens.length];
                for (int time = 0; time < tokens.length; time++) {
                    values[time] = parsePrimitiveFloatToken(
                            tokens[time], dataPath, instanceIndex,
                            dimension, time);
                }
                expectedLength = validateDimensionLength(
                        expectedLength, values.length, dataPath,
                        instanceIndex, dimension);
                data[dimension] = values;
            }
            return data;
        }

        double[][] data = new double[dimensions.size()][];
        int expectedLength = -1;
        for (int dimension = 0; dimension < dimensions.size(); dimension++) {
            String[] tokens = splitLiteral(
                    dimensions.get(dimension), innerSeparator);
            double[] values = new double[tokens.length];
            for (int time = 0; time < tokens.length; time++) {
                values[time] = parsePrimitiveDoubleToken(
                        tokens[time], dataPath, instanceIndex,
                        dimension, time);
            }
            expectedLength = validateDimensionLength(
                    expectedLength, values.length, dataPath,
                    instanceIndex, dimension);
            data[dimension] = values;
        }
        return data;
    }

    private Object[][] parseObjectMatrix(
            List<String> dimensions,
            Path dataPath,
            int instanceIndex
    ) {
        Object[][] data = new Object[dimensions.size()][];
        int expectedLength = -1;
        for (int dimension = 0; dimension < dimensions.size(); dimension++) {
            String[] tokens = splitLiteral(
                    dimensions.get(dimension), innerSeparator);
            Object[] values = new Object[tokens.length];
            for (int time = 0; time < tokens.length; time++) {
                values[time] = parseGenericToken(
                        tokens[time], dataPath, instanceIndex,
                        dimension, time);
            }
            expectedLength = validateDimensionLength(
                    expectedLength, values.length, dataPath,
                    instanceIndex, dimension);
            data[dimension] = values;
        }
        return data;
    }

    private double parsePrimitiveDoubleToken(
            String token,
            Path dataPath,
            int instanceIndex,
            int fieldIndex
    ) {
        String value = token == null ? "" : token.trim();
        if (RowParser.isMissingToken(value)) {
            requireMissingValuesEnabled(
                    dataPath, instanceIndex, "field " + fieldIndex);
            return Double.NaN;
        }
        try {
            return JavaDoubleParser.parseDouble(value);
        } catch (NumberFormatException e) {
            throw numericParseFailure(
                    value, dataPath, instanceIndex, "field " + fieldIndex, e);
        }
    }

    private float parsePrimitiveFloatToken(
            String token,
            Path dataPath,
            int instanceIndex,
            int fieldIndex
    ) {
        String value = token == null ? "" : token.trim();
        if (RowParser.isMissingToken(value)) {
            requireMissingValuesEnabled(
                    dataPath, instanceIndex, "field " + fieldIndex);
            return Float.NaN;
        }
        try {
            return JavaFloatParser.parseFloat(value);
        } catch (NumberFormatException e) {
            throw numericParseFailure(
                    value, dataPath, instanceIndex, "field " + fieldIndex, e);
        }
    }

    private double parsePrimitiveDoubleToken(
            String token,
            Path dataPath,
            int instanceIndex,
            int dimension,
            int position
    ) {
        String value = token == null ? "" : token.trim();
        String location = "dimension " + dimension + ", position " + position;
        if (RowParser.isMissingToken(value)) {
            requireMissingValuesEnabled(dataPath, instanceIndex, location);
            return Double.NaN;
        }
        try {
            return JavaDoubleParser.parseDouble(value);
        } catch (NumberFormatException e) {
            throw numericParseFailure(
                    value, dataPath, instanceIndex, location, e);
        }
    }

    private float parsePrimitiveFloatToken(
            String token,
            Path dataPath,
            int instanceIndex,
            int dimension,
            int position
    ) {
        String value = token == null ? "" : token.trim();
        String location = "dimension " + dimension + ", position " + position;
        if (RowParser.isMissingToken(value)) {
            requireMissingValuesEnabled(dataPath, instanceIndex, location);
            return Float.NaN;
        }
        try {
            return JavaFloatParser.parseFloat(value);
        } catch (NumberFormatException e) {
            throw numericParseFailure(
                    value, dataPath, instanceIndex, location, e);
        }
    }

    private Object parseGenericToken(
            String token,
            Path dataPath,
            int instanceIndex,
            int fieldIndex
    ) {
        if (RowParser.isMissingToken(token)) {
            requireMissingValuesEnabled(
                    dataPath, instanceIndex, "field " + fieldIndex);
            return null;
        }
        return RowParser.parseNonMissingValue(token);
    }

    private Object parseGenericToken(
            String token,
            Path dataPath,
            int instanceIndex,
            int dimension,
            int position
    ) {
        if (RowParser.isMissingToken(token)) {
            requireMissingValuesEnabled(
                    dataPath, instanceIndex,
                    "dimension " + dimension + ", position " + position);
            return null;
        }
        return RowParser.parseNonMissingValue(token);
    }

    private void requireMissingValuesEnabled(
            Path dataPath,
            int instanceIndex,
            String location
    ) {
        if (!hasMissingValues) {
            throw new IllegalArgumentException(
                    "Missing value at instance " + instanceIndex + ", "
                            + location + " in " + dataPath
                            + ", but hasMissingValues=false.");
        }
    }

    private static IllegalArgumentException numericParseFailure(
            String value,
            Path dataPath,
            int instanceIndex,
            String location,
            NumberFormatException cause
    ) {
        return new IllegalArgumentException(
                "Could not parse numeric value '" + value + "' at instance "
                        + instanceIndex + ", " + location + " in " + dataPath
                        + ".", cause);
    }

    private static int numericMatrixLength(Object matrix, int instanceIndex) {
        if (matrix instanceof double[][] values) {
            if (values.length == 0) {
                throw emptyMatrix(instanceIndex);
            }
            return values[0].length;
        }
        if (matrix instanceof float[][] values) {
            if (values.length == 0) {
                throw emptyMatrix(instanceIndex);
            }
            return values[0].length;
        }
        throw new IllegalArgumentException(
                "Unsupported numeric matrix representation.");
    }

    private static IllegalArgumentException emptyMatrix(int instanceIndex) {
        return new IllegalArgumentException(
                "Numeric matrix has no dimensions at instance "
                        + instanceIndex + ".");
    }

    private boolean shouldParseEmbeddedLabel() {
        return labelFileName == null
                && !AppContext.isIsolationMode()
                && (!isTest || AppContext.exists_testlabels);
    }

    private void validateEmbeddedLabelRecord(
            List<String> fields,
            Path dataPath,
            int instanceIndex
    ) {
        if (fields.size() < 2) {
            throw new IllegalArgumentException(
                    "Delimited record " + instanceIndex + " in file "
                            + dataPath + " must contain at least one feature "
                            + "and one embedded label.");
        }
    }

    private Object parseLabel(String token) {
        if (isRegression) {
            return JavaDoubleParser.parseDouble(
                    requireNonemptyToken(token, "Embedded regression label"));
        }
        return RowParser.tryParseLabel(token);
    }

    private int validateDimensionLength(
            int expectedLength,
            int actualLength,
            Path dataPath,
            int instanceIndex,
            int dimension
    ) {
        if (expectedLength < 0) {
            return actualLength;
        }
        if (actualLength != expectedLength) {
            throw new IllegalArgumentException(
                    "Row-encoded 2D instance " + instanceIndex + " in file "
                            + dataPath + " has inconsistent dimension lengths. "
                            + "Expected " + expectedLength + " values but dimension "
                            + dimension + " contains " + actualLength + ".");
        }
        return expectedLength;
    }

    private Object getSeparateLabel(List<Object> labels, int index) {
        if (labelFileName == null) {
            return null;
        }
        if (index >= labels.size()) {
            throw new IllegalArgumentException(
                    "The separate label file contains fewer labels than the "
                            + "data file. Missing label for instance " + index + ".");
        }
        return labels.get(index);
    }

    private void validateSeparateLabelCount(
            List<Object> labels,
            int instanceCount
    ) {
        if (labelFileName != null && labels.size() != instanceCount) {
            throw new IllegalArgumentException(
                    "Separate label count does not match the number of data "
                            + "instances. Labels=" + labels.size()
                            + ", instances=" + instanceCount + ".");
        }
    }

    private static NumericStorageType resolveStorageType(
            NumericStorageType requested
    ) {
        NumericStorageType value = Objects.requireNonNull(
                requested, "NumericStorageType cannot be null.");
        return value == NumericStorageType.AUTO
                ? NumericStorageType.FLOAT64
                : value;
    }

    private static void validateDataFile(Path dataPath) throws IOException {
        if (!Files.exists(dataPath)) {
            throw new IOException(
                    "Delimited data file does not exist: " + dataPath);
        }
        if (!Files.isRegularFile(dataPath)) {
            throw new IOException(
                    "Delimited data path is not a regular file: " + dataPath);
        }
        if (!Files.isReadable(dataPath)) {
            throw new IOException(
                    "Delimited data file is not readable: " + dataPath);
        }
    }

    private static String[] splitLiteral(String value, char separator) {
        if (value == null) {
            return new String[]{null};
        }
        int fieldCount = 1;
        for (int index = 0; index < value.length(); index++) {
            if (value.charAt(index) == separator) {
                fieldCount++;
            }
        }
        String[] result = new String[fieldCount];
        int resultIndex = 0;
        int fieldStart = 0;
        for (int index = 0; index < value.length(); index++) {
            if (value.charAt(index) == separator) {
                result[resultIndex++] = value.substring(fieldStart, index);
                fieldStart = index + 1;
            }
        }
        result[resultIndex] = value.substring(fieldStart);
        return result;
    }

    private static String requireNonblank(String value, String argumentName) {
        if (value == null || value.isBlank()) {
            throw new IllegalArgumentException(
                    argumentName + " cannot be null or blank.");
        }
        return value.trim();
    }

    private static String requireNonemptyToken(String value, String role) {
        if (value == null || value.trim().isEmpty()) {
            throw new IllegalArgumentException(role + " cannot be missing.");
        }
        return value.trim();
    }

    private static String normalizeNullableString(String value) {
        if (value == null) {
            return null;
        }
        String trimmed = value.trim();
        return trimmed.isEmpty() || trimmed.equalsIgnoreCase("None")
                ? null
                : trimmed;
    }

    private static String normalizeSeparator(String separator) {
        if (separator == null || separator.isEmpty()) {
            return null;
        }
        return switch (separator) {
            case "\\t" -> "\t";
            case "\\n" -> "\n";
            case "\\r" -> "\r";
            default -> separator;
        };
    }

    private static char requireSingleCharacterSeparator(
            String separator,
            String argumentName
    ) {
        if (separator == null || separator.isEmpty()) {
            throw new IllegalArgumentException(
                    "DelimitedFileReader requires a non-empty "
                            + argumentName + ".");
        }
        if (separator.length() != 1) {
            throw new IllegalArgumentException(
                    "FastCSV-backed DelimitedFileReader requires "
                            + argumentName + " to contain exactly one character. "
                            + "Received: '" + separator + "'.");
        }
        char value = separator.charAt(0);
        if (value == '\n' || value == '\r') {
            throw new IllegalArgumentException(
                    argumentName + " cannot be a line-separator character.");
        }
        return value;
    }

    public static List<Integer> readLabels(
            String labelFileName,
            boolean hasHeader
    ) throws IOException {
        List<Integer> labels = new ArrayList<>();
        if (labelFileName == null) {
            return labels;
        }
        try (BufferedReader reader = Files.newBufferedReader(
                Path.of(labelFileName), StandardCharsets.UTF_8)) {
            if (hasHeader) {
                reader.readLine();
            }
            String line;
            while ((line = reader.readLine()) != null) {
                String trimmed = line.trim();
                if (trimmed.isEmpty()) {
                    throw new IllegalArgumentException(
                            "Encountered an empty integer label in file: "
                                    + labelFileName);
                }
                labels.add(Integer.parseInt(trimmed));
            }
        }
        return labels;
    }

    public static List<Object> readGenericLabels(
            String labelFileName,
            boolean hasHeader,
            boolean isRegression
    ) throws IOException {
        List<Object> labels = new ArrayList<>();
        if (labelFileName == null) {
            return labels;
        }
        try (BufferedReader reader = Files.newBufferedReader(
                Path.of(labelFileName), StandardCharsets.UTF_8)) {
            if (hasHeader) {
                reader.readLine();
            }
            String line;
            while ((line = reader.readLine()) != null) {
                String trimmed = line.trim();
                if (trimmed.isEmpty()) {
                    throw new IllegalArgumentException(
                            "Encountered an empty label in file: "
                                    + labelFileName);
                }
                labels.add(isRegression
                        ? JavaDoubleParser.parseDouble(trimmed)
                        : RowParser.tryParseLabel(trimmed));
            }
        }
        return labels;
    }

    public static class ParsedDoubleRow {
        public final Object label;
        public final double[] features;

        public ParsedDoubleRow(Object label, double[] features) {
            this.label = label;
            this.features = features;
        }
    }

    private static final class ParsedInstance {
        private final Object label;
        private final Object data;
        private final int length;

        private ParsedInstance(Object label, Object data, int length) {
            this.label = label;
            this.data = data;
            this.length = length;
        }
    }

    public static class ProgressLogger {
        public static void logProgress(int index) {
            if (index % 1000 != 0) {
                return;
            }
            if (index % 100000 == 0) {
                System.out.print("\n");
                if (index % 1000000 == 0) {
                    long usedMemory = AppContext.runtime.totalMemory()
                            - AppContext.runtime.freeMemory();
                    System.out.print(
                            index + ":" + usedMemory / 1024 / 1024 + "mb\n");
                }
                return;
            }
            System.out.print(".");
        }

        public static void logDuration(long start, long end) {
            long elapsed = end - start;
            String timeDuration = DurationFormatUtils.formatDuration(
                    (long) (elapsed / 1e6), "H:m:s.SSS");
            System.out.println("finished in " + timeDuration);
        }
    }

    /**
     * Retained for compatibility with existing callers. This helper performs a
     * complete record-count pass and should not be called before {@link #read()}
     * merely to preallocate the dataset.
     */
    public static class FileInfoExtractor {
        public static int[] getFileInformation(
                String fileName,
                boolean hasHeader,
                String separator
        ) throws IOException {
            char fieldSeparator = requireSingleCharacterSeparator(
                    normalizeSeparator(separator), "separator");
            int recordCount = 0;
            int columnCount = 0;
            try (CsvReader<CsvRecord> csvReader = CsvReader.builder()
                    .fieldSeparator(fieldSeparator)
                    .skipEmptyLines(false)
                    .detectBomHeader(true)
                    .ofCsvRecord(Path.of(fileName))) {
                var iterator = csvReader.iterator();
                if (hasHeader && iterator.hasNext()) {
                    columnCount = iterator.next().getFields().size();
                }
                while (iterator.hasNext()) {
                    CsvRecord record = iterator.next();
                    if (columnCount == 0) {
                        columnCount = record.getFields().size();
                    }
                    recordCount++;
                }
            }
            return new int[]{recordCount, columnCount};
        }
    }

    /** General row-parsing utilities retained for existing reader callers. */
    public static class RowParser {
        private static volatile Set<String> missingIndicators =
                normalizeMissingIndicators(AppContext.MissingStrings);

        public static void setMissingIndicators(Set<String> indicators) {
            missingIndicators = normalizeMissingIndicators(indicators);
        }

        public static Set<String> getMissingIndicators() {
            return missingIndicators;
        }

        public static boolean isMissingToken(String token) {
            if (token == null) {
                return true;
            }
            String trimmed = token.trim();
            return trimmed.isEmpty()
                    || missingIndicators.contains(
                    trimmed.toUpperCase(Locale.ROOT));
        }

        public static ParsedDoubleRow parseDoubleRow(
                String[] lineArray,
                boolean targetColumnIsFirst,
                boolean isRegression
        ) {
            int dataLength = lineArray.length - 1;
            double[] features = new double[dataLength];
            int labelIndex = targetColumnIsFirst ? 0 : dataLength;
            Object label = isRegression
                    ? JavaDoubleParser.parseDouble(lineArray[labelIndex].trim())
                    : tryParseLabel(lineArray[labelIndex]);
            int outputIndex = 0;
            for (int index = 0; index < lineArray.length; index++) {
                if (index != labelIndex) {
                    features[outputIndex++] = JavaDoubleParser.parseDouble(
                            lineArray[index].trim());
                }
            }
            return new ParsedDoubleRow(label, features);
        }

        public static Object tryParseLabel(String token) {
            String trimmed = token.trim();
            try {
                return Integer.parseInt(trimmed);
            } catch (NumberFormatException ignored) {
                try {
                    double value = JavaDoubleParser.parseDouble(trimmed);
                    if (value == Math.rint(value)
                            && value >= Integer.MIN_VALUE
                            && value <= Integer.MAX_VALUE) {
                        return (int) value;
                    }
                    return value;
                } catch (NumberFormatException ignoredAgain) {
                    return trimmed;
                }
            }
        }

        public static double[] parseDoubleArray(
                String row,
                String separator
        ) {
            char delimiter = requireSingleCharacterSeparator(
                    normalizeSeparator(separator), "separator");
            String[] tokens = splitLiteral(row, delimiter);
            double[] parsed = new double[tokens.length];
            for (int index = 0; index < tokens.length; index++) {
                parsed[index] = JavaDoubleParser.parseDouble(
                        tokens[index].trim());
            }
            return parsed;
        }

        public static double[][] parseDoubleMatrix(
                String row,
                String arraySeparator,
                String entrySeparator
        ) {
            char outer = requireSingleCharacterSeparator(
                    normalizeSeparator(arraySeparator), "arraySeparator");
            String[] rowStrings = splitLiteral(row, outer);
            double[][] matrix = new double[rowStrings.length][];
            for (int index = 0; index < rowStrings.length; index++) {
                matrix[index] = parseDoubleArray(
                        rowStrings[index], entrySeparator);
            }
            return matrix;
        }

        public static Object[] parse1DRow(String row, String separator) {
            char delimiter = requireSingleCharacterSeparator(
                    normalizeSeparator(separator), "separator");
            String[] tokens = splitLiteral(row, delimiter);
            Object[] parsed = new Object[tokens.length];
            for (int index = 0; index < tokens.length; index++) {
                parsed[index] = parseValue(tokens[index]);
            }
            return parsed;
        }

        public static Object[][] parse2DRow(
                String row,
                String arraySeparator,
                String entrySeparator
        ) {
            char outer = requireSingleCharacterSeparator(
                    normalizeSeparator(arraySeparator), "arraySeparator");
            String[] rowStrings = splitLiteral(row, outer);
            Object[][] matrix = new Object[rowStrings.length][];
            for (int index = 0; index < rowStrings.length; index++) {
                matrix[index] = parse1DRow(
                        rowStrings[index], entrySeparator);
            }
            return matrix;
        }

        public static Object parseValue(String token) {
            return isMissingToken(token)
                    ? null
                    : parseNonMissingValue(token);
        }

        private static Object parseNonMissingValue(String token) {
            String trimmed = token.trim();
            try {
                return JavaDoubleParser.parseDouble(trimmed);
            } catch (NumberFormatException ignored) {
                if (trimmed.equalsIgnoreCase("true")
                        || trimmed.equalsIgnoreCase("false")) {
                    return Boolean.parseBoolean(trimmed);
                }
                return trimmed;
            }
        }

        private static Set<String> normalizeMissingIndicators(
                Set<String> indicators
        ) {
            if (indicators == null || indicators.isEmpty()) {
                return Set.of();
            }
            Set<String> normalized = new HashSet<>();
            for (String indicator : indicators) {
                if (indicator == null) {
                    continue;
                }
                String trimmed = indicator.trim();
                if (!trimmed.isEmpty()) {
                    normalized.add(trimmed.toUpperCase(Locale.ROOT));
                }
            }
            return normalized.isEmpty()
                    ? Set.of()
                    : Collections.unmodifiableSet(normalized);
        }
    }
}
