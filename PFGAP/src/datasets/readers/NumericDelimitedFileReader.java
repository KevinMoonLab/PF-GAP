package datasets.readers;

import ch.randelshofer.fastdoubleparser.JavaDoubleParser;
import ch.randelshofer.fastdoubleparser.JavaFloatParser;
import core.AppContext;
import datasets.ListObjectDataset;
import datasets.NumericStorageType;
import de.siegmar.fastcsv.reader.AbstractBaseCsvCallbackHandler;
import de.siegmar.fastcsv.reader.CsvReader;

import java.io.IOException;
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
 * High-throughput reader for row-oriented numeric delimited datasets.
 *
 * <p>One delimited record becomes one dataset instance. Under
 * {@link NumericStorageType#FLOAT32}, each instance is stored as a primitive
 * {@code float[]}. Under {@link NumericStorageType#FLOAT64}, each instance is
 * stored as a primitive {@code double[]}. Because delimited text has no source
 * binary dtype, {@link NumericStorageType#AUTO} resolves to FLOAT64.</p>
 *
 * <p>Numeric fields are parsed directly from FastCSV's character buffer with
 * {@link JavaFloatParser} or {@link JavaDoubleParser}. The selected callback
 * has a type-specialized hot path, so parsing a numeric field does not perform
 * a per-value storage-type branch and does not create a temporary String.
 * Except for the first headerless record, the reader also avoids creating a
 * CsvRecord, a List of field strings, or one String per numeric value.</p>
 *
 * <p>Configured missing tokens and blank numeric fields become
 * {@link Float#NaN} or {@link Double#NaN} when missing values are enabled.
 * This reader never creates boxed numeric arrays.</p>
 *
 * <p>The reader returns raw eager data. Standardization, missing-index
 * construction, and imputation remain owned by PFGAP's preparation pipeline.</p>
 */
public class NumericDelimitedFileReader implements DatasetReader {

    private final String dataFileName;
    private final String labelFileName;
    private final String entrySeparator;
    private final char fieldSeparator;
    private final boolean hasHeader;
    private final boolean hasMissingValues;
    private final boolean targetColumnIsFirst;
    private final boolean isTest;
    private final boolean isRegression;
    private final NumericStorageType numericStorageType;
    private final Set<String> missingStrings;

    public NumericDelimitedFileReader(
            ReaderOptions options
    ) {
        this(
                Objects.requireNonNull(
                        options,
                        "ReaderOptions cannot be null."
                ).getDataPath(),
                options.getLabelPath(),
                options.getEntrySeparator(),
                options.hasHeader(),
                options.hasMissingValues(),
                options.targetColumnIsFirst(),
                options.isTest(),
                options.isRegression(),
                options.getNumericStorageType()
        );
    }

    /**
     * Compatibility constructor for direct Java callers. Delimited text
     * defaults to FLOAT64 under AUTO.
     */
    public NumericDelimitedFileReader(
            String dataFileName,
            String labelFileName,
            String entrySeparator,
            boolean hasHeader,
            boolean hasMissingValues,
            boolean targetColumnIsFirst,
            boolean isTest,
            boolean isRegression
    ) {
        this(
                dataFileName,
                labelFileName,
                entrySeparator,
                hasHeader,
                hasMissingValues,
                targetColumnIsFirst,
                isTest,
                isRegression,
                NumericStorageType.AUTO
        );
    }

    public NumericDelimitedFileReader(
            String dataFileName,
            String labelFileName,
            String entrySeparator,
            boolean hasHeader,
            boolean hasMissingValues,
            boolean targetColumnIsFirst,
            boolean isTest,
            boolean isRegression,
            NumericStorageType numericStorageType
    ) {
        this.dataFileName = requireNonblank(
                dataFileName,
                "dataFileName"
        );
        this.labelFileName = normalizeNullableString(labelFileName);
        this.entrySeparator =
                validateAndNormalizeSeparator(entrySeparator);
        this.fieldSeparator = this.entrySeparator.charAt(0);
        this.hasHeader = hasHeader;
        this.hasMissingValues = hasMissingValues;
        this.targetColumnIsFirst = targetColumnIsFirst;
        this.isTest = isTest;
        this.isRegression = isRegression;
        this.numericStorageType = resolveDelimitedStorageType(
                numericStorageType
        );
        this.missingStrings = snapshotMissingStrings();
    }

    @Override
    public ListObjectDataset read() throws IOException {
        long start = System.nanoTime();
        Path dataPath = Path.of(dataFileName);
        validateDataFile(dataPath);

        List<Object> labels = labelFileName == null
                ? List.of()
                : DelimitedFileReader.readGenericLabels(
                        labelFileName,
                        hasHeader,
                        isRegression
                );
        ListObjectDataset dataset = new ListObjectDataset();
        NumericDatasetCallbackHandler handler = createHandler(
                dataPath,
                dataset,
                labels
        );

        try (CsvReader<Boolean> csvReader = CsvReader.builder()
                .fieldSeparator(fieldSeparator)
                .skipEmptyLines(false)
                .detectBomHeader(true)
                .build(handler, dataPath)) {
            for (Boolean ignored : csvReader) {
                // Parsing and dataset construction occur in the callback.
            }
        } catch (IllegalArgumentException e) {
            throw e;
        } catch (RuntimeException e) {
            throw new IOException(
                    "Failed while parsing numeric delimited dataset: "
                            + dataPath,
                    e
            );
        }

        int instanceCount = handler.getInstanceCount();
        if (instanceCount == 0) {
            throw new IOException(
                    "Numeric delimited dataset contains no data records: "
                            + dataPath
            );
        }
        validateSeparateLabelCount(labels, instanceCount);

        int commonLength = handler.hasCommonFeatureCount()
                ? handler.getCommonFeatureCount()
                : 0;
        dataset.setLength(commonLength);
        AppContext.length = commonLength;
        DelimitedFileReader.ProgressLogger.logDuration(
                start,
                System.nanoTime()
        );
        return dataset;
    }

    private NumericDatasetCallbackHandler createHandler(
            Path dataPath,
            ListObjectDataset dataset,
            List<Object> labels
    ) {
        boolean embeddedLabel = shouldParseEmbeddedLabel();
        if (numericStorageType == NumericStorageType.FLOAT32) {
            return new FloatNumericDatasetCallbackHandler(
                    dataPath,
                    dataset,
                    labels,
                    hasHeader,
                    hasMissingValues,
                    embeddedLabel,
                    targetColumnIsFirst,
                    isRegression,
                    missingStrings
            );
        }
        return new DoubleNumericDatasetCallbackHandler(
                dataPath,
                dataset,
                labels,
                hasHeader,
                hasMissingValues,
                embeddedLabel,
                targetColumnIsFirst,
                isRegression,
                missingStrings
        );
    }

    private boolean shouldParseEmbeddedLabel() {
        return labelFileName == null
                && !AppContext.isIsolationMode()
                && (!isTest || AppContext.exists_testlabels);
    }

    private void validateSeparateLabelCount(
            List<Object> labels,
            int instanceCount
    ) {
        if (labelFileName == null) {
            return;
        }
        if (labels.size() != instanceCount) {
            throw new IllegalArgumentException(
                    "Separate label count does not match the number of data "
                            + "instances. Labels="
                            + labels.size()
                            + ", instances="
                            + instanceCount
                            + "."
            );
        }
    }

    private static NumericStorageType resolveDelimitedStorageType(
            NumericStorageType requested
    ) {
        NumericStorageType nonnull = Objects.requireNonNull(
                requested,
                "numericStorageType cannot be null."
        );
        return nonnull == NumericStorageType.AUTO
                ? NumericStorageType.FLOAT64
                : nonnull;
    }

    private static void validateDataFile(
            Path dataPath
    ) throws IOException {
        if (!Files.exists(dataPath)) {
            throw new IOException(
                    "Numeric delimited data file does not exist: "
                            + dataPath
            );
        }
        if (!Files.isRegularFile(dataPath)) {
            throw new IOException(
                    "Numeric delimited data path is not a regular file: "
                            + dataPath
            );
        }
        if (!Files.isReadable(dataPath)) {
            throw new IOException(
                    "Numeric delimited data file is not readable: "
                            + dataPath
            );
        }
    }

    private static String requireNonblank(
            String value,
            String argumentName
    ) {
        if (value == null || value.isBlank()) {
            throw new IllegalArgumentException(
                    argumentName + " cannot be null or blank."
            );
        }
        return value.trim();
    }

    private static String normalizeNullableString(
            String value
    ) {
        if (value == null) {
            return null;
        }
        String trimmed = value.trim();
        if (trimmed.isEmpty()
                || trimmed.equalsIgnoreCase("None")) {
            return null;
        }
        return trimmed;
    }

    private static String validateAndNormalizeSeparator(
            String separator
    ) {
        if (separator == null || separator.isEmpty()) {
            throw new IllegalArgumentException(
                    "NumericDelimitedFileReader requires a non-empty "
                            + "entry separator."
            );
        }
        String normalized = switch (separator) {
            case "\\t" -> "\t";
            case "\\n" -> "\n";
            case "\\r" -> "\r";
            default -> separator;
        };
        if (normalized.length() != 1) {
            throw new IllegalArgumentException(
                    "NumericDelimitedFileReader requires a "
                            + "single-character entry separator. Received: '"
                            + separator
                            + "'."
            );
        }
        char delimiter = normalized.charAt(0);
        if (delimiter == '\n' || delimiter == '\r') {
            throw new IllegalArgumentException(
                    "A line-separator character cannot be used as the "
                            + "entry separator."
            );
        }
        return normalized;
    }

    private static Set<String> snapshotMissingStrings() {
        if (AppContext.MissingStrings == null
                || AppContext.MissingStrings.isEmpty()) {
            return Set.of();
        }
        Set<String> normalized = new HashSet<>();
        for (String value : AppContext.MissingStrings) {
            if (value == null) {
                continue;
            }
            String trimmed = value.trim();
            if (!trimmed.isEmpty()) {
                normalized.add(
                        trimmed.toUpperCase(Locale.ROOT)
                );
            }
        }
        return normalized.isEmpty()
                ? Set.of()
                : Collections.unmodifiableSet(normalized);
    }

    /**
     * Shared record/schema/label handling. Numeric field parsing and primitive
     * row storage remain type-specialized in the concrete callback classes.
     */
    private abstract static class NumericDatasetCallbackHandler
            extends AbstractBaseCsvCallbackHandler<Boolean> {

        protected final Path file;
        protected final ListObjectDataset dataset;
        protected final boolean hasMissingValues;
        protected final boolean embeddedLabel;

        private final List<Object> separateLabels;
        private final boolean hasHeader;
        private final boolean targetColumnIsFirst;
        private final boolean isRegression;
        private final Set<String> missingStrings;

        private List<String> firstRecordFields = new ArrayList<>();
        private int expectedFieldCount = -1;
        private int expectedFeatureCount = -1;
        private int instanceCount;
        private int commonFeatureCount = -1;
        private String currentLabelToken;
        private boolean commonFeatureCountValid = true;
        private boolean schemaResolved;

        protected NumericDatasetCallbackHandler(
                Path file,
                ListObjectDataset dataset,
                List<Object> separateLabels,
                boolean hasHeader,
                boolean hasMissingValues,
                boolean embeddedLabel,
                boolean targetColumnIsFirst,
                boolean isRegression,
                Set<String> missingStrings
        ) {
            this.file = file;
            this.dataset = dataset;
            this.separateLabels = separateLabels;
            this.hasHeader = hasHeader;
            this.hasMissingValues = hasMissingValues;
            this.embeddedLabel = embeddedLabel;
            this.targetColumnIsFirst = targetColumnIsFirst;
            this.isRegression = isRegression;
            this.missingStrings = missingStrings;
        }

        @Override
        public final void handleField(
                int fieldIndex,
                char[] buffer,
                int offset,
                int length,
                boolean quoted
        ) {
            if (!schemaResolved) {
                firstRecordFields.add(
                        new String(buffer, offset, length)
                );
                return;
            }
            if (fieldIndex >= expectedFieldCount) {
                throw inconsistentColumnCount(fieldIndex + 1);
            }
            ensureCurrentRow();
            if (embeddedLabel && isLabelField(fieldIndex)) {
                currentLabelToken =
                        new String(buffer, offset, length).trim();
                return;
            }
            appendNumericField(
                    buffer,
                    offset,
                    length,
                    fieldIndex
            );
        }

        @Override
        protected final Boolean buildRecord() {
            int actualFieldCount = getFieldCount();
            if (!schemaResolved) {
                resolveSchema(actualFieldCount);
                schemaResolved = true;
                if (hasHeader) {
                    firstRecordFields = null;
                    return null;
                }
                appendBufferedFirstDataRecord();
                firstRecordFields = null;
                return Boolean.TRUE;
            }

            if (actualFieldCount != expectedFieldCount) {
                throw inconsistentColumnCount(actualFieldCount);
            }
            if (!hasCurrentRow()) {
                throw new IllegalArgumentException(
                        "Encountered an empty numeric record in file "
                                + file
                                + " at CSV line "
                                + getStartingLineNumber()
                                + "."
                );
            }
            if (getCurrentOutputIndex() != expectedFeatureCount) {
                throw new IllegalStateException(
                        "Numeric record in file "
                                + file
                                + " produced "
                                + getCurrentOutputIndex()
                                + " features; expected "
                                + expectedFeatureCount
                                + "."
                );
            }

            Object label = resolveCurrentLabel();
            finishCurrentRow(label);
            currentLabelToken = null;
            return Boolean.TRUE;
        }

        protected abstract void ensureCurrentRow();

        protected abstract boolean hasCurrentRow();

        protected abstract int getCurrentOutputIndex();

        protected abstract void appendNumericField(
                char[] buffer,
                int offset,
                int length,
                int fieldIndex
        );

        protected abstract void appendNumericToken(
                String token,
                int fieldIndex
        );

        protected abstract void finishCurrentRow(Object label);

        protected final int getExpectedFeatureCount() {
            return expectedFeatureCount;
        }

        protected final double parseDoubleField(
                char[] buffer,
                int offset,
                int length,
                int fieldIndex
        ) {
            int start = trimStart(buffer, offset, length);
            int end = trimEnd(buffer, start, offset + length);
            if (isMissingToken(buffer, start, end - start)) {
                if (hasMissingValues) {
                    return Double.NaN;
                }
                throw missingValue(fieldIndex);
            }
            try {
                return JavaDoubleParser.parseDouble(
                        buffer,
                        start,
                        end - start
                );
            } catch (NumberFormatException e) {
                throw numericParseFailure(fieldIndex, e);
            }
        }

        protected final float parseFloatField(
                char[] buffer,
                int offset,
                int length,
                int fieldIndex
        ) {
            int start = trimStart(buffer, offset, length);
            int end = trimEnd(buffer, start, offset + length);
            if (isMissingToken(buffer, start, end - start)) {
                if (hasMissingValues) {
                    return Float.NaN;
                }
                throw missingValue(fieldIndex);
            }
            try {
                return JavaFloatParser.parseFloat(
                        buffer,
                        start,
                        end - start
                );
            } catch (NumberFormatException e) {
                throw numericParseFailure(fieldIndex, e);
            }
        }

        protected final double parseDoubleToken(
                String token,
                int fieldIndex
        ) {
            String trimmed = token == null ? "" : token.trim();
            if (isMissingToken(trimmed)) {
                if (hasMissingValues) {
                    return Double.NaN;
                }
                throw missingValue(fieldIndex);
            }
            try {
                return JavaDoubleParser.parseDouble(trimmed);
            } catch (NumberFormatException e) {
                throw firstRecordParseFailure(
                        trimmed,
                        fieldIndex,
                        e
                );
            }
        }

        protected final float parseFloatToken(
                String token,
                int fieldIndex
        ) {
            String trimmed = token == null ? "" : token.trim();
            if (isMissingToken(trimmed)) {
                if (hasMissingValues) {
                    return Float.NaN;
                }
                throw missingValue(fieldIndex);
            }
            try {
                return JavaFloatParser.parseFloat(trimmed);
            } catch (NumberFormatException e) {
                throw firstRecordParseFailure(
                        trimmed,
                        fieldIndex,
                        e
                );
            }
        }

        protected final void addInstance(
                Object label,
                Object features,
                int featureCount
        ) {
            dataset.add(label, features, instanceCount);
            if (commonFeatureCount < 0) {
                commonFeatureCount = featureCount;
            } else if (featureCount != commonFeatureCount) {
                commonFeatureCountValid = false;
            }
            DelimitedFileReader.ProgressLogger.logProgress(
                    instanceCount
            );
            instanceCount++;
        }

        private void resolveSchema(int fieldCount) {
            if (fieldCount <= 0
                    || firstRecordFields.size() != fieldCount) {
                throw new IllegalArgumentException(
                        "Numeric delimited file has no valid columns: "
                                + file
                );
            }
            if (embeddedLabel && fieldCount < 2) {
                throw new IllegalArgumentException(
                        "Numeric delimited records in file "
                                + file
                                + " must contain at least one feature and "
                                + "one label."
                );
            }
            expectedFieldCount = fieldCount;
            expectedFeatureCount = embeddedLabel
                    ? fieldCount - 1
                    : fieldCount;
            if (expectedFeatureCount <= 0) {
                throw new IllegalArgumentException(
                        "No numeric feature columns are available in file: "
                                + file
                );
            }
        }

        private void appendBufferedFirstDataRecord() {
            ensureCurrentRow();
            String labelToken = null;
            for (int fieldIndex = 0;
                    fieldIndex < expectedFieldCount;
                    fieldIndex++) {
                String token = firstRecordFields.get(fieldIndex);
                if (embeddedLabel && isLabelField(fieldIndex)) {
                    labelToken = token == null ? "" : token.trim();
                    continue;
                }
                appendNumericToken(token, fieldIndex);
            }
            Object label = embeddedLabel
                    ? parseEmbeddedLabel(labelToken)
                    : getSeparateLabel(instanceCount);
            finishCurrentRow(label);
        }

        private boolean isLabelField(int fieldIndex) {
            return targetColumnIsFirst
                    ? fieldIndex == 0
                    : fieldIndex == expectedFieldCount - 1;
        }

        private static int trimStart(
                char[] buffer,
                int offset,
                int length
        ) {
            int start = offset;
            int end = offset + length;
            while (start < end
                    && Character.isWhitespace(buffer[start])) {
                start++;
            }
            return start;
        }

        private static int trimEnd(
                char[] buffer,
                int start,
                int endExclusive
        ) {
            int end = endExclusive;
            while (end > start
                    && Character.isWhitespace(buffer[end - 1])) {
                end--;
            }
            return end;
        }

        private boolean isMissingToken(
                char[] buffer,
                int offset,
                int length
        ) {
            if (length == 0) {
                return true;
            }
            if (missingStrings.isEmpty()) {
                return false;
            }
            for (String indicator : missingStrings) {
                if (indicator.length() != length) {
                    continue;
                }
                boolean matches = true;
                for (int index = 0; index < length; index++) {
                    char actual = Character.toUpperCase(
                            buffer[offset + index]
                    );
                    if (actual != indicator.charAt(index)) {
                        matches = false;
                        break;
                    }
                }
                if (matches) {
                    return true;
                }
            }
            return false;
        }

        private boolean isMissingToken(String token) {
            return token == null
                    || token.isEmpty()
                    || missingStrings.contains(
                            token.toUpperCase(Locale.ROOT)
                    );
        }

        private IllegalArgumentException missingValue(
                int fieldIndex
        ) {
            return new IllegalArgumentException(
                    "Encountered a missing numeric value in file "
                            + file
                            + " at instance "
                            + instanceCount
                            + ", field "
                            + fieldIndex
                            + ", but hasMissingValues is false."
            );
        }

        private IllegalArgumentException numericParseFailure(
                int fieldIndex,
                NumberFormatException cause
        ) {
            return new IllegalArgumentException(
                    "Could not parse numeric value in file "
                            + file
                            + " at instance "
                            + instanceCount
                            + ", field "
                            + fieldIndex
                            + ", starting CSV line "
                            + getStartingLineNumber()
                            + ".",
                    cause
            );
        }

        private IllegalArgumentException firstRecordParseFailure(
                String token,
                int fieldIndex,
                NumberFormatException cause
        ) {
            return new IllegalArgumentException(
                    "Could not parse numeric value '"
                            + token
                            + "' in file "
                            + file
                            + " at instance 0, field "
                            + fieldIndex
                            + ".",
                    cause
            );
        }

        private Object resolveCurrentLabel() {
            if (embeddedLabel) {
                return parseEmbeddedLabel(currentLabelToken);
            }
            return getSeparateLabel(instanceCount);
        }

        private Object parseEmbeddedLabel(String token) {
            if (token == null || token.isEmpty()) {
                throw new IllegalArgumentException(
                        "Embedded label cannot be missing in file "
                                + file
                                + " at instance "
                                + instanceCount
                                + "."
                );
            }
            if (isRegression) {
                try {
                    return JavaDoubleParser.parseDouble(token);
                } catch (NumberFormatException e) {
                    throw new IllegalArgumentException(
                            "Could not parse regression label '"
                                    + token
                                    + "' in file "
                                    + file
                                    + " at instance "
                                    + instanceCount
                                    + ".",
                            e
                    );
                }
            }
            return DelimitedFileReader.RowParser.tryParseLabel(token);
        }

        private Object getSeparateLabel(int index) {
            if (separateLabels.isEmpty()) {
                return null;
            }
            if (index >= separateLabels.size()) {
                throw new IllegalArgumentException(
                        "The separate label file contains fewer labels than "
                                + "the data file. Missing label for instance "
                                + index
                                + "."
                );
            }
            return separateLabels.get(index);
        }

        private IllegalArgumentException inconsistentColumnCount(
                int actualFieldCount
        ) {
            return new IllegalArgumentException(
                    "Inconsistent column count in file "
                            + file
                            + " at CSV record beginning on line "
                            + getStartingLineNumber()
                            + ". Expected "
                            + expectedFieldCount
                            + " columns but found "
                            + actualFieldCount
                            + "."
            );
        }

        private int getInstanceCount() {
            return instanceCount;
        }

        private boolean hasCommonFeatureCount() {
            return commonFeatureCountValid
                    && commonFeatureCount >= 0;
        }

        private int getCommonFeatureCount() {
            return commonFeatureCount;
        }
    }

    private static final class DoubleNumericDatasetCallbackHandler
            extends NumericDatasetCallbackHandler {

        private double[] currentFeatures;
        private int currentOutputIndex;

        private DoubleNumericDatasetCallbackHandler(
                Path file,
                ListObjectDataset dataset,
                List<Object> separateLabels,
                boolean hasHeader,
                boolean hasMissingValues,
                boolean embeddedLabel,
                boolean targetColumnIsFirst,
                boolean isRegression,
                Set<String> missingStrings
        ) {
            super(
                    file,
                    dataset,
                    separateLabels,
                    hasHeader,
                    hasMissingValues,
                    embeddedLabel,
                    targetColumnIsFirst,
                    isRegression,
                    missingStrings
            );
        }

        @Override
        protected void ensureCurrentRow() {
            if (currentFeatures == null) {
                currentFeatures =
                        new double[getExpectedFeatureCount()];
                currentOutputIndex = 0;
            }
        }

        @Override
        protected boolean hasCurrentRow() {
            return currentFeatures != null;
        }

        @Override
        protected int getCurrentOutputIndex() {
            return currentOutputIndex;
        }

        @Override
        protected void appendNumericField(
                char[] buffer,
                int offset,
                int length,
                int fieldIndex
        ) {
            currentFeatures[currentOutputIndex++] =
                    parseDoubleField(
                            buffer,
                            offset,
                            length,
                            fieldIndex
                    );
        }

        @Override
        protected void appendNumericToken(
                String token,
                int fieldIndex
        ) {
            currentFeatures[currentOutputIndex++] =
                    parseDoubleToken(token, fieldIndex);
        }

        @Override
        protected void finishCurrentRow(Object label) {
            addInstance(
                    label,
                    currentFeatures,
                    currentFeatures.length
            );
            currentFeatures = null;
            currentOutputIndex = 0;
        }
    }

    private static final class FloatNumericDatasetCallbackHandler
            extends NumericDatasetCallbackHandler {

        private float[] currentFeatures;
        private int currentOutputIndex;

        private FloatNumericDatasetCallbackHandler(
                Path file,
                ListObjectDataset dataset,
                List<Object> separateLabels,
                boolean hasHeader,
                boolean hasMissingValues,
                boolean embeddedLabel,
                boolean targetColumnIsFirst,
                boolean isRegression,
                Set<String> missingStrings
        ) {
            super(
                    file,
                    dataset,
                    separateLabels,
                    hasHeader,
                    hasMissingValues,
                    embeddedLabel,
                    targetColumnIsFirst,
                    isRegression,
                    missingStrings
            );
        }

        @Override
        protected void ensureCurrentRow() {
            if (currentFeatures == null) {
                currentFeatures =
                        new float[getExpectedFeatureCount()];
                currentOutputIndex = 0;
            }
        }

        @Override
        protected boolean hasCurrentRow() {
            return currentFeatures != null;
        }

        @Override
        protected int getCurrentOutputIndex() {
            return currentOutputIndex;
        }

        @Override
        protected void appendNumericField(
                char[] buffer,
                int offset,
                int length,
                int fieldIndex
        ) {
            currentFeatures[currentOutputIndex++] =
                    parseFloatField(
                            buffer,
                            offset,
                            length,
                            fieldIndex
                    );
        }

        @Override
        protected void appendNumericToken(
                String token,
                int fieldIndex
        ) {
            currentFeatures[currentOutputIndex++] =
                    parseFloatToken(token, fieldIndex);
        }

        @Override
        protected void finishCurrentRow(Object label) {
            addInstance(
                    label,
                    currentFeatures,
                    currentFeatures.length
            );
            currentFeatures = null;
            currentOutputIndex = 0;
        }
    }
}
