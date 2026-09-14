package datasets.readers;

import ch.randelshofer.fastdoubleparser.JavaDoubleParser;
import datasets.readers.lazy.LazySeriesReader;
import datasets.readers.lazy.LazySeriesRef;
import de.siegmar.fastcsv.reader.AbstractBaseCsvCallbackHandler;
import de.siegmar.fastcsv.reader.CsvReader;
import preprocessing.standardization.StandardizationStats;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * High-throughput numeric reader for per-file delimited time-series data.
 *
 * <p>Storage assumption:</p>
 *
 * <pre>
 * one delimited file = one dataset instance / time series
 * one CSV record = one time point
 * one selected column = one time-series dimension
 * </pre>
 *
 * <p>Returned representation:</p>
 *
 * <pre>
 * double[dimension][time]
 * </pre>
 *
 * <p>This reader is deliberately specialized for numeric files without
 * missing values. It does not support categorical dimensions or nullable
 * numeric output. Use {@link PerFileDelimitedSeriesReader} when generic or
 * missing-value support is required.</p>
 *
 * <p>The implementation uses FastCSV's custom callback API. Numeric fields
 * are parsed directly from FastCSV's character buffer through
 * {@link JavaDoubleParser}, avoiding the ordinary intermediate path through
 * {@code CsvRecord}, {@code List<String>}, and one {@code String} object per
 * selected numeric value.</p>
 *
 * <p>Header behavior:</p>
 *
 * <ul>
 *     <li>
 *         When {@code hasHeader=true}, {@code featureColumns} and
 *         {@code timeColumn} are interpreted as column names.
 *     </li>
 *     <li>
 *         When {@code hasHeader=false}, they are interpreted as zero-based
 *         integer column indices.
 *     </li>
 *     <li>
 *         When {@code featureColumns} is empty, every column except the
 *         configured time column is treated as a feature column.
 *     </li>
 * </ul>
 *
 * <p>The configured field separator must contain exactly one character.
 * Escaped tab separators such as {@code "\\t"} are normalized.</p>
 *
 * <p>When prepared statistics are supplied, values are standardized as
 * FastCSV fields are parsed. Centers and inverse scales are resolved once per
 * dimension, so lazy materialization needs no post-read traversal.</p>
 *
 * <p>Instances may have unequal time lengths across files. Within one file,
 * however, every selected feature column necessarily receives one value per
 * record and therefore has the same length.</p>
 */
public class NumericPerFileDelimitedSeriesReader
        implements LazySeriesReader {

    private static final int DEFAULT_INITIAL_TIME_CAPACITY =
            256;

    private final String entrySeparator;
    private final char fieldSeparator;
    private final boolean hasHeader;
    private final String timeColumn;
    private final List<String> featureColumns;
    private final StandardizationStats standardizationStats;
    private final int initialTimeCapacity;

    public NumericPerFileDelimitedSeriesReader(
            String entrySeparator,
            boolean hasHeader,
            String timeColumn,
            List<String> featureColumns,
            StandardizationStats standardizationStats
    ) {
        this(
                entrySeparator,
                hasHeader,
                timeColumn,
                featureColumns,
                standardizationStats,
                DEFAULT_INITIAL_TIME_CAPACITY
        );
    }

    public NumericPerFileDelimitedSeriesReader(
            String entrySeparator,
            boolean hasHeader,
            String timeColumn,
            List<String> featureColumns
    ) {
        this(
                entrySeparator,
                hasHeader,
                timeColumn,
                featureColumns,
                null,
                DEFAULT_INITIAL_TIME_CAPACITY
        );
    }

    public NumericPerFileDelimitedSeriesReader(
            String entrySeparator,
            boolean hasHeader,
            String timeColumn,
            List<String> featureColumns,
            StandardizationStats standardizationStats,
            int initialTimeCapacity
    ) {
        this.entrySeparator =
                validateAndNormalizeSeparator(
                        entrySeparator
                );

        this.fieldSeparator =
                this.entrySeparator.charAt(0);

        this.hasHeader =
                hasHeader;

        this.timeColumn =
                normalizeNullableString(
                        timeColumn
                );

        this.featureColumns =
                featureColumns == null
                        ? List.of()
                        : List.copyOf(
                        featureColumns
                );

        this.standardizationStats =
                standardizationStats;

        if (initialTimeCapacity < 1) {
            throw new IllegalArgumentException(
                    "NumericPerFileDelimitedSeriesReader "
                            + "initialTimeCapacity must be at least 1. "
                            + "Received: "
                            + initialTimeCapacity
                            + "."
            );
        }

        this.initialTimeCapacity =
                initialTimeCapacity;

        validateFeatureConfiguration();
        validateStandardizationConfiguration();
    }

    /**
     * Materializes the file represented by a lazy series reference.
     *
     * <p>The referenced path was already discovered by the dataset reader,
     * so this path avoids repeating the full set of filesystem metadata
     * checks on every lazy materialization.</p>
     *
     * @param reference lazy per-file series reference
     * @return a {@code double[dimension][time]} series
     */
    @Override
    public Object read(
            LazySeriesRef reference
    ) {
        if (reference == null) {
            throw new IllegalArgumentException(
                    "Cannot read a null LazySeriesRef."
            );
        }

        Path file =
                reference.getFile();

        try {
            return readFileInternal(
                    file,
                    false
            );
        } catch (IOException e) {
            throw new IllegalStateException(
                    "Failed to read numeric delimited time-series file: "
                            + file,
                    e
            );
        }
    }

    /**
     * Reads one numeric per-file time series directly.
     *
     * <p>This public path validates that the file exists, is regular, and is
     * readable before attempting to parse it.</p>
     *
     * @param file numeric delimited time-series file
     * @return a {@code double[dimension][time]} series
     * @throws IOException if the file cannot be opened or parsed
     */
    public double[][] readFile(
            Path file
    ) throws IOException {
        return readFileInternal(
                file,
                true
        );
    }

    private double[][] readFileInternal(
            Path file,
            boolean validateFileMetadata
    ) throws IOException {
        if (file == null) {
            throw new IllegalArgumentException(
                    "NumericPerFileDelimitedSeriesReader requires "
                            + "a non-null file."
            );
        }

        if (validateFileMetadata) {
            validateFile(
                    file
            );
        }

        NumericSeriesCallbackHandler handler =
                new NumericSeriesCallbackHandler(
                        file,
                        hasHeader,
                        timeColumn,
                        featureColumns,
                        standardizationStats,
                        initialTimeCapacity
                );

        try (CsvReader<Boolean> csvReader =
                     CsvReader.builder()
                             .fieldSeparator(fieldSeparator)
                             .skipEmptyLines(false)
                             .detectBomHeader(true)
                             .build(
                                     handler,
                                     file
                             )) {

            /*
             * FastCSV performs parsing as the reader is consumed.
             * The Boolean values themselves are only lightweight completion
             * markers. All materialized series data remains in the handler.
             */
            for (Boolean ignored : csvReader) {
                // Intentionally empty.
            }
        } catch (IllegalArgumentException e) {
            throw e;
        } catch (RuntimeException e) {
            throw new IOException(
                    "Failed while parsing numeric delimited "
                            + "time-series file: "
                            + file,
                    e
            );
        }

        return handler.toSeries();
    }

    private void validateFile(
            Path file
    ) throws IOException {
        if (!Files.exists(file)) {
            throw new IOException(
                    "Numeric delimited time-series file does not exist: "
                            + file
            );
        }

        if (!Files.isRegularFile(file)) {
            throw new IOException(
                    "Numeric delimited time-series path is not "
                            + "a regular file: "
                            + file
            );
        }

        if (!Files.isReadable(file)) {
            throw new IOException(
                    "Numeric delimited time-series file is not readable: "
                            + file
            );
        }
    }

    private void validateFeatureConfiguration() {
        for (String featureColumn : featureColumns) {
            if (featureColumn == null
                    || featureColumn.isBlank()) {

                throw new IllegalArgumentException(
                        "Numeric per-file feature-column names or indices "
                                + "cannot be null or blank."
                );
            }
        }
    }

    private void validateStandardizationConfiguration() {
        if (standardizationStats == null) {
            return;
        }

        /*
         * When featureColumns is empty, every non-time column is selected.
         * In that case the final dimension count is validated after column
         * selection, before standardization parameters are initialized.
         */
        if (!featureColumns.isEmpty()) {
            standardizationStats.validateFeatureCompatibility(
                    featureColumns
            );
        }
    }

    private static String validateAndNormalizeSeparator(
            String separator
    ) {
        if (separator == null || separator.isEmpty()) {
            throw new IllegalArgumentException(
                    "NumericPerFileDelimitedSeriesReader requires "
                            + "a non-empty entry separator."
            );
        }

        String normalized =
                normalizeSeparator(
                        separator
                );

        if (normalized.length() != 1) {
            throw new IllegalArgumentException(
                    "NumericPerFileDelimitedSeriesReader requires "
                            + "a single-character entry separator. "
                            + "Received: '"
                            + separator
                            + "'."
            );
        }

        char delimiter =
                normalized.charAt(0);

        if (delimiter == '\n' || delimiter == '\r') {
            throw new IllegalArgumentException(
                    "A line-separator character cannot be used as "
                            + "the entry separator."
            );
        }

        return normalized;
    }

    private static String normalizeSeparator(
            String separator
    ) {
        return switch (separator) {
            case "\\t" -> "\t";
            case "\\n" -> "\n";
            case "\\r" -> "\r";
            default -> separator;
        };
    }

    private static String normalizeNullableString(
            String value
    ) {
        if (value == null) {
            return null;
        }

        String trimmed =
                value.trim();

        if (trimmed.isEmpty()
                || trimmed.equalsIgnoreCase("None")) {

            return null;
        }

        return trimmed;
    }

    /**
     * FastCSV callback that maps numeric CSV fields directly into
     * dimension-major primitive buffers.
     *
     * <p>The first record is temporarily materialized as strings because the
     * field count and, when applicable, column names are needed to resolve the
     * selected columns. After selection has been resolved, subsequent selected
     * numerical fields are parsed directly from FastCSV's character buffer.</p>
     */
    private static final class NumericSeriesCallbackHandler
            extends AbstractBaseCsvCallbackHandler<Boolean> {

        private final Path file;
        private final boolean hasHeader;
        private final String timeColumn;
        private final List<String> featureColumns;
        private final StandardizationStats standardizationStats;
        private final int initialTimeCapacity;

        private double[] centers;
        private double[] inverseScales;
        private List<String> firstRecordFields;
        private int[] columnToDimension;
        private PrimitiveDoubleBuffer[] dimensions;

        private long physicalRecordIndex;
        private int dataRecordCount;
        private boolean selectionResolved;

        private NumericSeriesCallbackHandler(
                Path file,
                boolean hasHeader,
                String timeColumn,
                List<String> featureColumns,
                StandardizationStats standardizationStats,
                int initialTimeCapacity
        ) {
            this.file =
                    file;

            this.hasHeader =
                    hasHeader;

            this.timeColumn =
                    timeColumn;

            this.featureColumns =
                    featureColumns;

            this.standardizationStats =
                    standardizationStats;

            this.initialTimeCapacity =
                    initialTimeCapacity;

            this.firstRecordFields =
                    new ArrayList<>();
        }

        /**
         * Receives one parsed CSV field from FastCSV.
         *
         * <p>After the first record has established the schema, unselected
         * fields are ignored and selected fields are converted directly from
         * the supplied character range.</p>
         */
        @Override
        public void handleField(
                int fieldIndex,
                char[] buffer,
                int offset,
                int length,
                boolean quoted
        ) {
            if (!selectionResolved) {
                firstRecordFields.add(
                        new String(
                                buffer,
                                offset,
                                length
                        )
                );

                return;
            }

            if (fieldIndex >= columnToDimension.length) {
                throw inconsistentColumnCount(
                        fieldIndex + 1
                );
            }

            int dimension =
                    columnToDimension[fieldIndex];

            if (dimension < 0) {
                return;
            }

            if (isBlank(
                    buffer,
                    offset,
                    length
            )) {
                throw new IllegalArgumentException(
                        "Encountered a missing numeric value in file "
                                + file
                                + " at data record "
                                + dataRecordCount
                                + ", column "
                                + fieldIndex
                                + ". NumericPerFileDelimitedSeriesReader "
                                + "does not support missing values."
                );
            }

            try {
                double value =
                        JavaDoubleParser.parseDouble(
                                buffer,
                                offset,
                                length
                        );

                dimensions[dimension].add(
                        standardizeIfConfigured(value, dimension)
                );
            } catch (NumberFormatException e) {
                throw new IllegalArgumentException(
                        "Could not parse numeric value in file "
                                + file
                                + " at data record "
                                + dataRecordCount
                                + ", column "
                                + fieldIndex
                                + ", starting CSV line "
                                + getStartingLineNumber()
                                + ".",
                        e
                );
            }
        }

        /**
         * Finalizes one CSV record.
         *
         * <p>FastCSV invokes this method after all fields in the record have
         * been supplied to {@link #handleField(int, char[], int, int, boolean)}.
         * The first record resolves the schema. For a headerless file, it is
         * then parsed as the first data record.</p>
         */
        @Override
        protected Boolean buildRecord() {
            int fieldCount =
                    getFieldCount();

            if (!selectionResolved) {
                resolveSelection(
                        fieldCount
                );

                selectionResolved =
                        true;

                if (hasHeader) {
                    physicalRecordIndex++;
                    firstRecordFields =
                            null;

                    /*
                     * Returning null tells FastCSV that the header does not
                     * represent a materialized output record.
                     */
                    return null;
                }

                appendBufferedFirstDataRecord();

                dataRecordCount++;
                physicalRecordIndex++;
                firstRecordFields =
                        null;

                return Boolean.TRUE;
            }

            if (fieldCount != columnToDimension.length) {
                throw inconsistentColumnCount(
                        fieldCount
                );
            }

            dataRecordCount++;
            physicalRecordIndex++;

            return Boolean.TRUE;
        }

        private void resolveSelection(
                int columnCount
        ) {
            if (columnCount <= 0) {
                throw new IllegalArgumentException(
                        "Numeric delimited file has no columns: "
                                + file
                );
            }

            if (firstRecordFields.size() != columnCount) {
                throw new IllegalStateException(
                        "FastCSV field count did not match the buffered "
                                + "first-record field count in file "
                                + file
                                + ". FastCSV count="
                                + columnCount
                                + ", buffered count="
                                + firstRecordFields.size()
                                + "."
                );
            }

            int timeColumnIndex =
                    resolveTimeColumnIndex(
                            columnCount
                    );

            int[] featureIndices =
                    resolveFeatureIndices(
                            columnCount,
                            timeColumnIndex
                    );

            if (featureIndices.length == 0) {
                throw new IllegalArgumentException(
                        "No feature columns were selected for file: "
                                + file
                );
            }

            columnToDimension =
                    new int[columnCount];

            Arrays.fill(
                    columnToDimension,
                    -1
            );

            dimensions =
                    new PrimitiveDoubleBuffer[featureIndices.length];

            for (int dimension = 0;
                 dimension < featureIndices.length;
                 dimension++) {

                int columnIndex =
                        featureIndices[dimension];

                columnToDimension[columnIndex] =
                        dimension;

                dimensions[dimension] =
                        new PrimitiveDoubleBuffer(
                                initialTimeCapacity
                        );
            }

            initializeStandardizationParameters(
                    featureIndices.length
            );
        }

        private int resolveTimeColumnIndex(
                int columnCount
        ) {
            if (timeColumn == null) {
                return -1;
            }

            if (hasHeader) {
                Map<String, Integer> headerIndex =
                        buildHeaderIndex();

                Integer index =
                        headerIndex.get(
                                timeColumn
                        );

                if (index == null) {
                    throw new IllegalArgumentException(
                            "Could not find time column '"
                                    + timeColumn
                                    + "' in file "
                                    + file
                                    + ". Available columns: "
                                    + headerIndex.keySet()
                    );
                }

                return index;
            }

            int index =
                    parseColumnIndex(
                            timeColumn,
                            "time_column"
                    );

            validateColumnIndex(
                    index,
                    columnCount,
                    "time_column"
            );

            return index;
        }

        private int[] resolveFeatureIndices(
                int columnCount,
                int timeColumnIndex
        ) {
            if (featureColumns.isEmpty()) {
                return allColumnsExcept(
                        columnCount,
                        timeColumnIndex
                );
            }

            int[] featureIndices =
                    new int[featureColumns.size()];

            boolean[] used =
                    new boolean[columnCount];

            Map<String, Integer> headerIndex =
                    hasHeader
                            ? buildHeaderIndex()
                            : Map.of();

            for (int dimension = 0;
                 dimension < featureColumns.size();
                 dimension++) {

                String feature =
                        featureColumns.get(dimension);

                int columnIndex;

                if (hasHeader) {
                    Integer resolved =
                            headerIndex.get(
                                    feature
                            );

                    if (resolved == null) {
                        throw new IllegalArgumentException(
                                "Could not find feature column '"
                                        + feature
                                        + "' in file "
                                        + file
                                        + ". Available columns: "
                                        + headerIndex.keySet()
                        );
                    }

                    columnIndex =
                            resolved;
                } else {
                    columnIndex =
                            parseColumnIndex(
                                    feature,
                                    "feature column"
                            );

                    validateColumnIndex(
                            columnIndex,
                            columnCount,
                            "feature column"
                    );
                }

                if (columnIndex == timeColumnIndex) {
                    throw new IllegalArgumentException(
                            "Column "
                                    + feature
                                    + " is configured as both the time "
                                    + "column and a feature column in file "
                                    + file
                    );
                }

                if (used[columnIndex]) {
                    throw new IllegalArgumentException(
                            "Feature column was selected more than once: "
                                    + feature
                                    + " in file "
                                    + file
                    );
                }

                used[columnIndex] =
                        true;

                featureIndices[dimension] =
                        columnIndex;
            }

            return featureIndices;
        }

        private Map<String, Integer> buildHeaderIndex() {
            Map<String, Integer> indices =
                    new HashMap<>(
                            Math.max(
                                    16,
                                    firstRecordFields.size() * 2
                            )
                    );

            for (int columnIndex = 0;
                 columnIndex < firstRecordFields.size();
                 columnIndex++) {

                String rawName =
                        firstRecordFields.get(columnIndex);

                String name =
                        rawName == null
                                ? ""
                                : rawName.trim();

                if (name.isEmpty()) {
                    throw new IllegalArgumentException(
                            "Numeric delimited file contains a blank "
                                    + "header at column "
                                    + columnIndex
                                    + ": "
                                    + file
                    );
                }

                Integer previous =
                        indices.put(
                                name,
                                columnIndex
                        );

                if (previous != null) {
                    throw new IllegalArgumentException(
                            "Numeric delimited file contains duplicate "
                                    + "header '"
                                    + name
                                    + "': "
                                    + file
                    );
                }
            }

            return indices;
        }

        /**
         * Parses the first record when the file does not contain a header.
         *
         * <p>The first record had to be buffered as strings because column
         * selection could not be resolved until FastCSV reported its complete
         * field count.</p>
         */
        private void appendBufferedFirstDataRecord() {
            for (int columnIndex = 0;
                 columnIndex < firstRecordFields.size();
                 columnIndex++) {

                int dimension =
                        columnToDimension[columnIndex];

                if (dimension < 0) {
                    continue;
                }

                String token =
                        firstRecordFields.get(columnIndex);

                String trimmed =
                        token == null
                                ? ""
                                : token.trim();

                if (trimmed.isEmpty()) {
                    throw new IllegalArgumentException(
                            "Encountered a missing numeric value in file "
                                    + file
                                    + " at data record 0, column "
                                    + columnIndex
                                    + ". NumericPerFileDelimitedSeriesReader "
                                    + "does not support missing values."
                    );
                }

                try {
                    double value =
                            JavaDoubleParser.parseDouble(trimmed);

                    dimensions[dimension].add(
                            standardizeIfConfigured(value, dimension)
                    );
                } catch (NumberFormatException e) {
                    throw new IllegalArgumentException(
                            "Could not parse numeric value '"
                                    + trimmed
                                    + "' in file "
                                    + file
                                    + " at data record 0, column "
                                    + columnIndex
                                    + ".",
                            e
                    );
                }
            }
        }

        private void initializeStandardizationParameters(
                int dimensionCount
        ) {
            if (standardizationStats == null) {
                return;
            }

            if (standardizationStats.getScope()
                    == preprocessing.standardization.StandardizationScope
                    .PER_DIMENSION
                    && standardizationStats.getStatisticGroupCount()
                    != dimensionCount) {

                throw new IllegalArgumentException(
                        "Numeric delimited series contains "
                                + dimensionCount
                                + " dimensions, but PER_DIMENSION "
                                + "statistics contain "
                                + standardizationStats.getStatisticGroupCount()
                                + " groups."
                );
            }

            centers = new double[dimensionCount];
            inverseScales = new double[dimensionCount];

            for (int dimension = 0;
                 dimension < dimensionCount;
                 dimension++) {

                centers[dimension] =
                        standardizationStats.getCenterForDimension(dimension);

                inverseScales[dimension] =
                        1.0 / standardizationStats
                                .getScaleForDimension(dimension);
            }
        }

        private double standardizeIfConfigured(
                double value,
                int dimension
        ) {
            if (centers == null) {
                return value;
            }

            return (value - centers[dimension])
                    * inverseScales[dimension];
        }

        private int parseColumnIndex(
                String value,
                String role
        ) {
            try {
                return Integer.parseInt(
                        value.trim()
                );
            } catch (NumberFormatException e) {
                throw new IllegalArgumentException(
                        role
                                + " must be a zero-based integer column "
                                + "index when hasHeader=false. Received '"
                                + value
                                + "' for file "
                                + file
                                + ".",
                        e
                );
            }
        }

        private void validateColumnIndex(
                int index,
                int columnCount,
                String role
        ) {
            if (index < 0 || index >= columnCount) {
                throw new IllegalArgumentException(
                        role
                                + " index "
                                + index
                                + " is outside the valid range [0, "
                                + (columnCount - 1)
                                + "] for file "
                                + file
                                + "."
                );
            }
        }

        private int[] allColumnsExcept(
                int columnCount,
                int excludedColumn
        ) {
            int resultLength =
                    excludedColumn >= 0
                            ? columnCount - 1
                            : columnCount;

            int[] result =
                    new int[resultLength];

            int outputIndex =
                    0;

            for (int columnIndex = 0;
                 columnIndex < columnCount;
                 columnIndex++) {

                if (columnIndex == excludedColumn) {
                    continue;
                }

                result[outputIndex++] =
                        columnIndex;
            }

            return result;
        }

        private IllegalArgumentException inconsistentColumnCount(
                int actualColumnCount
        ) {
            int expectedColumnCount =
                    columnToDimension == null
                            ? firstRecordFields.size()
                            : columnToDimension.length;

            return new IllegalArgumentException(
                    "Inconsistent column count in file "
                            + file
                            + " at CSV record beginning on line "
                            + getStartingLineNumber()
                            + ". Expected "
                            + expectedColumnCount
                            + " columns but found "
                            + actualColumnCount
                            + "."
            );
        }

        private double[][] toSeries() {
            if (!selectionResolved) {
                throw new IllegalArgumentException(
                        "Numeric delimited time-series file is empty: "
                                + file
                );
            }

            if (dataRecordCount == 0) {
                throw new IllegalArgumentException(
                        "Numeric delimited time-series file contains "
                                + "no data records: "
                                + file
                );
            }

            double[][] result =
                    new double[dimensions.length][];

            int expectedLength =
                    -1;

            for (int dimension = 0;
                 dimension < dimensions.length;
                 dimension++) {

                result[dimension] =
                        dimensions[dimension].toArray();

                if (expectedLength < 0) {
                    expectedLength =
                            result[dimension].length;
                } else if (result[dimension].length
                        != expectedLength) {

                    throw new IllegalStateException(
                            "Numeric parsing produced inconsistent "
                                    + "dimension lengths in file "
                                    + file
                                    + ". Expected "
                                    + expectedLength
                                    + " values but dimension "
                                    + dimension
                                    + " contains "
                                    + result[dimension].length
                                    + "."
                    );
                }
            }

            return result;
        }

        private static boolean isBlank(
                char[] buffer,
                int offset,
                int length
        ) {
            if (length == 0) {
                return true;
            }

            int end =
                    offset + length;

            for (int index = offset;
                 index < end;
                 index++) {

                if (!Character.isWhitespace(buffer[index])) {
                    return false;
                }
            }

            return true;
        }
    }

    /**
     * Small growable primitive buffer used once per selected dimension.
     */
    private static final class PrimitiveDoubleBuffer {

        private double[] values;
        private int size;

        private PrimitiveDoubleBuffer(
                int initialCapacity
        ) {
            values =
                    new double[
                            Math.max(
                                    1,
                                    initialCapacity
                            )
                            ];
        }

        private void add(
                double value
        ) {
            ensureCapacity(
                    size + 1
            );

            values[size++] =
                    value;
        }

        private double[] toArray() {
            if (size == values.length) {
                return values;
            }

            return Arrays.copyOf(
                    values,
                    size
            );
        }

        private void ensureCapacity(
                int requiredCapacity
        ) {
            if (requiredCapacity <= values.length) {
                return;
            }

            int currentCapacity =
                    values.length;

            int expandedCapacity =
                    currentCapacity <= Integer.MAX_VALUE / 2
                            ? currentCapacity << 1
                            : Integer.MAX_VALUE;

            if (expandedCapacity < requiredCapacity) {
                expandedCapacity =
                        requiredCapacity;
            }

            if (expandedCapacity < 0
                    || expandedCapacity < currentCapacity) {

                throw new OutOfMemoryError(
                        "Required numeric series buffer is too large."
                );
            }

            values =
                    Arrays.copyOf(
                            values,
                            expandedCapacity
                    );
        }
    }
}