package datasets.readers;

import ch.randelshofer.fastdoubleparser.JavaDoubleParser;
import core.AppContext;
import datasets.readers.lazy.LazySeriesReader;
import datasets.readers.lazy.LazySeriesRef;
import dev.hardwood.InputFile;
import dev.hardwood.reader.ParquetFileReader;
import dev.hardwood.reader.RowReader;
import dev.hardwood.schema.ColumnProjection;
import preprocessing.standardization.StandardizationStats;
import preprocessing.standardization.Standardizer;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Locale;
import java.util.Set;

/**
 * Resolves one per-file Parquet time series.
 *
 * <p>Storage convention:</p>
 *
 * <pre>
 * one Parquet file = one PFGAP instance
 * one Parquet record = one time point
 * one projected feature column = one time-series dimension
 * </pre>
 *
 * <p>Physical Parquet record order is preserved. The configured time column
 * is retained for configuration compatibility but is not projected merely
 * for sorting.</p>
 *
 * <p>Numeric flat-column data is delegated to
 * {@link NumericPerFileParquetSeriesReader}, which uses Hardwood's
 * batch-oriented column API and primitive arrays.</p>
 *
 * <p>Nonnumeric and generic data uses Hardwood's row API as a compatibility
 * path. That path supports logical values and generic objects while avoiding
 * one intermediate row object per Parquet record.</p>
 *
 * <p>Returned representations:</p>
 *
 * <pre>
 * numeric, one feature, no missing:
 *     double[time]
 *
 * numeric, multiple features, no missing:
 *     double[dimension][time]
 *
 * numeric, one feature, missing:
 *     Double[time]
 *
 * numeric, multiple features, missing:
 *     Double[dimension][time]
 *
 * generic, one feature:
 *     Object[time]
 *
 * generic, multiple features:
 *     Object[dimension][time]
 * </pre>
 */
public class PerFileParquetSeriesReader
        implements LazySeriesReader {

    private static final int DEFAULT_INITIAL_TIME_CAPACITY =
            1024;

    private final String timeColumn;
    private final List<String> featureColumns;
    private final boolean isNumeric;
    private final boolean hasMissingValues;
    private final StandardizationStats standardizationStats;
    private final int initialTimeCapacity;

    /**
     * Projection used only by the generic row-based fallback.
     *
     * <p>The numeric path constructs its own feature-only column projection
     * through NumericPerFileParquetSeriesReader.</p>
     */
    private final ColumnProjection genericProjection;

    private final Set<String> missingIndicators;

    public PerFileParquetSeriesReader(
            String timeColumn,
            List<String> featureColumns,
            boolean isNumeric,
            boolean hasMissingValues,
            StandardizationStats standardizationStats
    ) {
        this(
                timeColumn,
                featureColumns,
                isNumeric,
                hasMissingValues,
                standardizationStats,
                DEFAULT_INITIAL_TIME_CAPACITY
        );
    }

    public PerFileParquetSeriesReader(
            String timeColumn,
            List<String> featureColumns,
            boolean isNumeric,
            boolean hasMissingValues
    ) {
        this(
                timeColumn,
                featureColumns,
                isNumeric,
                hasMissingValues,
                null,
                DEFAULT_INITIAL_TIME_CAPACITY
        );
    }

    /**
     * Full constructor.
     *
     * @param timeColumn optional descriptive time column
     * @param featureColumns projected feature columns
     * @param isNumeric whether features are numerical
     * @param hasMissingValues whether null feature values are permitted
     * @param standardizationStats optional standardization statistics
     * @param initialTimeCapacity initial allocation hint per feature
     */
    public PerFileParquetSeriesReader(
            String timeColumn,
            List<String> featureColumns,
            boolean isNumeric,
            boolean hasMissingValues,
            StandardizationStats standardizationStats,
            int initialTimeCapacity
    ) {
        this.timeColumn =
                normalizeNullableString(
                        timeColumn
                );

        this.featureColumns =
                copyAndValidateFeatureColumns(
                        featureColumns
                );

        this.isNumeric =
                isNumeric;

        this.hasMissingValues =
                hasMissingValues;

        this.standardizationStats =
                standardizationStats;

        if (initialTimeCapacity < 1) {
            throw new IllegalArgumentException(
                    "PerFileParquetSeriesReader initialTimeCapacity "
                            + "must be at least 1. Received: "
                            + initialTimeCapacity
                            + "."
            );
        }

        this.initialTimeCapacity =
                initialTimeCapacity;

        this.missingIndicators =
                snapshotMissingIndicators();

        validateConfiguration();

        this.genericProjection =
                buildGenericProjection();
    }

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
                    "Failed to lazily read Parquet time series from: "
                            + file,
                    e
            );
        }
    }

    /**
     * Reads one per-file Parquet time series directly.
     *
     * @param file Parquet file
     * @return materialized series
     * @throws IOException if the file cannot be read
     */
    public Object readFile(
            Path file
    ) throws IOException {
        return readFileInternal(
                file,
                true
        );
    }

    private Object readFileInternal(
            Path file,
            boolean validateFileMetadata
    ) throws IOException {
        if (file == null) {
            throw new IllegalArgumentException(
                    "PerFileParquetSeriesReader requires a non-null file."
            );
        }

        if (validateFileMetadata) {
            validateFile(
                    file
            );
        }

        /*
         * Numeric flat-column data uses Hardwood's batch-oriented column API.
         * Physical Parquet record order is retained.
         */
        if (isNumeric) {
            NumericPerFileParquetSeriesReader numericReader =
                    new NumericPerFileParquetSeriesReader(
                            timeColumn,
                            featureColumns,
                            hasMissingValues,
                            standardizationStats,
                            initialTimeCapacity,
                            NumericPerFileParquetSeriesReader
                                    .TimeOrderPolicy
                                    .FILE_ORDER
                    );

            /*
             * Avoid repeating filesystem metadata checks for a LazySeriesRef.
             * The public readFile path has already validated the file.
             *
             * NumericPerFileParquetSeriesReader.readFile performs its own
             * validation, so the current public API still incurs that small
             * check here. A future package-private trusted read method can
             * remove the duplicate check without changing public behavior.
             */
            return numericReader.readFile(
                    file
            );
        }

        return readGenericFile(
                file
        );
    }

    /**
     * Reads nonnumeric or generic flat feature columns.
     *
     * <p>This compatibility path uses RowReader because it performs Hardwood's
     * logical-type conversions and supports values that do not naturally map
     * onto one primitive numerical accessor.</p>
     *
     * <p>Values are appended directly into one growable buffer per feature.
     * No PerFileParquetRow objects or row-level Object arrays are retained.</p>
     */
    private Object readGenericFile(
            Path file
    ) throws IOException {

        ObjectBuffer[] featureBuffers =
                new ObjectBuffer[featureColumns.size()];

        for (int dimension = 0;
             dimension < featureBuffers.length;
             dimension++) {

            featureBuffers[dimension] =
                    new ObjectBuffer(
                            initialTimeCapacity
                    );
        }

        int rowCount =
                0;

        try (ParquetFileReader fileReader =
                     ParquetFileReader.open(
                             InputFile.of(
                                     file
                             )
                     );

             RowReader reader =
                     fileReader.buildRowReader()
                             .projection(
                                     genericProjection
                             )
                             .build()) {

            while (reader.hasNext()) {
                reader.next();

                for (int dimension = 0;
                     dimension < featureColumns.size();
                     dimension++) {

                    String featureColumn =
                            featureColumns.get(
                                    dimension
                            );

                    Object value =
                            readGenericValue(
                                    reader,
                                    featureColumn,
                                    file,
                                    rowCount
                            );

                    featureBuffers[dimension].add(
                            value
                    );
                }

                rowCount++;
            }
        } catch (IllegalArgumentException e) {
            throw e;
        } catch (RuntimeException e) {
            throw new IOException(
                    "Failed while reading generic Parquet columns from: "
                            + file,
                    e
            );
        }

        if (rowCount == 0) {
            throw new IOException(
                    "Per-file Parquet time-series file contains no records: "
                            + file
            );
        }

        Object series =
                materializeGenericSeries(
                        featureBuffers,
                        rowCount,
                        file
                );

        /*
         * Standardization is prohibited for nonnumeric readers by constructor
         * validation, so no transformation is performed here.
         */
        return series;
    }

    private Object readGenericValue(
            RowReader reader,
            String featureColumn,
            Path file,
            int rowIndex
    ) {
        if (reader.isNull(featureColumn)) {
            if (!hasMissingValues) {
                throw new IllegalArgumentException(
                        "Encountered null in feature column '"
                                + featureColumn
                                + "' in file "
                                + file
                                + " at row "
                                + rowIndex
                                + ", but hasMissingValues=false."
                );
            }

            return null;
        }

        Object rawValue =
                reader.getValue(
                        featureColumn
                );

        Object parsedValue =
                parseGenericValue(
                        rawValue
                );

        if (parsedValue == null
                && !hasMissingValues) {

            throw new IllegalArgumentException(
                    "Encountered a configured missing value in feature "
                            + "column '"
                            + featureColumn
                            + "' in file "
                            + file
                            + " at row "
                            + rowIndex
                            + ", but hasMissingValues=false."
            );
        }

        return parsedValue;
    }

    private Object materializeGenericSeries(
            ObjectBuffer[] featureBuffers,
            int expectedLength,
            Path file
    ) {
        for (int dimension = 0;
             dimension < featureBuffers.length;
             dimension++) {

            if (featureBuffers[dimension].size()
                    != expectedLength) {

                throw new IllegalStateException(
                        "Generic Parquet feature column '"
                                + featureColumns.get(dimension)
                                + "' produced "
                                + featureBuffers[dimension].size()
                                + " values, but "
                                + expectedLength
                                + " records were expected in file "
                                + file
                                + "."
                );
            }
        }

        if (featureBuffers.length == 1) {
            return featureBuffers[0].toArray();
        }

        Object[][] result =
                new Object[featureBuffers.length][];

        for (int dimension = 0;
             dimension < featureBuffers.length;
             dimension++) {

            result[dimension] =
                    featureBuffers[dimension].toArray();
        }

        return result;
    }

    private Object parseGenericValue(
            Object value
    ) {
        if (value == null) {
            return null;
        }

        /*
         * Preserve values already materialized by Hardwood's logical-type
         * converters. Dates, timestamps, decimals, UUIDs, Booleans, and other
         * typed objects should not be converted through String unless they
         * actually arrive as strings.
         */
        if (!(value instanceof CharSequence)) {
            return value;
        }

        String trimmed =
                value.toString()
                        .trim();

        if (isMissingToken(
                trimmed
        )) {
            return null;
        }

        /*
         * Preserve the prior generic inference behavior for Parquet string
         * columns that contain textual numeric or Boolean values.
         */
        try {
            return JavaDoubleParser.parseDouble(
                    trimmed
            );
        } catch (NumberFormatException ignored) {
            if (trimmed.equalsIgnoreCase("true")
                    || trimmed.equalsIgnoreCase("false")) {

                return Boolean.parseBoolean(
                        trimmed
                );
            }

            return trimmed;
        }
    }

    private ColumnProjection buildGenericProjection() {
        /*
         * Physical record order is assumed correct, so the time column is not
         * projected merely for sorting.
         */
        return ColumnProjection.columns(
                featureColumns.toArray(
                        String[]::new
                )
        );
    }

    private void validateConfiguration() {
        if (timeColumn != null
                && featureColumns.contains(timeColumn)) {

            throw new IllegalArgumentException(
                    "The configured time column cannot also be selected "
                            + "as a feature column: "
                            + timeColumn
            );
        }

        if (standardizationStats != null) {
            if (!isNumeric) {
                throw new IllegalArgumentException(
                        "Standardization statistics cannot be applied by "
                                + "PerFileParquetSeriesReader when "
                                + "isNumeric=false."
                );
            }

            standardizationStats.validateFeatureCompatibility(
                    featureColumns
            );
        }
    }

    private void validateFile(
            Path file
    ) throws IOException {
        if (!Files.exists(file)) {
            throw new IOException(
                    "Per-file Parquet time-series file does not exist: "
                            + file
            );
        }

        if (!Files.isRegularFile(file)) {
            throw new IOException(
                    "Per-file Parquet time-series path is not a regular "
                            + "file: "
                            + file
            );
        }

        if (!Files.isReadable(file)) {
            throw new IOException(
                    "Per-file Parquet time-series file is not readable: "
                            + file
            );
        }
    }

    private boolean isMissingToken(
            String token
    ) {
        if (token == null) {
            return true;
        }

        String trimmed =
                token.trim();

        if (trimmed.isEmpty()) {
            return true;
        }

        return missingIndicators.contains(
                trimmed.toUpperCase(
                        Locale.ROOT
                )
        );
    }

    private static Set<String> snapshotMissingIndicators() {
        if (AppContext.MissingStrings == null
                || AppContext.MissingStrings.isEmpty()) {

            return Set.of();
        }

        Set<String> normalized =
                new HashSet<>();

        for (String indicator : AppContext.MissingStrings) {
            if (indicator == null) {
                continue;
            }

            String trimmed =
                    indicator.trim();

            if (!trimmed.isEmpty()) {
                normalized.add(
                        trimmed.toUpperCase(
                                Locale.ROOT
                        )
                );
            }
        }

        if (normalized.isEmpty()) {
            return Set.of();
        }

        return Collections.unmodifiableSet(
                normalized
        );
    }

    private static List<String> copyAndValidateFeatureColumns(
            List<String> featureColumns
    ) {
        if (featureColumns == null || featureColumns.isEmpty()) {
            throw new IllegalArgumentException(
                    "PerFileParquetSeriesReader requires at least one "
                            + "feature column."
            );
        }

        List<String> copy =
                new ArrayList<>(
                        featureColumns.size()
                );

        Set<String> used =
                new HashSet<>();

        for (String featureColumn : featureColumns) {
            if (featureColumn == null
                    || featureColumn.isBlank()) {

                throw new IllegalArgumentException(
                        "Parquet feature-column names cannot be null "
                                + "or blank."
                );
            }

            String normalized =
                    featureColumn.trim();

            if (!used.add(normalized)) {
                throw new IllegalArgumentException(
                        "Parquet feature column was selected more than once: "
                                + normalized
                );
            }

            copy.add(
                    normalized
            );
        }

        return List.copyOf(
                copy
        );
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

    private static int nextCapacity(
            int currentCapacity,
            int requiredCapacity
    ) {
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
                    "Required generic Parquet series buffer is too large."
            );
        }

        return expandedCapacity;
    }

    private static final class ObjectBuffer {

        private Object[] values;
        private int size;

        private ObjectBuffer(
                int initialCapacity
        ) {
            values =
                    new Object[
                            Math.max(
                                    1,
                                    initialCapacity
                            )
                            ];
        }

        private void add(
                Object value
        ) {
            ensureCapacity(
                    size + 1
            );

            values[size++] =
                    value;
        }

        private int size() {
            return size;
        }

        private Object[] toArray() {
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

            values =
                    Arrays.copyOf(
                            values,
                            nextCapacity(
                                    values.length,
                                    requiredCapacity
                            )
                    );
        }
    }
}