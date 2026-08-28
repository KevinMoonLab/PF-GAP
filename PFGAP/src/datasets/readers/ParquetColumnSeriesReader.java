package datasets.readers;

import ch.randelshofer.fastdoubleparser.JavaDoubleParser;
import core.AppContext;
import datasets.ListObjectDataset;
import dev.hardwood.InputFile;
import dev.hardwood.reader.ColumnReader;
import dev.hardwood.reader.ColumnReaders;
import dev.hardwood.reader.RowReader;
import dev.hardwood.reader.Validity;
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
 * Reads one ordinary Parquet file as a collection of univariate series.
 *
 * <p>Dataset convention:</p>
 *
 * <pre>
 * one selected Parquet column = one PFGAP dataset instance
 * one Parquet record = one position in that univariate instance
 * </pre>
 *
 * <p>For example, this Parquet table:</p>
 *
 * <pre>
 * series_0   series_1   series_2
 * 1.0        4.0        7.0
 * 2.0        5.0        8.0
 * 3.0        6.0        9.0
 * </pre>
 *
 * <p>produces three PFGAP instances:</p>
 *
 * <pre>
 * [1.0, 2.0, 3.0]
 * [4.0, 5.0, 6.0]
 * [7.0, 8.0, 9.0]
 * </pre>
 *
 * <p>Numeric flat DOUBLE columns use Hardwood's batch-oriented
 * {@link ColumnReaders} API. This avoids row-by-row access, generic object
 * materialization, and boxing when missing values are absent.</p>
 *
 * <p>Nonnumeric columns use Hardwood's {@link RowReader} API as a
 * compatibility path because it performs logical-type conversion and
 * supports generic values.</p>
 *
 * <p>Returned representations:</p>
 *
 * <pre>
 * numeric column, no missing values:
 *     double[]
 *
 * numeric column, missing values enabled:
 *     Double[]
 *
 * generic column:
 *     Object[]
 * </pre>
 *
 * <p>Labels are currently {@code null}. A label column in this file would
 * contain one value per Parquet record rather than one label per selected
 * series, so it does not naturally satisfy the one-column-per-instance
 * convention. Per-series labels should be supplied through separate metadata
 * or another explicit label mechanism.</p>
 *
 * <p>For multivariate time-series Parquet data, use a long-format reader, a
 * per-file reader, or a user-defined reader.</p>
 */
public class ParquetColumnSeriesReader
        implements DatasetReader {

    private static final int DEFAULT_INITIAL_SERIES_CAPACITY =
            4096;

    private final String dataFileName;
    private final List<String> featureColumns;
    private final boolean isNumeric;
    private final boolean hasMissingValues;
    private final StandardizationStats standardizationStats;
    private final int initialSeriesCapacity;

    private final ColumnProjection projection;
    private final Set<String> missingIndicators;

    /**
     * Constructs the reader from PFGAP reader options.
     */
    public ParquetColumnSeriesReader(
            ReaderOptions options
    ) {
        this(
                requireOptions(options).getDataPath(),
                options.getFeatureColumns(),
                options.isNumeric(),
                options.hasMissingValues(),
                options.getStandardizationStats(),
                DEFAULT_INITIAL_SERIES_CAPACITY
        );

        if (options.getLabelColumns() != null
                && !options.getLabelColumns().isEmpty()) {

            throw new IllegalArgumentException(
                    "ParquetFileReader does not interpret ordinary Parquet "
                            + "columns as per-series labels. Each selected "
                            + "feature column is itself one dataset instance. "
                            + "Use separate metadata or another reader for "
                            + "labels."
            );
        }
    }

    /**
     * Constructs the reader with the default initial series capacity.
     */
    public ParquetColumnSeriesReader(
            String dataFileName,
            List<String> featureColumns,
            boolean isNumeric,
            boolean hasMissingValues,
            StandardizationStats standardizationStats
    ) {
        this(
                dataFileName,
                featureColumns,
                isNumeric,
                hasMissingValues,
                standardizationStats,
                DEFAULT_INITIAL_SERIES_CAPACITY
        );
    }

    /**
     * Full constructor.
     *
     * @param dataFileName          Parquet file
     * @param featureColumns        columns that become dataset instances
     * @param isNumeric             whether selected columns are numerical
     * @param hasMissingValues      whether null or configured missing values
     *                              are permitted
     * @param standardizationStats  optional standardization statistics
     * @param initialSeriesCapacity initial allocation hint per selected
     *                              column
     */
    public ParquetColumnSeriesReader(
            String dataFileName,
            List<String> featureColumns,
            boolean isNumeric,
            boolean hasMissingValues,
            StandardizationStats standardizationStats,
            int initialSeriesCapacity
    ) {
        this.dataFileName =
                requireNonblank(
                        dataFileName,
                        "dataFileName"
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

        if (initialSeriesCapacity < 1) {
            throw new IllegalArgumentException(
                    "ParquetFileReader initialSeriesCapacity must be "
                            + "at least 1. Received: "
                            + initialSeriesCapacity
                            + "."
            );
        }

        this.initialSeriesCapacity =
                initialSeriesCapacity;

        this.missingIndicators =
                snapshotMissingIndicators();

        validateStandardizationConfiguration();

        this.projection =
                ColumnProjection.columns(
                        this.featureColumns.toArray(
                                String[]::new
                        )
                );
    }

    @Override
    public ListObjectDataset read()
            throws IOException {

        Path file =
                validateFile();

        if (isNumeric) {
            return readNumericColumns(
                    file
            );
        }

        return readGenericColumns(
                file
        );
    }

    /**
     * Reads selected physical DOUBLE columns in batches.
     */
    private ListObjectDataset readNumericColumns(
            Path file
    ) throws IOException {

        int columnCount =
                featureColumns.size();

        PrimitiveDoubleBuffer[] valueBuffers =
                new PrimitiveDoubleBuffer[columnCount];

        MissingBuffer[] missingBuffers =
                hasMissingValues
                        ? new MissingBuffer[columnCount]
                        : null;

        for (int columnIndex = 0;
             columnIndex < columnCount;
             columnIndex++) {

            valueBuffers[columnIndex] =
                    new PrimitiveDoubleBuffer(
                            initialSeriesCapacity
                    );

            if (hasMissingValues) {
                missingBuffers[columnIndex] =
                        new MissingBuffer(
                                initialSeriesCapacity
                        );
            }
        }

        int recordCount =
                0;

        /*
         * The fully qualified class name avoids a naming collision with this
         * PFGAP reader.
         */
        try (dev.hardwood.reader.ParquetFileReader hardwoodReader =
                     dev.hardwood.reader.ParquetFileReader.open(
                             InputFile.of(
                                     file
                             )
                     );

             ColumnReaders columns =
                     hardwoodReader.buildColumnReaders(
                                     projection
                             )
                             .build()) {

            while (columns.nextBatch()) {
                int batchRecordCount =
                        columns.getRecordCount();

                if (batchRecordCount == 0) {
                    continue;
                }

                for (int columnIndex = 0;
                     columnIndex < columnCount;
                     columnIndex++) {

                    ColumnReader columnReader =
                            columns.getColumnReader(
                                    columnIndex
                            );

                    appendNumericBatch(
                            columnReader,
                            batchRecordCount,
                            valueBuffers[columnIndex],
                            hasMissingValues
                                    ? missingBuffers[columnIndex]
                                    : null,
                            featureColumns.get(columnIndex),
                            recordCount,
                            file
                    );
                }

                recordCount +=
                        batchRecordCount;
            }
        } catch (IllegalArgumentException e) {
            throw e;
        } catch (RuntimeException e) {
            throw new IOException(
                    "Failed while reading numeric Parquet columns from: "
                            + file,
                    e
            );
        }

        if (recordCount == 0) {
            throw new IOException(
                    "Parquet file contains no records: "
                            + file
            );
        }

        ListObjectDataset dataset =
                new ListObjectDataset(
                        columnCount
                );

        for (int columnIndex = 0;
             columnIndex < columnCount;
             columnIndex++) {

            if (valueBuffers[columnIndex].size()
                    != recordCount) {

                throw new IllegalStateException(
                        "Parquet column '"
                                + featureColumns.get(columnIndex)
                                + "' produced "
                                + valueBuffers[columnIndex].size()
                                + " values, but "
                                + recordCount
                                + " records were expected."
                );
            }

            Object series;

            if (hasMissingValues) {
                if (missingBuffers[columnIndex].size()
                        != recordCount) {

                    throw new IllegalStateException(
                            "Missing-position buffer for Parquet column '"
                                    + featureColumns.get(columnIndex)
                                    + "' has an inconsistent length."
                    );
                }

                series =
                        valueBuffers[columnIndex]
                                .toNullableBoxedArray(
                                        missingBuffers[columnIndex]
                                );
            } else {
                series =
                        valueBuffers[columnIndex].toArray();
            }

            if (standardizationStats != null) {
                Standardizer.transformInstanceInPlace(
                        series,
                        standardizationStats
                );
            }

            dataset.add(
                    null,
                    series,
                    columnIndex
            );
        }

        dataset.setLength(
                recordCount
        );

        AppContext.length =
                recordCount;

        return dataset;
    }

    /**
     * Appends one Hardwood primitive DOUBLE batch.
     *
     * <p>Hardwood provides one validity object per batch. Hoisting
     * {@code hasNulls()} preserves a bulk-copy fast path for the common
     * no-null case.</p>
     */
    private void appendNumericBatch(
            ColumnReader columnReader,
            int batchRecordCount,
            PrimitiveDoubleBuffer valueBuffer,
            MissingBuffer missingBuffer,
            String columnName,
            int recordOffset,
            Path file
    ) {
        double[] values;

        try {
            values =
                    columnReader.getDoubles();
        } catch (RuntimeException e) {
            throw new IllegalArgumentException(
                    "ParquetFileReader currently requires numeric column '"
                            + columnName
                            + "' to have Parquet physical type DOUBLE in file "
                            + file
                            + ". Integer, float, decimal, and other physical "
                            + "types require explicit type dispatch.",
                    e
            );
        }

        if (values.length < batchRecordCount) {
            throw new IllegalStateException(
                    "Hardwood returned fewer primitive values than records "
                            + "for column '"
                            + columnName
                            + "' in file "
                            + file
                            + ". Records="
                            + batchRecordCount
                            + ", values="
                            + values.length
                            + "."
            );
        }

        Validity validity =
                columnReader.getLeafValidity();

        boolean batchHasNulls =
                validity.hasNulls();

        if (!batchHasNulls) {
            valueBuffer.addAll(
                    values,
                    0,
                    batchRecordCount
            );

            if (missingBuffer != null) {
                missingBuffer.addRepeated(
                        false,
                        batchRecordCount
                );
            }

            return;
        }

        if (!hasMissingValues) {
            for (int recordIndex = 0;
                 recordIndex < batchRecordCount;
                 recordIndex++) {

                if (validity.isNull(recordIndex)) {
                    throw new IllegalArgumentException(
                            "Encountered null in Parquet column '"
                                    + columnName
                                    + "' in file "
                                    + file
                                    + " at record "
                                    + (recordOffset + recordIndex)
                                    + ", but hasMissingValues=false."
                    );
                }
            }
        }

        for (int recordIndex = 0;
             recordIndex < batchRecordCount;
             recordIndex++) {

            boolean missing =
                    validity.isNull(
                            recordIndex
                    );

            valueBuffer.add(
                    missing
                            ? 0.0
                            : values[recordIndex]
            );

            if (missingBuffer != null) {
                missingBuffer.add(
                        missing
                );
            }
        }
    }

    /**
     * Reads generic columns through Hardwood's row API.
     *
     * <p>This remains the compatibility path for strings, Booleans, logical
     * dates or timestamps, decimals, and other values that cannot be safely
     * consumed through {@code getDoubles()}.</p>
     */
    private ListObjectDataset readGenericColumns(
            Path file
    ) throws IOException {

        ObjectBuffer[] buffers =
                new ObjectBuffer[featureColumns.size()];

        for (int columnIndex = 0;
             columnIndex < buffers.length;
             columnIndex++) {

            buffers[columnIndex] =
                    new ObjectBuffer(
                            initialSeriesCapacity
                    );
        }

        int recordCount =
                0;

        try (dev.hardwood.reader.ParquetFileReader hardwoodReader =
                     dev.hardwood.reader.ParquetFileReader.open(
                             InputFile.of(
                                     file
                             )
                     );

             RowReader rowReader =
                     hardwoodReader.buildRowReader()
                             .projection(
                                     projection
                             )
                             .build()) {

            while (rowReader.hasNext()) {
                rowReader.next();

                for (int columnIndex = 0;
                     columnIndex < featureColumns.size();
                     columnIndex++) {

                    String columnName =
                            featureColumns.get(
                                    columnIndex
                            );

                    Object value;

                    if (rowReader.isNull(columnName)) {
                        if (!hasMissingValues) {
                            throw new IllegalArgumentException(
                                    "Encountered null in Parquet column '"
                                            + columnName
                                            + "' in file "
                                            + file
                                            + " at record "
                                            + recordCount
                                            + ", but hasMissingValues=false."
                            );
                        }

                        value =
                                null;
                    } else {
                        value =
                                parseGenericValue(
                                        rowReader.getValue(
                                                columnName
                                        )
                                );

                        if (value == null
                                && !hasMissingValues) {

                            throw new IllegalArgumentException(
                                    "Encountered a configured missing value "
                                            + "in Parquet column '"
                                            + columnName
                                            + "' in file "
                                            + file
                                            + " at record "
                                            + recordCount
                                            + ", but hasMissingValues=false."
                            );
                        }
                    }

                    buffers[columnIndex].add(
                            value
                    );
                }

                recordCount++;
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

        if (recordCount == 0) {
            throw new IOException(
                    "Parquet file contains no records: "
                            + file
            );
        }

        ListObjectDataset dataset =
                new ListObjectDataset(
                        featureColumns.size()
                );

        for (int columnIndex = 0;
             columnIndex < buffers.length;
             columnIndex++) {

            if (buffers[columnIndex].size()
                    != recordCount) {

                throw new IllegalStateException(
                        "Generic Parquet column '"
                                + featureColumns.get(columnIndex)
                                + "' produced an inconsistent number "
                                + "of values."
                );
            }

            dataset.add(
                    null,
                    buffers[columnIndex].toArray(),
                    columnIndex
            );
        }

        dataset.setLength(
                recordCount
        );

        AppContext.length =
                recordCount;

        return dataset;
    }

    private Object parseGenericValue(
            Object value
    ) {
        if (value == null) {
            return null;
        }

        /*
         * Preserve objects already converted by Hardwood's logical-type
         * handling, including dates, timestamps, decimals, UUIDs, and
         * Booleans.
         */
        if (!(value instanceof CharSequence)) {
            return value;
        }

        String token =
                value.toString()
                        .trim();

        if (isMissingToken(token)) {
            return null;
        }

        try {
            return JavaDoubleParser.parseDouble(
                    token
            );
        } catch (NumberFormatException ignored) {
            if (token.equalsIgnoreCase("true")
                    || token.equalsIgnoreCase("false")) {

                return Boolean.parseBoolean(
                        token
                );
            }

            return token;
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

    /**
     * Validates standardization for the column-as-instance convention.
     *
     * <p>Each selected Parquet column becomes one univariate instance, so
     * standardization statistics must describe exactly one input dimension.
     * Feature-name-based compatibility across all selected columns is not
     * meaningful here because the column names identify instances rather than
     * dimensions.</p>
     */
    private void validateStandardizationConfiguration() {
        if (standardizationStats == null) {
            return;
        }

        if (!isNumeric) {
            throw new IllegalArgumentException(
                    "Standardization statistics cannot be applied by "
                            + "ParquetFileReader when isNumeric=false."
            );
        }

        /*
         * We deliberately do not call:
         *
         * standardizationStats.validateFeatureCompatibility(featureColumns)
         *
         * because featureColumns are dataset instances under this reader's
         * convention, not dimensions within one instance. Standardizer will
         * validate the realized univariate representation when applied.
         */
    }

    private Path validateFile()
            throws IOException {

        Path file =
                Path.of(
                        dataFileName
                );

        if (!Files.exists(file)) {
            throw new IOException(
                    "Parquet data file does not exist: "
                            + file
            );
        }

        if (!Files.isRegularFile(file)) {
            throw new IOException(
                    "Parquet data path is not a regular file: "
                            + file
            );
        }

        if (!Files.isReadable(file)) {
            throw new IOException(
                    "Parquet data file is not readable: "
                            + file
            );
        }

        return file;
    }

    private static ReaderOptions requireOptions(
            ReaderOptions options
    ) {
        if (options == null) {
            throw new IllegalArgumentException(
                    "ParquetFileReader requires non-null ReaderOptions."
            );
        }

        return options;
    }

    private static String requireNonblank(
            String value,
            String argumentName
    ) {
        if (value == null || value.isBlank()) {
            throw new IllegalArgumentException(
                    "ParquetFileReader requires "
                            + argumentName
                            + "."
            );
        }

        return value.trim();
    }

    private static List<String> copyAndValidateFeatureColumns(
            List<String> featureColumns
    ) {
        if (featureColumns == null || featureColumns.isEmpty()) {
            throw new IllegalArgumentException(
                    "ParquetFileReader requires at least one selected "
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

    private static int nextCapacity(
            int currentCapacity,
            int requiredCapacity
    ) {
        int expanded =
                currentCapacity <= Integer.MAX_VALUE / 2
                        ? currentCapacity << 1
                        : Integer.MAX_VALUE;

        if (expanded < requiredCapacity) {
            expanded =
                    requiredCapacity;
        }

        if (expanded < 0
                || expanded < currentCapacity) {

            throw new OutOfMemoryError(
                    "Required Parquet series buffer is too large."
            );
        }

        return expanded;
    }

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

        private void addAll(
                double[] source,
                int offset,
                int length
        ) {
            if (length == 0) {
                return;
            }

            ensureCapacity(
                    size + length
            );

            System.arraycopy(
                    source,
                    offset,
                    values,
                    size,
                    length
            );

            size +=
                    length;
        }

        private int size() {
            return size;
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

        private Double[] toNullableBoxedArray(
                MissingBuffer missing
        ) {
            if (missing.size() != size) {
                throw new IllegalStateException(
                        "Numeric and missing-position buffers have "
                                + "different lengths."
                );
            }

            Double[] result =
                    new Double[size];

            for (int index = 0;
                 index < size;
                 index++) {

                result[index] =
                        missing.get(index)
                                ? null
                                : values[index];
            }

            return result;
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

    private static final class MissingBuffer {

        private boolean[] values;
        private int size;

        private MissingBuffer(
                int initialCapacity
        ) {
            values =
                    new boolean[
                            Math.max(
                                    1,
                                    initialCapacity
                            )
                            ];
        }

        private void add(
                boolean value
        ) {
            ensureCapacity(
                    size + 1
            );

            values[size++] =
                    value;
        }

        private void addRepeated(
                boolean value,
                int count
        ) {
            if (count == 0) {
                return;
            }

            ensureCapacity(
                    size + count
            );

            if (value) {
                Arrays.fill(
                        values,
                        size,
                        size + count,
                        true
                );
            }

            size +=
                    count;
        }

        private boolean get(
                int index
        ) {
            return values[index];
        }

        private int size() {
            return size;
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