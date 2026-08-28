package datasets.readers;

import datasets.readers.lazy.LazySeriesReader;
import datasets.readers.lazy.LazySeriesRef;
import dev.hardwood.InputFile;
import dev.hardwood.reader.ColumnReader;
import dev.hardwood.reader.ColumnReaders;
import dev.hardwood.reader.ParquetFileReader;
import dev.hardwood.reader.Validity;
import dev.hardwood.schema.ColumnProjection;
import preprocessing.standardization.StandardizationStats;
import preprocessing.standardization.Standardizer;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

/**
 * High-throughput numeric reader for one time series stored in one
 * Parquet file.
 *
 * <p>Expected physical organization:</p>
 *
 * <pre>
 * one Parquet record = one time point
 * one projected feature column = one time-series dimension
 * one Parquet file = one PFGAP instance
 * </pre>
 *
 * <p>Example univariate file:</p>
 *
 * <pre>
 * time,value
 * 0.0,1.2
 * 1.0,1.4
 * 2.0,1.1
 * </pre>
 *
 * <p>Example multivariate file:</p>
 *
 * <pre>
 * time,x,y,z
 * 0.0,1.2,3.4,5.6
 * 1.0,1.4,3.1,5.8
 * </pre>
 *
 * <p>Output representation:</p>
 *
 * <pre>
 * one feature, no missing:
 *     double[time]
 *
 * multiple features, no missing:
 *     double[dimension][time]
 *
 * one feature, missing enabled:
 *     Double[time]
 *
 * multiple features, missing enabled:
 *     Double[dimension][time]
 * </pre>
 *
 * <p>This reader uses Hardwood's batch-oriented {@link ColumnReaders} API.
 * Projected numerical columns are decoded into primitive arrays and copied
 * directly into dimension-major primitive buffers. It does not construct one
 * row object or one feature array per time point.</p>
 *
 * <p>The current optimized implementation requires all projected feature
 * columns to have Parquet physical type DOUBLE because it uses
 * {@link ColumnReader#getDoubles()}. When time sorting is enabled, the time
 * column must also have physical type DOUBLE.</p>
 *
 * <p>Use {@link PerFileParquetSeriesReader} for generic, nonnumeric, nested,
 * or otherwise unsupported Parquet schemas.</p>
 */
public class NumericPerFileParquetSeriesReader
        implements LazySeriesReader {

    private static final int DEFAULT_INITIAL_TIME_CAPACITY =
            1024;

    /**
     * Controls how records are ordered in the returned time series.
     */
    public enum TimeOrderPolicy {

        /**
         * Preserve the physical Parquet record order.
         *
         * <p>The configured time column is not projected or decoded. This is
         * the fastest mode and should be preferred when files are already
         * ordered correctly.</p>
         */
        FILE_ORDER,

        /**
         * Project a DOUBLE time column, sort records by that value, and use
         * physical input order as a deterministic tie-breaker.
         */
        SORT_DOUBLE_TIME
    }

    private final String timeColumn;
    private final List<String> featureColumns;
    private final boolean hasMissingValues;
    private final StandardizationStats standardizationStats;
    private final int initialTimeCapacity;
    private final TimeOrderPolicy timeOrderPolicy;
    private final ColumnProjection columnProjection;

    /**
     * Constructs the reader using physical file order.
     *
     * <p>The time column is retained as configuration but is not projected
     * when {@link TimeOrderPolicy#FILE_ORDER} is used.</p>
     */
    public NumericPerFileParquetSeriesReader(
            String timeColumn,
            List<String> featureColumns,
            boolean hasMissingValues,
            StandardizationStats standardizationStats
    ) {
        this(
                timeColumn,
                featureColumns,
                hasMissingValues,
                standardizationStats,
                DEFAULT_INITIAL_TIME_CAPACITY,
                TimeOrderPolicy.FILE_ORDER
        );
    }

    public NumericPerFileParquetSeriesReader(
            String timeColumn,
            List<String> featureColumns,
            boolean hasMissingValues
    ) {
        this(
                timeColumn,
                featureColumns,
                hasMissingValues,
                null,
                DEFAULT_INITIAL_TIME_CAPACITY,
                TimeOrderPolicy.FILE_ORDER
        );
    }

    /**
     * Full constructor.
     *
     * @param timeColumn          optional time column
     * @param featureColumns      projected DOUBLE feature columns
     * @param hasMissingValues    whether null feature values are permitted
     * @param standardizationStats optional standardization statistics
     * @param initialTimeCapacity initial allocation hint per dimension
     * @param timeOrderPolicy     physical-order or DOUBLE-time sorting policy
     */
    public NumericPerFileParquetSeriesReader(
            String timeColumn,
            List<String> featureColumns,
            boolean hasMissingValues,
            StandardizationStats standardizationStats,
            int initialTimeCapacity,
            TimeOrderPolicy timeOrderPolicy
    ) {
        this.timeColumn =
                normalizeNullableString(
                        timeColumn
                );

        this.featureColumns =
                copyAndValidateFeatureColumns(
                        featureColumns
                );

        this.hasMissingValues =
                hasMissingValues;

        this.standardizationStats =
                standardizationStats;

        if (initialTimeCapacity < 1) {
            throw new IllegalArgumentException(
                    "NumericPerFileParquetSeriesReader "
                            + "initialTimeCapacity must be at least 1. "
                            + "Received: "
                            + initialTimeCapacity
                            + "."
            );
        }

        this.initialTimeCapacity =
                initialTimeCapacity;

        this.timeOrderPolicy =
                timeOrderPolicy == null
                        ? TimeOrderPolicy.FILE_ORDER
                        : timeOrderPolicy;

        if (this.timeOrderPolicy
                == TimeOrderPolicy.SORT_DOUBLE_TIME
                && this.timeColumn == null) {

            throw new IllegalArgumentException(
                    "SORT_DOUBLE_TIME requires a nonempty timeColumn."
            );
        }

        validateStandardizationConfiguration();

        this.columnProjection =
                buildColumnProjection();
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
                    "Failed to lazily read numeric Parquet time series "
                            + "from: "
                            + file,
                    e
            );
        }
    }

    /**
     * Reads one numeric per-file Parquet time series directly.
     *
     * @param file Parquet file
     * @return materialized time series
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
                    "NumericPerFileParquetSeriesReader requires "
                            + "a non-null file."
            );
        }

        if (validateFileMetadata) {
            validateFile(
                    file
            );
        }

        int dimensionCount =
                featureColumns.size();

        PrimitiveDoubleBuffer[] featureBuffers =
                new PrimitiveDoubleBuffer[dimensionCount];

        MissingBuffer[] missingBuffers =
                hasMissingValues
                        ? new MissingBuffer[dimensionCount]
                        : null;

        for (int dimension = 0;
             dimension < dimensionCount;
             dimension++) {

            featureBuffers[dimension] =
                    new PrimitiveDoubleBuffer(
                            initialTimeCapacity
                    );

            if (hasMissingValues) {
                missingBuffers[dimension] =
                        new MissingBuffer(
                                initialTimeCapacity
                        );
            }
        }

        PrimitiveDoubleBuffer timeValues =
                timeOrderPolicy
                        == TimeOrderPolicy.SORT_DOUBLE_TIME
                        ? new PrimitiveDoubleBuffer(
                        initialTimeCapacity
                )
                        : null;

        MissingBuffer timeMissing =
                timeOrderPolicy
                        == TimeOrderPolicy.SORT_DOUBLE_TIME
                        ? new MissingBuffer(
                        initialTimeCapacity
                )
                        : null;

        IntBuffer inputOrders =
                timeOrderPolicy
                        == TimeOrderPolicy.SORT_DOUBLE_TIME
                        ? new IntBuffer(
                        initialTimeCapacity
                )
                        : null;

        int totalRecordCount =
                0;

        try (ParquetFileReader fileReader =
                     ParquetFileReader.open(
                             InputFile.of(
                                     file
                             )
                     );

             ColumnReaders columns =
                     fileReader.buildColumnReaders(
                                     columnProjection
                             )
                             .build()) {

            int timeColumnOffset =
                    timeOrderPolicy
                            == TimeOrderPolicy.SORT_DOUBLE_TIME
                            ? 1
                            : 0;

            while (columns.nextBatch()) {
                int batchRecordCount =
                        columns.getRecordCount();

                if (batchRecordCount == 0) {
                    continue;
                }

                if (timeOrderPolicy
                        == TimeOrderPolicy.SORT_DOUBLE_TIME) {

                    ColumnReader timeReader =
                            columns.getColumnReader(
                                    0
                            );

                    appendTimeBatch(
                            timeReader,
                            batchRecordCount,
                            timeValues,
                            timeMissing,
                            inputOrders,
                            totalRecordCount,
                            file
                    );
                }

                for (int dimension = 0;
                     dimension < dimensionCount;
                     dimension++) {

                    ColumnReader featureReader =
                            columns.getColumnReader(
                                    timeColumnOffset + dimension
                            );

                    appendFeatureBatch(
                            featureReader,
                            batchRecordCount,
                            featureBuffers[dimension],
                            hasMissingValues
                                    ? missingBuffers[dimension]
                                    : null,
                            featureColumns.get(dimension),
                            totalRecordCount,
                            file
                    );
                }

                totalRecordCount +=
                        batchRecordCount;
            }
        } catch (IllegalArgumentException e) {
            throw e;
        } catch (RuntimeException e) {
            throw new IOException(
                    "Failed while decoding numeric Parquet columns from: "
                            + file,
                    e
            );
        }

        if (totalRecordCount == 0) {
            throw new IOException(
                    "Numeric Parquet time-series file contains no records: "
                            + file
            );
        }

        validateBufferLengths(
                featureBuffers,
                missingBuffers,
                totalRecordCount,
                file
        );

        if (timeOrderPolicy
                == TimeOrderPolicy.SORT_DOUBLE_TIME) {

            sortByTime(
                    featureBuffers,
                    missingBuffers,
                    timeValues,
                    timeMissing,
                    inputOrders,
                    file
            );
        }

        Object series =
                materializeSeries(
                        featureBuffers,
                        missingBuffers
                );

        if (standardizationStats != null) {
            Standardizer.transformInstanceInPlace(
                    series,
                    standardizationStats
            );
        }

        return series;
    }

    /**
     * Appends one batch from a projected DOUBLE feature column.
     *
     * <p>Hardwood exposes flat column batches as primitive arrays plus a
     * validity bitmap. Hoisting {@code hasNulls()} outside the inner loop
     * preserves the no-null fast path.</p>
     */
    private void appendFeatureBatch(
            ColumnReader columnReader,
            int batchRecordCount,
            PrimitiveDoubleBuffer valuesBuffer,
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
                    "NumericPerFileParquetSeriesReader currently requires "
                            + "feature column '"
                            + columnName
                            + "' to have Parquet physical type DOUBLE in file "
                            + file
                            + ".",
                    e
            );
        }

        if (values.length < batchRecordCount) {
            throw new IllegalStateException(
                    "Hardwood returned fewer primitive values than records "
                            + "for feature column '"
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
            valuesBuffer.addAll(
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
            for (int index = 0;
                 index < batchRecordCount;
                 index++) {

                if (validity.isNull(index)) {
                    throw new IllegalArgumentException(
                            "Encountered null in feature column '"
                                    + columnName
                                    + "' in file "
                                    + file
                                    + " at record "
                                    + (recordOffset + index)
                                    + ", but hasMissingValues=false."
                    );
                }
            }
        }

        for (int index = 0;
             index < batchRecordCount;
             index++) {

            boolean missing =
                    validity.isNull(index);

            valuesBuffer.add(
                    missing
                            ? 0.0
                            : values[index]
            );

            if (missingBuffer != null) {
                missingBuffer.add(
                        missing
                );
            }
        }
    }

    private void appendTimeBatch(
            ColumnReader timeReader,
            int batchRecordCount,
            PrimitiveDoubleBuffer timeValues,
            MissingBuffer timeMissing,
            IntBuffer inputOrders,
            int startingRecordIndex,
            Path file
    ) {
        double[] values;

        try {
            values =
                    timeReader.getDoubles();
        } catch (RuntimeException e) {
            throw new IllegalArgumentException(
                    "SORT_DOUBLE_TIME currently requires time column '"
                            + timeColumn
                            + "' to have Parquet physical type DOUBLE in file "
                            + file
                            + ".",
                    e
            );
        }

        if (values.length < batchRecordCount) {
            throw new IllegalStateException(
                    "Hardwood returned fewer primitive time values than "
                            + "records for column '"
                            + timeColumn
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
                timeReader.getLeafValidity();

        boolean batchHasNulls =
                validity.hasNulls();

        for (int index = 0;
             index < batchRecordCount;
             index++) {

            boolean missing =
                    batchHasNulls
                            && validity.isNull(index);

            timeValues.add(
                    missing
                            ? 0.0
                            : values[index]
            );

            timeMissing.add(
                    missing
            );

            inputOrders.add(
                    startingRecordIndex + index
            );
        }
    }

    private void sortByTime(
            PrimitiveDoubleBuffer[] featureBuffers,
            MissingBuffer[] missingBuffers,
            PrimitiveDoubleBuffer timeValues,
            MissingBuffer timeMissing,
            IntBuffer inputOrders,
            Path file
    ) {
        int size =
                timeValues.size();

        if (size < 2) {
            return;
        }

        if (timeMissing.size() != size
                || inputOrders.size() != size) {

            throw new IllegalStateException(
                    "Time-order buffers have inconsistent lengths for file "
                            + file
                            + "."
            );
        }

        Integer[] order =
                new Integer[size];

        for (int index = 0;
             index < size;
             index++) {

            order[index] =
                    index;
        }

        Arrays.sort(
                order,
                Comparator
                        .comparing(
                                (Integer index) ->
                                        new TimeSortKey(
                                                timeMissing.get(index),
                                                timeValues.get(index)
                                        )
                        )
                        .thenComparingInt(
                                inputOrders::get
                        )
        );

        boolean alreadySorted =
                true;

        for (int index = 0;
             index < order.length;
             index++) {

            if (order[index] != index) {
                alreadySorted =
                        false;

                break;
            }
        }

        if (alreadySorted) {
            return;
        }

        for (PrimitiveDoubleBuffer buffer : featureBuffers) {
            buffer.reorder(
                    order
            );
        }

        if (missingBuffers != null) {
            for (MissingBuffer buffer : missingBuffers) {
                buffer.reorder(
                        order
                );
            }
        }
    }

    private void validateBufferLengths(
            PrimitiveDoubleBuffer[] featureBuffers,
            MissingBuffer[] missingBuffers,
            int expectedLength,
            Path file
    ) {
        for (int dimension = 0;
             dimension < featureBuffers.length;
             dimension++) {

            int actualLength =
                    featureBuffers[dimension].size();

            if (actualLength != expectedLength) {
                throw new IllegalStateException(
                        "Feature column '"
                                + featureColumns.get(dimension)
                                + "' produced "
                                + actualLength
                                + " values, but "
                                + expectedLength
                                + " records were expected in file "
                                + file
                                + "."
                );
            }

            if (missingBuffers != null
                    && missingBuffers[dimension].size()
                    != expectedLength) {

                throw new IllegalStateException(
                        "Missing-position buffer for feature column '"
                                + featureColumns.get(dimension)
                                + "' has an inconsistent length in file "
                                + file
                                + "."
                );
            }
        }
    }

    private Object materializeSeries(
            PrimitiveDoubleBuffer[] featureBuffers,
            MissingBuffer[] missingBuffers
    ) {
        if (featureBuffers.length == 1) {
            if (!hasMissingValues) {
                return featureBuffers[0].toArray();
            }

            return featureBuffers[0].toNullableBoxedArray(
                    missingBuffers[0]
            );
        }

        if (!hasMissingValues) {
            double[][] result =
                    new double[featureBuffers.length][];

            for (int dimension = 0;
                 dimension < featureBuffers.length;
                 dimension++) {

                result[dimension] =
                        featureBuffers[dimension].toArray();
            }

            return result;
        }

        Double[][] result =
                new Double[featureBuffers.length][];

        for (int dimension = 0;
             dimension < featureBuffers.length;
             dimension++) {

            result[dimension] =
                    featureBuffers[dimension]
                            .toNullableBoxedArray(
                                    missingBuffers[dimension]
                            );
        }

        return result;
    }

    private ColumnProjection buildColumnProjection() {
        List<String> columns =
                new ArrayList<>();

        if (timeOrderPolicy
                == TimeOrderPolicy.SORT_DOUBLE_TIME) {

            columns.add(
                    timeColumn
            );
        }

        columns.addAll(
                featureColumns
        );

        return ColumnProjection.columns(
                columns.toArray(
                        String[]::new
                )
        );
    }

    private void validateStandardizationConfiguration() {
        if (standardizationStats == null) {
            return;
        }

        standardizationStats.validateFeatureCompatibility(
                featureColumns
        );
    }

    private void validateFile(
            Path file
    ) throws IOException {
        if (!Files.exists(file)) {
            throw new IOException(
                    "Numeric Parquet time-series file does not exist: "
                            + file
            );
        }

        if (!Files.isRegularFile(file)) {
            throw new IOException(
                    "Numeric Parquet time-series path is not a regular file: "
                            + file
            );
        }

        if (!Files.isReadable(file)) {
            throw new IOException(
                    "Numeric Parquet time-series file is not readable: "
                            + file
            );
        }
    }

    private static List<String> copyAndValidateFeatureColumns(
            List<String> featureColumns
    ) {
        if (featureColumns == null || featureColumns.isEmpty()) {
            throw new IllegalArgumentException(
                    "NumericPerFileParquetSeriesReader requires at least "
                            + "one feature column."
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
                        "Numeric Parquet feature-column names cannot be "
                                + "null or blank."
                );
            }

            String normalized =
                    featureColumn.trim();

            if (!used.add(normalized)) {
                throw new IllegalArgumentException(
                        "Numeric Parquet feature column was selected more "
                                + "than once: "
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
                    "Required numeric Parquet series buffer is too large."
            );
        }

        return expandedCapacity;
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

        private double get(
                int index
        ) {
            return values[index];
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
                MissingBuffer missingBuffer
        ) {
            if (missingBuffer.size() != size) {
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
                        missingBuffer.get(index)
                                ? null
                                : values[index];
            }

            return result;
        }

        private void reorder(
                Integer[] order
        ) {
            double[] reordered =
                    new double[size];

            for (int outputIndex = 0;
                 outputIndex < order.length;
                 outputIndex++) {

                reordered[outputIndex] =
                        values[order[outputIndex]];
            }

            values =
                    reordered;
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

        private void reorder(
                Integer[] order
        ) {
            boolean[] reordered =
                    new boolean[size];

            for (int outputIndex = 0;
                 outputIndex < order.length;
                 outputIndex++) {

                reordered[outputIndex] =
                        values[order[outputIndex]];
            }

            values =
                    reordered;
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

    private static final class IntBuffer {

        private int[] values;
        private int size;

        private IntBuffer(
                int initialCapacity
        ) {
            values =
                    new int[
                            Math.max(
                                    1,
                                    initialCapacity
                            )
                            ];
        }

        private void add(
                int value
        ) {
            ensureCapacity(
                    size + 1
            );

            values[size++] =
                    value;
        }

        private int get(
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

    private static final class TimeSortKey
            implements Comparable<TimeSortKey> {

        private final boolean missing;
        private final double value;

        private TimeSortKey(
                boolean missing,
                double value
        ) {
            this.missing =
                    missing;

            this.value =
                    value;
        }

        @Override
        public int compareTo(
                TimeSortKey other
        ) {
            if (missing && other.missing) {
                return 0;
            }

            if (missing) {
                return -1;
            }

            if (other.missing) {
                return 1;
            }

            return Double.compare(
                    value,
                    other.value
            );
        }
    }
}