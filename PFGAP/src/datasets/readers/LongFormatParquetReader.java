package datasets.readers;

import core.AppContext;
import datasets.ListObjectDataset;
import dev.hardwood.InputFile;
import dev.hardwood.reader.ParquetFileReader;
import dev.hardwood.reader.RowReader;
//import dev.hardwood.schema.ColumnSchema;
//import dev.hardwood.schema.FileSchema;
import dev.hardwood.schema.ColumnProjection;
import org.apache.commons.lang3.time.DurationFormatUtils;

import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.stream.Collectors;

/**
 * Reader for long-format Parquet time-series data.
 *
 * Long-format data has one row per time point, not one row per instance.
 *
 * Example logical table:
 *
 *      id,time,temp,pressure,label
 *      A,0,10.1,100.0,class1
 *      A,1,10.4,101.2,class1
 *      A,2,10.2,100.8,class1
 *      B,0,5.2,80.1,class2
 *      B,1,5.4,81.0,class2
 *
 * Rows are grouped by idColumn. Within each group, rows are sorted by
 * timeColumn if supplied. Otherwise, input row order is preserved.
 *
 * Output shape:
 *
 * If featureColumns.size() == 1:
 *
 *      numeric, no missing:       double[]
 *      numeric, with missing:     Double[]
 *      generic:                   Object[]
 *
 * If featureColumns.size() > 1:
 *
 *      numeric, no missing:       double[][]
 *      numeric, with missing:     Double[][]
 *      generic:                   Object[][]
 *
 * For multivariate output, matrix orientation is:
 *
 *      feature x time
 *
 * Labels:
 *
 *      - If labelColumns is empty, label is null.
 *      - If labelColumns has one column, label is a scalar Object.
 *      - If labelColumns has multiple columns, label is List<Object>.
 *
 * Uses Hardwood for Parquet reading. Hardwood is a lightweight Java
 * implementation of the Parquet format that does not require Hadoop.
 */
public class LongFormatParquetReader implements DatasetReader {

    private final String dataFileName;
    private final boolean isNumeric;
    private final boolean hasMissingValues;
    private final boolean isRegression;

    private final String idColumn;
    private final String timeColumn;
    private final List<String> featureColumns;
    private final List<String> labelColumns;

    public LongFormatParquetReader(ReaderOptions options) {

        this(
                options.getDataPath(),
                options.isNumeric(),
                options.hasMissingValues(),
                options.isRegression(),
                options.getIdColumn(),
                options.getTimeColumn(),
                options.getFeatureColumns(),
                options.getLabelColumns()
        );
    }

    public LongFormatParquetReader(
            String dataFileName,
            boolean isNumeric,
            boolean hasMissingValues,
            boolean isRegression,
            String idColumn,
            String timeColumn,
            List<String> featureColumns,
            List<String> labelColumns
    ) {
        this.dataFileName = dataFileName;
        this.isNumeric = isNumeric;
        this.hasMissingValues = hasMissingValues;
        this.isRegression = isRegression;
        this.idColumn = idColumn;
        this.timeColumn = timeColumn;

        this.featureColumns =
                featureColumns == null
                        ? new ArrayList<>()
                        : new ArrayList<>(featureColumns);

        this.labelColumns =
                labelColumns == null
                        ? new ArrayList<>()
                        : new ArrayList<>(labelColumns);
    }

    @Override
    public ListObjectDataset read() throws IOException {
        validateOptions();

        if (idColumn == null || idColumn.trim().isEmpty()) {
            return readRowWiseDataset();
        }

        return readGroupedLongFormatDataset();
    }

    public ListObjectDataset readGroupedLongFormatDataset() throws IOException {

        long start = System.nanoTime();

        Map<Object, List<LongParquetRow>> groupedRows =
                new LinkedHashMap<>();

        Path path = Paths.get(dataFileName);

        //try (ParquetFileReader fileReader =
        //             ParquetFileReader.open(InputFile.of(path));
        //     RowReader reader = fileReader.rowReader()) {

        try (ParquetFileReader fileReader =
                     ParquetFileReader.open(InputFile.of(path));
             RowReader reader =
                     fileReader.buildRowReader()
                             .projection(buildColumnProjection())
                             .build()) {

            int rowNumber = 0;

            while (reader.hasNext()) {
                reader.next();

                Object id = normalizeValue(
                        getValue(reader, idColumn)
                );

                if (id == null) {
                    throw new IllegalArgumentException(
                            "Encountered null id value in column: " + idColumn
                    );
                }

                Object timeValue = null;

                if (timeColumn != null && !timeColumn.trim().isEmpty()) {
                    timeValue = normalizeValue(
                            getValue(reader, timeColumn)
                    );
                }

                Object[] featureValues =
                        new Object[featureColumns.size()];

                for (int j = 0; j < featureColumns.size(); j++) {

                    String featureColumn = featureColumns.get(j);

                    Object rawValue =
                            normalizeValue(
                                    getValue(reader, featureColumn)
                            );

                    featureValues[j] =
                            parseFeatureValue(
                                    rawValue,
                                    featureColumn
                            );
                }

                Object label = parseLabelValues(reader);

                LongParquetRow row =
                        new LongParquetRow(
                                id,
                                timeValue,
                                featureValues,
                                label,
                                rowNumber
                        );

                groupedRows
                        .computeIfAbsent(
                                id,
                                ignored -> new ArrayList<>()
                        )
                        .add(row);

                ProgressLogger.logProgress(rowNumber);
                rowNumber++;
            }
        }

        ListObjectDataset dataset =
                buildDataset(groupedRows);

        long end = System.nanoTime();

        ProgressLogger.logDuration(start, end);

        return dataset;
    }

    private ColumnProjection buildColumnProjection() {

        List<String> columns = new ArrayList<>();

        if (idColumn != null &&
                !idColumn.trim().isEmpty()) {

            columns.add(idColumn);
        }

        if (timeColumn != null &&
                !timeColumn.trim().isEmpty()) {

            columns.add(timeColumn);
        }

        columns.addAll(featureColumns);
        columns.addAll(labelColumns);

        return ColumnProjection.columns(
                columns.toArray(new String[0])
        );
    }

    private ListObjectDataset readRowWiseDataset()
            throws IOException {

        long start = System.nanoTime();

        ListObjectDataset dataset =
                new ListObjectDataset();

        Path path = Paths.get(dataFileName);

        //try (ParquetFileReader fileReader =
        //             ParquetFileReader.open(InputFile.of(path));
        //     RowReader reader = fileReader.rowReader()) {

        try (ParquetFileReader fileReader =
                     ParquetFileReader.open(InputFile.of(path));
             RowReader reader =
                     fileReader.buildRowReader()
                             .projection(buildColumnProjection())
                             .build()) {

            int rowIndex = 0;

            while (reader.hasNext()) {

                reader.next();

                Object[] featureValues =
                        new Object[featureColumns.size()];

                for (int j = 0;
                     j < featureColumns.size();
                     j++) {

                    String featureColumn =
                            featureColumns.get(j);

                    Object rawValue =
                            normalizeValue(
                                    getValue(reader, featureColumn)
                            );

                    featureValues[j] =
                            parseFeatureValue(
                                    rawValue,
                                    featureColumn
                            );
                }

                Object label =
                        parseLabelValues(reader);

                Object data =
                        buildRowWiseData(featureValues);

                dataset.add(
                        label,
                        data,
                        rowIndex
                );

                updateGlobalLength(data);

                ProgressLogger.logProgress(rowIndex);
                rowIndex++;
            }
        }

        long end = System.nanoTime();

        ProgressLogger.logDuration(start, end);

        return dataset;
    }

    /**
     * Retrieve a Parquet field through Hardwood's typed RowReader API.
     *
     * We deliberately return Object here because PF-GAP supports several
     * kinds of feature/label values and the existing reader normalizes
     * them afterward.
     */
    private static Object getValue(
            RowReader reader,
            String columnName
    ) {
        if (reader.isNull(columnName)) {
            return null;
        }

        return reader.getValue(columnName);
    }

    private ListObjectDataset buildDataset(
            Map<Object, List<LongParquetRow>> groupedRows
    ) {
        ListObjectDataset dataset =
                new ListObjectDataset();

        int instanceIndex = 0;

        for (Map.Entry<Object, List<LongParquetRow>> entry :
                groupedRows.entrySet()) {

            List<LongParquetRow> rows =
                    entry.getValue();

            sortRows(rows);

            Object label =
                    inferGroupLabel(rows);

            Object data =
                    buildSeriesData(rows);

            dataset.add(
                    label,
                    data,
                    instanceIndex
            );

            updateGlobalLength(data);

            instanceIndex++;
        }

        return dataset;
    }

    private Object buildRowWiseData(
            Object[] featureValues
    ) {
        int length = featureValues.length;

        if (isNumeric) {

            if (hasMissingValues) {

                Double[] data =
                        new Double[length];

                for (int i = 0; i < length; i++) {
                    data[i] =
                            toBoxedDouble(
                                    featureValues[i]
                            );
                }

                return data;
            }

            double[] data =
                    new double[length];

            for (int i = 0; i < length; i++) {
                data[i] =
                        toPrimitiveDouble(
                                featureValues[i]
                        );
            }

            return data;
        }

        Object[] data =
                new Object[length];

        System.arraycopy(
                featureValues,
                0,
                data,
                0,
                length
        );

        return data;
    }

    private void sortRows(
            List<LongParquetRow> rows
    ) {
        if (timeColumn == null ||
                timeColumn.trim().isEmpty()) {

            rows.sort(
                    Comparator.comparingInt(
                            row -> row.inputOrder
                    )
            );

            return;
        }

        rows.sort(
                (a, b) ->
                        compareTimeValues(
                                a.timeValue,
                                b.timeValue
                        )
        );
    }

    @SuppressWarnings({"rawtypes", "unchecked"})
    private int compareTimeValues(
            Object a,
            Object b
    ) {
        if (a == null && b == null) {
            return 0;
        }

        if (a == null) {
            return -1;
        }

        if (b == null) {
            return 1;
        }

        if (a instanceof Number &&
                b instanceof Number) {

            return Double.compare(
                    ((Number) a).doubleValue(),
                    ((Number) b).doubleValue()
            );
        }

        if (a instanceof Comparable &&
                a.getClass().isInstance(b)) {

            return ((Comparable) a).compareTo(b);
        }

        return a.toString()
                .compareTo(b.toString());
    }

    private Object inferGroupLabel(
            List<LongParquetRow> rows
    ) {
        if (rows.isEmpty()) {
            return null;
        }

        Object firstLabel =
                rows.get(0).label;

        for (LongParquetRow row : rows) {

            if (!labelsEqual(
                    firstLabel,
                    row.label
            )) {

                throw new IllegalArgumentException(
                        "Inconsistent labels found within "
                                + "long-format Parquet group for id: "
                                + row.id
                );
            }
        }

        return firstLabel;
    }

    private boolean labelsEqual(
            Object a,
            Object b
    ) {
        if (a == null && b == null) {
            return true;
        }

        if (a == null || b == null) {
            return false;
        }

        return a.equals(b);
    }

    private Object buildSeriesData(
            List<LongParquetRow> rows
    ) {
        int timeLength =
                rows.size();

        int dimensionCount =
                featureColumns.size();

        if (dimensionCount == 1) {
            return buildUnivariateSeries(
                    rows,
                    timeLength
            );
        }

        return buildMultivariateSeries(
                rows,
                dimensionCount,
                timeLength
        );
    }

    private Object buildUnivariateSeries(
            List<LongParquetRow> rows,
            int timeLength
    ) {
        if (isNumeric) {

            if (hasMissingValues) {

                Double[] data =
                        new Double[timeLength];

                for (int t = 0; t < timeLength; t++) {

                    data[t] =
                            toBoxedDouble(
                                    rows.get(t)
                                            .featureValues[0]
                            );
                }

                return data;
            }

            double[] data =
                    new double[timeLength];

            for (int t = 0; t < timeLength; t++) {

                data[t] =
                        toPrimitiveDouble(
                                rows.get(t)
                                        .featureValues[0]
                        );
            }

            return data;
        }

        Object[] data =
                new Object[timeLength];

        for (int t = 0; t < timeLength; t++) {

            data[t] =
                    rows.get(t)
                            .featureValues[0];
        }

        return data;
    }

    private Object buildMultivariateSeries(
            List<LongParquetRow> rows,
            int dimensionCount,
            int timeLength
    ) {
        if (isNumeric) {

            if (hasMissingValues) {

                Double[][] data =
                        new Double[
                                dimensionCount
                                ][timeLength];

                for (int t = 0; t < timeLength; t++) {

                    LongParquetRow row =
                            rows.get(t);

                    for (int d = 0;
                         d < dimensionCount;
                         d++) {

                        data[d][t] =
                                toBoxedDouble(
                                        row.featureValues[d]
                                );
                    }
                }

                return data;
            }

            double[][] data =
                    new double[
                            dimensionCount
                            ][timeLength];

            for (int t = 0; t < timeLength; t++) {

                LongParquetRow row =
                        rows.get(t);

                for (int d = 0;
                     d < dimensionCount;
                     d++) {

                    data[d][t] =
                            toPrimitiveDouble(
                                    row.featureValues[d]
                            );
                }
            }

            return data;
        }

        Object[][] data =
                new Object[
                        dimensionCount
                        ][timeLength];

        for (int t = 0; t < timeLength; t++) {

            LongParquetRow row =
                    rows.get(t);

            for (int d = 0;
                 d < dimensionCount;
                 d++) {

                data[d][t] =
                        row.featureValues[d];
            }
        }

        return data;
    }

    private Object parseLabelValues(
            RowReader reader
    ) {
        if (labelColumns.isEmpty()) {
            return null;
        }

        if (labelColumns.size() == 1) {

            return parseLabelValue(
                    normalizeValue(
                            getValue(
                                    reader,
                                    labelColumns.get(0)
                            )
                    )
            );
        }

        List<Object> labels =
                new ArrayList<>();

        for (String labelColumn :
                labelColumns) {

            labels.add(
                    parseLabelValue(
                            normalizeValue(
                                    getValue(
                                            reader,
                                            labelColumn
                                    )
                            )
                    )
            );
        }

        return labels;
    }

    private Object parseFeatureValue(
            Object value,
            String featureColumn
    ) {
        if (value == null) {

            if (hasMissingValues) {
                return null;
            }

            throw new IllegalArgumentException(
                    "Encountered null value in feature column '"
                            + featureColumn
                            + "', but hasMissingValues=false."
            );
        }

        if (isNumeric) {
            return toBoxedDouble(value);
        }

        return parseGenericValue(value);
    }

    private Object parseLabelValue(
            Object value
    ) {
        if (value == null) {
            return null;
        }

        if (isRegression) {
            return toPrimitiveDouble(value);
        }

        if (value instanceof Integer) {
            return value;
        }

        if (value instanceof Long) {

            long longValue =
                    (Long) value;

            if (longValue >= Integer.MIN_VALUE &&
                    longValue <= Integer.MAX_VALUE) {

                return (int) longValue;
            }

            return longValue;
        }

        if (value instanceof Number) {

            double doubleValue =
                    ((Number) value).doubleValue();

            if (doubleValue ==
                    Math.rint(doubleValue)
                    && doubleValue >= Integer.MIN_VALUE
                    && doubleValue <= Integer.MAX_VALUE) {

                return (int) doubleValue;
            }

            return doubleValue;
        }

        String trimmed =
                value.toString().trim();

        if (MissingValueParser.isMissing(trimmed)) {
            return null;
        }

        try {
            return Integer.parseInt(trimmed);

        } catch (NumberFormatException ignored) {

            try {

                double parsed =
                        Double.parseDouble(trimmed);

                if (parsed == Math.rint(parsed)
                        && parsed >= Integer.MIN_VALUE
                        && parsed <= Integer.MAX_VALUE) {

                    return (int) parsed;
                }

                return parsed;

            } catch (NumberFormatException ignoredAgain) {
                return trimmed;
            }
        }
    }

    private static Object parseGenericValue(
            Object value
    ) {
        if (value == null) {
            return null;
        }

        Object normalized =
                normalizeValue(value);

        if (normalized == null) {
            return null;
        }

        if (normalized instanceof String) {

            String trimmed =
                    normalized.toString().trim();

            if (MissingValueParser.isMissing(trimmed)) {
                return null;
            }

            try {
                return Double.parseDouble(trimmed);

            } catch (NumberFormatException e1) {

                if (trimmed.equalsIgnoreCase("true")
                        || trimmed.equalsIgnoreCase("false")) {

                    return Boolean.parseBoolean(trimmed);
                }

                return trimmed;
            }
        }

        return normalized;
    }

    /**
     * Hardwood already returns ordinary Java values, so no Avro-specific
     * normalization is normally required.
     */
    private static Object normalizeValue(
            Object value
    ) {
        return value;
    }

    private static Double toBoxedDouble(
            Object value
    ) {
        if (value == null) {
            return null;
        }

        Object normalized =
                normalizeValue(value);

        if (normalized == null) {
            return null;
        }

        if (normalized instanceof Number) {

            return ((Number) normalized)
                    .doubleValue();
        }

        String trimmed =
                normalized.toString().trim();

        if (MissingValueParser.isMissing(trimmed)) {
            return null;
        }

        return Double.valueOf(trimmed);
    }

    private static double toPrimitiveDouble(
            Object value
    ) {
        if (value == null) {

            throw new IllegalArgumentException(
                    "Encountered null numeric value, "
                            + "but hasMissingValues=false."
            );
        }

        Object normalized =
                normalizeValue(value);

        if (normalized instanceof Number) {

            return ((Number) normalized)
                    .doubleValue();
        }

        String trimmed =
                normalized.toString().trim();

        if (MissingValueParser.isMissing(trimmed)) {

            throw new IllegalArgumentException(
                    "Encountered missing numeric value, "
                            + "but hasMissingValues=false."
            );
        }

        return Double.parseDouble(trimmed);
    }

    private void validateOptions() {

        if (dataFileName == null ||
                dataFileName.trim().isEmpty()) {

            throw new IllegalArgumentException(
                    "LongFormatParquetReader requires dataFileName."
            );
        }

        if (featureColumns == null ||
                featureColumns.isEmpty()) {

            throw new IllegalArgumentException(
                    "LongFormatParquetReader requires "
                            + "at least one feature column."
            );
        }
    }

    private static void updateGlobalLength(
            Object data
    ) {
        if (data instanceof double[]) {

            AppContext.length =
                    ((double[]) data).length;

        } else if (data instanceof Double[]) {

            AppContext.length =
                    ((Double[]) data).length;

        } else if (data instanceof double[][]) {

            double[][] matrix =
                    (double[][]) data;

            if (matrix.length > 0) {
                AppContext.length =
                        matrix[0].length;
            }

        } else if (data instanceof Double[][]) {

            Double[][] matrix =
                    (Double[][]) data;

            if (matrix.length > 0) {
                AppContext.length =
                        matrix[0].length;
            }

        } else if (data instanceof Object[][]) {

            Object[][] matrix =
                    (Object[][]) data;

            if (matrix.length > 0) {
                AppContext.length =
                        matrix[0].length;
            }

        } else if (data instanceof Object[]) {

            AppContext.length =
                    ((Object[]) data).length;
        }
    }

    private static class LongParquetRow {

        private final Object id;
        private final Object timeValue;
        private final Object[] featureValues;
        private final Object label;
        private final int inputOrder;

        private LongParquetRow(
                Object id,
                Object timeValue,
                Object[] featureValues,
                Object label,
                int inputOrder
        ) {
            this.id = id;
            this.timeValue = timeValue;
            this.featureValues = featureValues;
            this.label = label;
            this.inputOrder = inputOrder;
        }
    }

    private static class MissingValueParser {

        private static Set<String> missingIndicators =
                AppContext.MissingStrings;

        public static void setMissingIndicators(
                Set<String> indicators
        ) {
            missingIndicators =
                    indicators.stream()
                            .map(String::toUpperCase)
                            .collect(Collectors.toSet());
        }

        public static boolean isMissing(
                String token
        ) {
            if (token == null) {
                return true;
            }

            return missingIndicators.contains(
                    token.trim().toUpperCase()
            );
        }
    }

    public static class ProgressLogger {

        public static void logProgress(int i) {

            if (i % 1000 == 0) {

                if (i % 100000 == 0) {

                    System.out.print("\n");

                    if (i % 1000000 == 0) {

                        long usedMem =
                                AppContext.runtime.totalMemory()
                                        - AppContext.runtime.freeMemory();

                        System.out.print(
                                i
                                        + ":"
                                        + usedMem / 1024 / 1024
                                        + "mb\n"
                        );
                    }

                } else {
                    System.out.print(".");
                }
            }
        }

        public static void logDuration(
                long start,
                long end
        ) {
            long elapsed =
                    end - start;

            String timeDuration =
                    DurationFormatUtils.formatDuration(
                            (long) (elapsed / 1e6),
                            "H:m:s.SSS"
                    );

            System.out.println(
                    "finished in "
                            + timeDuration
            );
        }
    }
}