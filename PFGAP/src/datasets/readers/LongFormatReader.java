package datasets.readers;

import ch.randelshofer.fastdoubleparser.JavaDoubleParser;
import core.AppContext;
import datasets.ListObjectDataset;
import de.siegmar.fastcsv.reader.CsvReader;
import de.siegmar.fastcsv.reader.CsvRecord;
import org.apache.commons.lang3.time.DurationFormatUtils;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;

/**
 * General reader for long-format delimited time-series data.
 *
 * <p>Long-format data has one CSV record per time point rather than one
 * record per complete time-series instance.</p>
 *
 * <p>Example:</p>
 *
 * <pre>
 * id,time,temp,pressure,label
 * A,0,10.1,100.0,class1
 * A,1,10.4,101.2,class1
 * A,2,10.2,100.8,class1
 * B,0,5.2,80.1,class2
 * B,1,5.4,81.0,class2
 * </pre>
 *
 * <p>When {@code idColumn} is configured, records are grouped by ID.
 * Records within each group are sorted by {@code timeColumn} when one is
 * configured. Otherwise, input record order is retained.</p>
 *
 * <p>When {@code idColumn} is absent, each record becomes one ordinary
 * row-wise dataset instance.</p>
 *
 * <p>Output representations:</p>
 *
 * <pre>
 * One feature:
 *     numeric, no missing:   double[]
 *     numeric, missing:      Double[]
 *     generic:               Object[]
 *
 * Multiple features:
 *     numeric, no missing:   double[feature][time]
 *     numeric, missing:      Double[feature][time]
 *     generic:               Object[feature][time]
 * </pre>
 *
 * <p>Labels:</p>
 *
 * <ul>
 *     <li>No label columns: {@code null}</li>
 *     <li>One label column: scalar {@code Object}</li>
 *     <li>Multiple label columns: {@code List<Object>}</li>
 * </ul>
 *
 * <p>This general reader preserves generic and missing-value behavior.
 * It uses FastCSV for structural CSV parsing and JavaDoubleParser for
 * explicitly numeric feature values. A later specialized numeric long-format
 * reader can eliminate the remaining boxed intermediate values and group
 * rows directly into primitive buffers.</p>
 */
public class LongFormatReader
        implements DatasetReader {

    private final String dataFileName;
    private final String entrySeparator;
    private final char fieldSeparator;
    private final boolean hasHeader;
    private final boolean isNumeric;
    private final boolean hasMissingValues;
    private final boolean isRegression;

    private final String idColumn;
    private final String timeColumn;
    private final List<String> featureColumns;
    private final List<String> labelColumns;
    private final Set<String> missingIndicators;

    public LongFormatReader(
            ReaderOptions options
    ) {
        this(
                requireOptions(options).getDataPath(),
                options.getEntrySeparator(),
                options.hasHeader(),
                options.isNumeric(),
                options.hasMissingValues(),
                options.isRegression(),
                options.getIdColumn(),
                options.getTimeColumn(),
                options.getFeatureColumns(),
                options.getLabelColumns()
        );
    }

    public LongFormatReader(
            String dataFileName,
            String entrySeparator,
            boolean hasHeader,
            boolean isNumeric,
            boolean hasMissingValues,
            boolean isRegression,
            String idColumn,
            String timeColumn,
            List<String> featureColumns,
            List<String> labelColumns
    ) {
        this.dataFileName =
                requireNonblank(
                        dataFileName,
                        "dataFileName"
                );

        this.entrySeparator =
                validateAndNormalizeSeparator(
                        entrySeparator
                );

        this.fieldSeparator =
                this.entrySeparator.charAt(0);

        this.hasHeader =
                hasHeader;

        this.isNumeric =
                isNumeric;

        this.hasMissingValues =
                hasMissingValues;

        this.isRegression =
                isRegression;

        this.idColumn =
                normalizeNullableString(
                        idColumn
                );

        this.timeColumn =
                normalizeNullableString(
                        timeColumn
                );

        this.featureColumns =
                copyAndValidateColumns(
                        featureColumns,
                        "featureColumns",
                        false
                );

        this.labelColumns =
                copyAndValidateColumns(
                        labelColumns,
                        "labelColumns",
                        true
                );

        this.missingIndicators =
                snapshotMissingIndicators();

        validateOptions();
    }

    @Override
    public ListObjectDataset read()
            throws IOException {

        if (idColumn == null) {
            return readRowWiseDataset();
        }

        return readGroupedLongFormatDataset();
    }

    /**
     * Reads true long-format data and groups records by the configured ID
     * column.
     */
    public ListObjectDataset readGroupedLongFormatDataset()
            throws IOException {

        if (idColumn == null) {
            throw new IllegalStateException(
                    "Grouped long-format reading requires idColumn."
            );
        }

        long start =
                System.nanoTime();

        Path dataPath =
                validateDataFile();

        Map<Object, List<LongRow>> groupedRows =
                new LinkedHashMap<>();

        try (CsvReader<CsvRecord> csvReader =
                     createCsvReader(
                             dataPath
                     )) {

            var iterator =
                    csvReader.iterator();

            CsvRecord headerRecord =
                    requireHeaderRecord(
                            iterator,
                            dataPath
                    );

            List<String> header =
                    headerRecord.getFields();

            Map<String, Integer> columnIndex =
                    buildColumnIndex(
                            header,
                            dataPath
                    );

            ColumnSelection selection =
                    resolveGroupedSelection(
                            columnIndex,
                            header.size()
                    );

            int inputOrder =
                    0;

            while (iterator.hasNext()) {
                CsvRecord record =
                        iterator.next();

                List<String> fields =
                        record.getFields();

                validateRecordWidth(
                        fields,
                        selection.columnCount,
                        dataPath,
                        record
                );

                Object id =
                        parseGenericValue(
                                getField(
                                        fields,
                                        selection.idIndex
                                )
                        );

                if (id == null) {
                    throw new IllegalArgumentException(
                            "Encountered a missing ID in long-format file "
                                    + dataPath
                                    + " at CSV record beginning on line "
                                    + record.getStartingLineNumber()
                                    + "."
                    );
                }

                Object timeValue =
                        selection.timeIndex < 0
                                ? null
                                : parseGenericValue(
                                getField(
                                        fields,
                                        selection.timeIndex
                                )
                        );

                Object[] featureValues =
                        parseFeatureValues(
                                fields,
                                selection.featureIndices,
                                dataPath,
                                record
                        );

                Object label =
                        parseLabelValues(
                                fields,
                                selection.labelIndices
                        );

                LongRow row =
                        new LongRow(
                                id,
                                timeValue,
                                featureValues,
                                label,
                                inputOrder
                        );

                groupedRows
                        .computeIfAbsent(
                                id,
                                ignored -> new ArrayList<>()
                        )
                        .add(
                                row
                        );

                ProgressLogger.logProgress(
                        inputOrder
                );

                inputOrder++;
            }
        } catch (IllegalArgumentException e) {
            throw e;
        } catch (RuntimeException e) {
            throw new IOException(
                    "Failed while parsing long-format delimited file: "
                            + dataPath,
                    e
            );
        }

        if (groupedRows.isEmpty()) {
            throw new IOException(
                    "Long-format file contains no data records: "
                            + dataPath
            );
        }

        ListObjectDataset dataset =
                buildDataset(
                        groupedRows
                );

        long end =
                System.nanoTime();

        ProgressLogger.logDuration(
                start,
                end
        );

        return dataset;
    }

    /**
     * Reads the configured file in row-wise mode.
     *
     * <p>This mode is selected when {@code idColumn} is absent. Each CSV
     * record becomes one dataset instance.</p>
     */
    private ListObjectDataset readRowWiseDataset()
            throws IOException {

        long start =
                System.nanoTime();

        Path dataPath =
                validateDataFile();

        ListObjectDataset dataset =
                new ListObjectDataset();

        int rowIndex =
                0;

        try (CsvReader<CsvRecord> csvReader =
                     createCsvReader(
                             dataPath
                     )) {

            var iterator =
                    csvReader.iterator();

            CsvRecord headerRecord =
                    requireHeaderRecord(
                            iterator,
                            dataPath
                    );

            List<String> header =
                    headerRecord.getFields();

            Map<String, Integer> columnIndex =
                    buildColumnIndex(
                            header,
                            dataPath
                    );

            ColumnSelection selection =
                    resolveRowWiseSelection(
                            columnIndex,
                            header.size()
                    );

            while (iterator.hasNext()) {
                CsvRecord record =
                        iterator.next();

                List<String> fields =
                        record.getFields();

                validateRecordWidth(
                        fields,
                        selection.columnCount,
                        dataPath,
                        record
                );

                Object[] featureValues =
                        parseFeatureValues(
                                fields,
                                selection.featureIndices,
                                dataPath,
                                record
                        );

                Object label =
                        parseLabelValues(
                                fields,
                                selection.labelIndices
                        );

                Object data =
                        buildRowWiseData(
                                featureValues
                        );

                dataset.add(
                        label,
                        data,
                        rowIndex
                );

                updateGlobalLength(
                        data
                );

                ProgressLogger.logProgress(
                        rowIndex
                );

                rowIndex++;
            }
        } catch (IllegalArgumentException e) {
            throw e;
        } catch (RuntimeException e) {
            throw new IOException(
                    "Failed while parsing row-wise delimited file: "
                            + dataPath,
                    e
            );
        }

        if (rowIndex == 0) {
            throw new IOException(
                    "Row-wise delimited file contains no data records: "
                            + dataPath
            );
        }

        long end =
                System.nanoTime();

        ProgressLogger.logDuration(
                start,
                end
        );

        return dataset;
    }

    private CsvReader<CsvRecord> createCsvReader(
            Path dataPath
    ) throws IOException {

        return CsvReader.builder()
                .fieldSeparator(
                        fieldSeparator
                )
                /*
                 * Empty physical lines are treated as formatting rather than
                 * dataset instances. Empty fields inside a real record are
                 * still preserved by FastCSV.
                 */
                .skipEmptyLines(
                        true
                )
                .detectBomHeader(
                        true
                )
                .ofCsvRecord(
                        dataPath
                );
    }

    private CsvRecord requireHeaderRecord(
            java.util.Iterator<CsvRecord> iterator,
            Path dataPath
    ) throws IOException {

        if (!iterator.hasNext()) {
            throw new IOException(
                    "Long-format file is empty: "
                            + dataPath
            );
        }

        if (!hasHeader) {
            throw new IOException(
                    "LongFormatReader currently requires hasHeader=true "
                            + "because idColumn, timeColumn, featureColumns, "
                            + "and labelColumns are specified by name."
            );
        }

        return iterator.next();
    }

    private ColumnSelection resolveGroupedSelection(
            Map<String, Integer> columnIndex,
            int columnCount
    ) {
        int idIndex =
                requireColumn(
                        columnIndex,
                        idColumn,
                        "idColumn"
                );

        int timeIndex =
                timeColumn == null
                        ? -1
                        : requireColumn(
                        columnIndex,
                        timeColumn,
                        "timeColumn"
                );

        int[] featureIndices =
                resolveColumns(
                        columnIndex,
                        featureColumns,
                        "featureColumns"
                );

        int[] labelIndices =
                resolveColumns(
                        columnIndex,
                        labelColumns,
                        "labelColumns"
                );

        validateSelectionOverlap(
                idIndex,
                timeIndex,
                featureIndices,
                labelIndices
        );

        return new ColumnSelection(
                columnCount,
                idIndex,
                timeIndex,
                featureIndices,
                labelIndices
        );
    }

    private ColumnSelection resolveRowWiseSelection(
            Map<String, Integer> columnIndex,
            int columnCount
    ) {
        int[] featureIndices =
                resolveColumns(
                        columnIndex,
                        featureColumns,
                        "featureColumns"
                );

        int[] labelIndices =
                resolveColumns(
                        columnIndex,
                        labelColumns,
                        "labelColumns"
                );

        validateFeatureLabelOverlap(
                featureIndices,
                labelIndices
        );

        return new ColumnSelection(
                columnCount,
                -1,
                -1,
                featureIndices,
                labelIndices
        );
    }

    private int[] resolveColumns(
            Map<String, Integer> columnIndex,
            List<String> requestedColumns,
            String optionName
    ) {
        int[] indices =
                new int[requestedColumns.size()];

        Set<Integer> used =
                new HashSet<>();

        for (int index = 0;
             index < requestedColumns.size();
             index++) {

            String columnName =
                    requestedColumns.get(
                            index
                    );

            int column =
                    requireColumn(
                            columnIndex,
                            columnName,
                            optionName
                    );

            if (!used.add(column)) {
                throw new IllegalArgumentException(
                        "Column was selected more than once for "
                                + optionName
                                + ": "
                                + columnName
                );
            }

            indices[index] =
                    column;
        }

        return indices;
    }

    private void validateSelectionOverlap(
            int idIndex,
            int timeIndex,
            int[] featureIndices,
            int[] labelIndices
    ) {
        for (int featureIndex : featureIndices) {
            if (featureIndex == idIndex) {
                throw new IllegalArgumentException(
                        "idColumn cannot also be a feature column."
                );
            }

            if (featureIndex == timeIndex) {
                throw new IllegalArgumentException(
                        "timeColumn cannot also be a feature column."
                );
            }
        }

        for (int labelIndex : labelIndices) {
            if (labelIndex == idIndex) {
                throw new IllegalArgumentException(
                        "idColumn cannot also be a label column."
                );
            }

            if (labelIndex == timeIndex) {
                throw new IllegalArgumentException(
                        "timeColumn cannot also be a label column."
                );
            }
        }

        validateFeatureLabelOverlap(
                featureIndices,
                labelIndices
        );
    }

    private void validateFeatureLabelOverlap(
            int[] featureIndices,
            int[] labelIndices
    ) {
        Set<Integer> featureSet =
                new HashSet<>();

        for (int featureIndex : featureIndices) {
            featureSet.add(
                    featureIndex
            );
        }

        for (int labelIndex : labelIndices) {
            if (featureSet.contains(labelIndex)) {
                throw new IllegalArgumentException(
                        "A column cannot be configured as both a feature "
                                + "column and a label column."
                );
            }
        }
    }

    private Object[] parseFeatureValues(
            List<String> fields,
            int[] featureIndices,
            Path dataPath,
            CsvRecord record
    ) {
        Object[] featureValues =
                new Object[featureIndices.length];

        for (int dimension = 0;
             dimension < featureIndices.length;
             dimension++) {

            int columnIndex =
                    featureIndices[dimension];

            featureValues[dimension] =
                    parseFeatureValue(
                            getField(
                                    fields,
                                    columnIndex
                            ),
                            dataPath,
                            record,
                            columnIndex
                    );
        }

        return featureValues;
    }

    private Object parseFeatureValue(
            String token,
            Path dataPath,
            CsvRecord record,
            int columnIndex
    ) {
        if (!isNumeric) {
            return parseGenericValue(
                    token
            );
        }

        String trimmed =
                token == null
                        ? ""
                        : token.trim();

        if (isMissingToken(trimmed)) {
            if (!hasMissingValues) {
                throw new IllegalArgumentException(
                        "Encountered a missing numeric value in file "
                                + dataPath
                                + " at CSV record beginning on line "
                                + record.getStartingLineNumber()
                                + ", column "
                                + columnIndex
                                + ", but hasMissingValues=false."
                );
            }

            return null;
        }

        try {
            /*
             * The general reader stores numeric values temporarily as Object
             * so grouped records can be sorted before final materialization.
             * JavaDoubleParser still accelerates the text-to-double step.
             */
            return JavaDoubleParser.parseDouble(
                    trimmed
            );
        } catch (NumberFormatException e) {
            throw new IllegalArgumentException(
                    "Could not parse numeric feature value '"
                            + trimmed
                            + "' in file "
                            + dataPath
                            + " at CSV record beginning on line "
                            + record.getStartingLineNumber()
                            + ", column "
                            + columnIndex
                            + ".",
                    e
            );
        }
    }

    private Object parseLabelValues(
            List<String> fields,
            int[] labelIndices
    ) {
        if (labelIndices.length == 0) {
            return null;
        }

        if (labelIndices.length == 1) {
            return parseLabelValue(
                    getField(
                            fields,
                            labelIndices[0]
                    )
            );
        }

        List<Object> labels =
                new ArrayList<>(
                        labelIndices.length
                );

        for (int labelIndex : labelIndices) {
            labels.add(
                    parseLabelValue(
                            getField(
                                    fields,
                                    labelIndex
                            )
                    )
            );
        }

        return Collections.unmodifiableList(
                labels
        );
    }

    private Object parseLabelValue(
            String token
    ) {
        if (token == null) {
            return null;
        }

        String trimmed =
                token.trim();

        if (isMissingToken(trimmed)) {
            return null;
        }

        if (isRegression) {
            return JavaDoubleParser.parseDouble(
                    trimmed
            );
        }

        try {
            return Integer.parseInt(
                    trimmed
            );
        } catch (NumberFormatException ignored) {
            try {
                double value =
                        JavaDoubleParser.parseDouble(
                                trimmed
                        );

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

    private Object parseGenericValue(
            String token
    ) {
        if (token == null) {
            return null;
        }

        String trimmed =
                token.trim();

        if (isMissingToken(trimmed)) {
            return null;
        }

        /*
         * Generic parsing is intentionally conservative. Numeric inference is
         * retained, but this remains the compatibility path rather than the
         * specialized numeric hot path.
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

    private ListObjectDataset buildDataset(
            Map<Object, List<LongRow>> groupedRows
    ) {
        ListObjectDataset dataset =
                new ListObjectDataset(
                        groupedRows.size()
                );

        int instanceIndex =
                0;

        int commonLength =
                -1;

        boolean unequalLengths =
                false;

        for (Map.Entry<Object, List<LongRow>> entry
                : groupedRows.entrySet()) {

            List<LongRow> rows =
                    entry.getValue();

            sortRows(
                    rows
            );

            Object label =
                    inferGroupLabel(
                            rows
                    );

            Object data =
                    buildSeriesData(
                            rows
                    );

            dataset.add(
                    label,
                    data,
                    instanceIndex
            );

            int length =
                    getDataLength(
                            data
                    );

            if (commonLength < 0) {
                commonLength =
                        length;
            } else if (length != commonLength) {
                unequalLengths =
                        true;
            }

            instanceIndex++;
        }

        int datasetLength =
                unequalLengths
                        ? 0
                        : Math.max(
                        commonLength,
                        0
                );

        dataset.setLength(
                datasetLength
        );

        AppContext.length =
                datasetLength;

        return dataset;
    }

    private void sortRows(
            List<LongRow> rows
    ) {
        if (timeColumn == null) {
            /*
             * LinkedHashMap and ArrayList already preserve input order.
             * Sorting by inputOrder would add O(n log n) work without changing
             * the result.
             */
            return;
        }

        rows.sort(
                (first, second) -> {
                    int timeComparison =
                            compareTimeValues(
                                    first.timeValue,
                                    second.timeValue
                            );

                    if (timeComparison != 0) {
                        return timeComparison;
                    }

                    /*
                     * Stable and deterministic ordering for duplicate time
                     * values.
                     */
                    return Integer.compare(
                            first.inputOrder,
                            second.inputOrder
                    );
                }
        );
    }

    @SuppressWarnings({
            "rawtypes",
            "unchecked"
    })
    private int compareTimeValues(
            Object first,
            Object second
    ) {
        if (first == null && second == null) {
            return 0;
        }

        if (first == null) {
            return -1;
        }

        if (second == null) {
            return 1;
        }

        if (first instanceof Number
                && second instanceof Number) {

            return Double.compare(
                    ((Number) first).doubleValue(),
                    ((Number) second).doubleValue()
            );
        }

        if (first instanceof Comparable
                && first.getClass().isInstance(second)) {

            return ((Comparable) first).compareTo(
                    second
            );
        }

        return first.toString().compareTo(
                second.toString()
        );
    }

    private Object inferGroupLabel(
            List<LongRow> rows
    ) {
        if (rows.isEmpty()) {
            return null;
        }

        Object firstLabel =
                rows.get(0).label;

        for (LongRow row : rows) {
            if (!java.util.Objects.equals(
                    firstLabel,
                    row.label
            )) {
                throw new IllegalArgumentException(
                        "Inconsistent labels found within long-format "
                                + "group for id: "
                                + row.id
                );
            }
        }

        return firstLabel;
    }

    private Object buildSeriesData(
            List<LongRow> rows
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
            List<LongRow> rows,
            int timeLength
    ) {
        if (isNumeric) {
            if (hasMissingValues) {
                Double[] data =
                        new Double[timeLength];

                for (int timeIndex = 0;
                     timeIndex < timeLength;
                     timeIndex++) {

                    data[timeIndex] =
                            toBoxedDouble(
                                    rows.get(timeIndex)
                                            .featureValues[0]
                            );
                }

                return data;
            }

            double[] data =
                    new double[timeLength];

            for (int timeIndex = 0;
                 timeIndex < timeLength;
                 timeIndex++) {

                data[timeIndex] =
                        toPrimitiveDouble(
                                rows.get(timeIndex)
                                        .featureValues[0]
                        );
            }

            return data;
        }

        Object[] data =
                new Object[timeLength];

        for (int timeIndex = 0;
             timeIndex < timeLength;
             timeIndex++) {

            data[timeIndex] =
                    rows.get(timeIndex)
                            .featureValues[0];
        }

        return data;
    }

    private Object buildMultivariateSeries(
            List<LongRow> rows,
            int dimensionCount,
            int timeLength
    ) {
        if (isNumeric) {
            if (hasMissingValues) {
                Double[][] data =
                        new Double[dimensionCount][timeLength];

                for (int timeIndex = 0;
                     timeIndex < timeLength;
                     timeIndex++) {

                    Object[] values =
                            rows.get(timeIndex)
                                    .featureValues;

                    for (int dimension = 0;
                         dimension < dimensionCount;
                         dimension++) {

                        data[dimension][timeIndex] =
                                toBoxedDouble(
                                        values[dimension]
                                );
                    }
                }

                return data;
            }

            double[][] data =
                    new double[dimensionCount][timeLength];

            for (int timeIndex = 0;
                 timeIndex < timeLength;
                 timeIndex++) {

                Object[] values =
                        rows.get(timeIndex)
                                .featureValues;

                for (int dimension = 0;
                     dimension < dimensionCount;
                     dimension++) {

                    data[dimension][timeIndex] =
                            toPrimitiveDouble(
                                    values[dimension]
                            );
                }
            }

            return data;
        }

        Object[][] data =
                new Object[dimensionCount][timeLength];

        for (int timeIndex = 0;
             timeIndex < timeLength;
             timeIndex++) {

            Object[] values =
                    rows.get(timeIndex)
                            .featureValues;

            for (int dimension = 0;
                 dimension < dimensionCount;
                 dimension++) {

                data[dimension][timeIndex] =
                        values[dimension];
            }
        }

        return data;
    }

    private Object buildRowWiseData(
            Object[] featureValues
    ) {
        int length =
                featureValues.length;

        if (isNumeric) {
            if (hasMissingValues) {
                Double[] data =
                        new Double[length];

                for (int index = 0;
                     index < length;
                     index++) {

                    data[index] =
                            toBoxedDouble(
                                    featureValues[index]
                            );
                }

                return data;
            }

            double[] data =
                    new double[length];

            for (int index = 0;
                 index < length;
                 index++) {

                data[index] =
                        toPrimitiveDouble(
                                featureValues[index]
                        );
            }

            return data;
        }

        return featureValues.clone();
    }

    private static Double toBoxedDouble(
            Object value
    ) {
        if (value == null) {
            return null;
        }

        if (value instanceof Double doubleValue) {
            return doubleValue;
        }

        if (value instanceof Number number) {
            return number.doubleValue();
        }

        return JavaDoubleParser.parseDouble(
                value.toString()
        );
    }

    private static double toPrimitiveDouble(
            Object value
    ) {
        if (value == null) {
            throw new IllegalArgumentException(
                    "Encountered null numeric value, but "
                            + "hasMissingValues=false."
            );
        }

        if (value instanceof Number number) {
            return number.doubleValue();
        }

        return JavaDoubleParser.parseDouble(
                value.toString()
        );
    }

    private static int getDataLength(
            Object data
    ) {
        if (data instanceof double[] array) {
            return array.length;
        }

        if (data instanceof Double[] array) {
            return array.length;
        }

        if (data instanceof double[][] matrix) {
            return matrix.length == 0
                    ? 0
                    : matrix[0].length;
        }

        if (data instanceof Double[][] matrix) {
            return matrix.length == 0
                    ? 0
                    : matrix[0].length;
        }

        if (data instanceof Object[][] matrix) {
            return matrix.length == 0
                    ? 0
                    : matrix[0].length;
        }

        if (data instanceof Object[] array) {
            return array.length;
        }

        return 0;
    }

    private static void updateGlobalLength(
            Object data
    ) {
        AppContext.length =
                getDataLength(
                        data
                );
    }

    private Map<String, Integer> buildColumnIndex(
            List<String> header,
            Path dataPath
    ) {
        Map<String, Integer> columnIndex =
                new LinkedHashMap<>(
                        Math.max(
                                16,
                                header.size() * 2
                        )
                );

        for (int index = 0;
             index < header.size();
             index++) {

            String rawName =
                    header.get(index);

            String columnName =
                    rawName == null
                            ? ""
                            : rawName.trim();

            if (columnName.isEmpty()) {
                throw new IllegalArgumentException(
                        "Blank column name at index "
                                + index
                                + " in long-format file "
                                + dataPath
                                + "."
                );
            }

            Integer previous =
                    columnIndex.put(
                            columnName,
                            index
                    );

            if (previous != null) {
                throw new IllegalArgumentException(
                        "Duplicate column name in long-format file: "
                                + columnName
                );
            }
        }

        return columnIndex;
    }

    private static int requireColumn(
            Map<String, Integer> columnIndex,
            String columnName,
            String optionName
    ) {
        if (columnName == null || columnName.isBlank()) {
            throw new IllegalArgumentException(
                    "LongFormatReader requires "
                            + optionName
                            + "."
            );
        }

        Integer index =
                columnIndex.get(
                        columnName
                );

        if (index == null) {
            throw new IllegalArgumentException(
                    "Column not found for "
                            + optionName
                            + ": "
                            + columnName
                            + ". Available columns: "
                            + columnIndex.keySet()
            );
        }

        return index;
    }

    private void validateRecordWidth(
            List<String> fields,
            int expectedWidth,
            Path dataPath,
            CsvRecord record
    ) {
        if (fields.size() == expectedWidth) {
            return;
        }

        throw new IllegalArgumentException(
                "Inconsistent column count in long-format file "
                        + dataPath
                        + " at CSV record beginning on line "
                        + record.getStartingLineNumber()
                        + ". Expected "
                        + expectedWidth
                        + " columns but found "
                        + fields.size()
                        + "."
        );
    }

    private static String getField(
            List<String> fields,
            int index
    ) {
        if (index < 0 || index >= fields.size()) {
            throw new IllegalArgumentException(
                    "Column index out of bounds: "
                            + index
            );
        }

        return fields.get(
                index
        );
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

    private void validateOptions() {
        if (!hasHeader) {
            throw new IllegalArgumentException(
                    "LongFormatReader currently requires hasHeader=true."
            );
        }

        if (featureColumns.isEmpty()) {
            throw new IllegalArgumentException(
                    "LongFormatReader requires at least one "
                            + "feature column."
            );
        }

        if (idColumn == null && timeColumn != null) {
            throw new IllegalArgumentException(
                    "timeColumn is only meaningful in grouped long-format "
                            + "mode. Configure idColumn or remove timeColumn."
            );
        }
    }

    private Path validateDataFile()
            throws IOException {

        Path dataPath =
                Path.of(
                        dataFileName
                );

        if (!Files.exists(dataPath)) {
            throw new IOException(
                    "Long-format data file does not exist: "
                            + dataPath
            );
        }

        if (!Files.isRegularFile(dataPath)) {
            throw new IOException(
                    "Long-format data path is not a regular file: "
                            + dataPath
            );
        }

        if (!Files.isReadable(dataPath)) {
            throw new IOException(
                    "Long-format data file is not readable: "
                            + dataPath
            );
        }

        return dataPath;
    }

    private static ReaderOptions requireOptions(
            ReaderOptions options
    ) {
        if (options == null) {
            throw new IllegalArgumentException(
                    "LongFormatReader requires non-null ReaderOptions."
            );
        }

        return options;
    }

    private static String requireNonblank(
            String value,
            String optionName
    ) {
        if (value == null || value.isBlank()) {
            throw new IllegalArgumentException(
                    "LongFormatReader requires "
                            + optionName
                            + "."
            );
        }

        return value.trim();
    }

    private static String validateAndNormalizeSeparator(
            String separator
    ) {
        if (separator == null || separator.isEmpty()) {
            throw new IllegalArgumentException(
                    "LongFormatReader requires entrySeparator."
            );
        }

        String normalized =
                switch (separator) {
                    case "\\t" -> "\t";
                    case "\\n" -> "\n";
                    case "\\r" -> "\r";
                    default -> separator;
                };

        if (normalized.length() != 1) {
            throw new IllegalArgumentException(
                    "FastCSV-backed LongFormatReader requires a "
                            + "single-character entrySeparator. Received: '"
                            + separator
                            + "'."
            );
        }

        char value =
                normalized.charAt(0);

        if (value == '\n' || value == '\r') {
            throw new IllegalArgumentException(
                    "entrySeparator cannot be a line-separator character."
            );
        }

        return normalized;
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

    private static List<String> copyAndValidateColumns(
            List<String> columns,
            String optionName,
            boolean allowEmpty
    ) {
        if (columns == null || columns.isEmpty()) {
            if (allowEmpty) {
                return List.of();
            }

            return List.of();
        }

        List<String> copy =
                new ArrayList<>(
                        columns.size()
                );

        Set<String> used =
                new HashSet<>();

        for (String column : columns) {
            if (column == null || column.isBlank()) {
                throw new IllegalArgumentException(
                        optionName
                                + " cannot contain null or blank names."
                );
            }

            String normalized =
                    column.trim();

            if (!used.add(normalized)) {
                throw new IllegalArgumentException(
                        optionName
                                + " contains a duplicate column: "
                                + normalized
                );
            }

            copy.add(
                    normalized
            );
        }

        return Collections.unmodifiableList(
                copy
        );
    }

    private static final class ColumnSelection {

        private final int columnCount;
        private final int idIndex;
        private final int timeIndex;
        private final int[] featureIndices;
        private final int[] labelIndices;

        private ColumnSelection(
                int columnCount,
                int idIndex,
                int timeIndex,
                int[] featureIndices,
                int[] labelIndices
        ) {
            this.columnCount =
                    columnCount;

            this.idIndex =
                    idIndex;

            this.timeIndex =
                    timeIndex;

            this.featureIndices =
                    featureIndices.clone();

            this.labelIndices =
                    labelIndices.clone();
        }
    }

    private static final class LongRow {

        private final Object id;
        private final Object timeValue;
        private final Object[] featureValues;
        private final Object label;
        private final int inputOrder;

        private LongRow(
                Object id,
                Object timeValue,
                Object[] featureValues,
                Object label,
                int inputOrder
        ) {
            this.id =
                    id;

            this.timeValue =
                    timeValue;

            this.featureValues =
                    featureValues;

            this.label =
                    label;

            this.inputOrder =
                    inputOrder;
        }
    }

    public static class ProgressLogger {

        public static void logProgress(
                int index
        ) {
            if (index % 1000 != 0) {
                return;
            }

            if (index % 100000 == 0) {
                System.out.print("\n");

                if (index % 1000000 == 0) {
                    long usedMemory =
                            AppContext.runtime.totalMemory()
                                    - AppContext.runtime.freeMemory();

                    System.out.print(
                            index
                                    + ":"
                                    + usedMemory / 1024 / 1024
                                    + "mb\n"
                    );
                }

                return;
            }

            System.out.print(".");
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