package datasets.readers;

import ch.randelshofer.fastdoubleparser.JavaDoubleParser;
import core.AppContext;
import datasets.ListObjectDataset;
import de.siegmar.fastcsv.reader.CsvReader;
import de.siegmar.fastcsv.reader.CsvRecord;
import org.apache.commons.lang3.time.DurationFormatUtils;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.io.LineNumberReader;
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
import java.util.stream.Collectors;

/**
 * Reader for delimited files such as CSV, TSV, TXT, and row-encoded
 * multivariate time-series files.
 *
 * <p>Each CSV record becomes one {@link ListObjectDataset} instance.</p>
 *
 * <p>Supported instance shapes:</p>
 *
 * <ol>
 *     <li>Numeric 1D without missing values: {@code double[]}</li>
 *     <li>Numeric 1D with missing values: {@code Double[]}</li>
 *     <li>Numeric 2D without missing values: {@code double[][]}</li>
 *     <li>Numeric 2D with missing values: {@code Double[][]}</li>
 *     <li>Generic 1D: {@code Object[]}</li>
 *     <li>Generic 2D: {@code Object[][]}</li>
 * </ol>
 *
 * <p>For 1D files, {@code entrySeparator} separates the entries in a record.</p>
 *
 * <p>For row-encoded 2D files, {@code arraySeparator} separates dimensions and
 * {@code entrySeparator} separates values within each dimension. For example,
 * with {@code arraySeparator=";"} and {@code entrySeparator=","}:</p>
 *
 * <pre>
 * 1,2,3;4,5,6
 * </pre>
 *
 * <p>represents two dimensions, each containing three time points.</p>
 *
 * <p>The outer file parsing is performed by FastCSV. Consequently, quoted
 * fields and delimiters inside quoted fields are supported. The configured
 * outer separator must be exactly one character:</p>
 *
 * <ul>
 *     <li>{@code entrySeparator} for 1D data</li>
 *     <li>{@code arraySeparator} for 2D data</li>
 * </ul>
 *
 * <p>The inner {@code entrySeparator} for row-encoded 2D data must also be
 * exactly one character.</p>
 */
public class DelimitedFileReader
        implements DatasetReader {

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
        this.dataFileName =
                requireNonblank(
                        dataFileName,
                        "dataFileName"
                );

        this.labelFileName =
                normalizeNullableString(
                        labelFileName
                );

        this.entrySeparator =
                normalizeSeparator(
                        entrySeparator
                );

        this.arraySeparator =
                normalizeSeparator(
                        arraySeparator
                );

        this.hasHeader =
                hasHeader;

        this.is2D =
                is2D;

        this.isNumeric =
                isNumeric;

        this.hasMissingValues =
                hasMissingValues;

        this.targetColumnIsFirst =
                targetColumnIsFirst;

        this.isTest =
                isTest;

        this.isRegression =
                isRegression;

        this.innerSeparator =
                requireSingleCharacterSeparator(
                        this.entrySeparator,
                        "entrySeparator"
                );

        this.outerSeparator =
                is2D
                        ? requireSingleCharacterSeparator(
                        this.arraySeparator,
                        "arraySeparator"
                )
                        : this.innerSeparator;
    }

    @Override
    public ListObjectDataset read()
            throws IOException {

        long start =
                System.nanoTime();

        List<Object> labels =
                labelFileName == null
                        ? List.of()
                        : readGenericLabels(
                        labelFileName,
                        hasHeader,
                        isRegression
                );

        Path dataPath =
                Path.of(
                        dataFileName
                );

        validateDataFile(
                dataPath
        );

        ListObjectDataset dataset =
                new ListObjectDataset();

        int instanceIndex =
                0;

        try (CsvReader<CsvRecord> csvReader =
                     CsvReader.builder()
                             .fieldSeparator(outerSeparator)
                             /*
                              * Preserve blank records rather than silently
                              * dropping a possible all-missing instance.
                              */
                             .skipEmptyLines(false)
                             .detectBomHeader(true)
                             .ofCsvRecord(dataPath)) {

            var iterator =
                    csvReader.iterator();

            if (hasHeader) {
                if (!iterator.hasNext()) {
                    throw new IOException(
                            "Delimited file contains no header or data: "
                                    + dataPath
                    );
                }

                iterator.next();
            }

            while (iterator.hasNext()) {
                CsvRecord record =
                        iterator.next();

                List<String> fields =
                        record.getFields();

                if (fields.isEmpty()) {
                    throw new IllegalArgumentException(
                            "Encountered a delimited record with no fields "
                                    + "in file "
                                    + dataPath
                                    + "."
                    );
                }

                ParsedInstance parsed =
                        parseRecord(
                                fields,
                                labels,
                                instanceIndex,
                                dataPath
                        );

                dataset.add(
                        parsed.label,
                        parsed.data,
                        instanceIndex
                );

                AppContext.length =
                        parsed.length;

                ProgressLogger.logProgress(
                        instanceIndex
                );

                instanceIndex++;
            }
        } catch (IllegalArgumentException e) {
            throw e;
        } catch (RuntimeException e) {
            throw new IOException(
                    "Failed while parsing delimited dataset: "
                            + dataPath,
                    e
            );
        }

        if (instanceIndex == 0) {
            throw new IOException(
                    "Delimited dataset contains no data records: "
                            + dataPath
            );
        }

        validateSeparateLabelCount(
                labels,
                instanceIndex
        );

        long end =
                System.nanoTime();

        ProgressLogger.logDuration(
                start,
                end
        );

        return dataset;
    }

    private ParsedInstance parseRecord(
            List<String> fields,
            List<Object> labels,
            int instanceIndex,
            Path dataPath
    ) {
        if (isNumeric) {
            if (hasMissingValues) {
                if (is2D) {
                    Double[][] data =
                            parseBoxedDoubleMatrix(
                                    fields,
                                    dataPath,
                                    instanceIndex
                            );

                    return new ParsedInstance(
                            getSeparateLabel(
                                    labels,
                                    instanceIndex
                            ),
                            data,
                            requireMatrixLength(
                                    data,
                                    dataPath,
                                    instanceIndex
                            )
                    );
                }

                return parseBoxedDoubleVector(
                        fields,
                        labels,
                        instanceIndex,
                        dataPath
                );
            }

            if (is2D) {
                double[][] data =
                        parseDoubleMatrix(
                                fields,
                                dataPath,
                                instanceIndex
                        );

                return new ParsedInstance(
                        getSeparateLabel(
                                labels,
                                instanceIndex
                        ),
                        data,
                        requireMatrixLength(
                                data,
                                dataPath,
                                instanceIndex
                        )
                );
            }

            return parseDoubleVector(
                    fields,
                    labels,
                    instanceIndex,
                    dataPath
            );
        }

        if (is2D) {
            Object[][] data =
                    parseObjectMatrix(
                            fields,
                            dataPath,
                            instanceIndex
                    );

            return new ParsedInstance(
                    getSeparateLabel(
                            labels,
                            instanceIndex
                    ),
                    data,
                    requireMatrixLength(
                            data,
                            dataPath,
                            instanceIndex
                    )
            );
        }

        Object[] data =
                parseObjectVector(
                        fields
                );

        return new ParsedInstance(
                getSeparateLabel(
                        labels,
                        instanceIndex
                ),
                data,
                data.length
        );
    }

    private ParsedInstance parseDoubleVector(
            List<String> fields,
            List<Object> labels,
            int instanceIndex,
            Path dataPath
    ) {
        boolean parseEmbeddedLabel =
                shouldParseEmbeddedLabel();

        if (!parseEmbeddedLabel) {
            double[] data =
                    new double[fields.size()];

            for (int fieldIndex = 0;
                 fieldIndex < fields.size();
                 fieldIndex++) {

                data[fieldIndex] =
                        parseRequiredDouble(
                                fields.get(fieldIndex),
                                dataPath,
                                instanceIndex,
                                fieldIndex
                        );
            }

            return new ParsedInstance(
                    getSeparateLabel(
                            labels,
                            instanceIndex
                    ),
                    data,
                    data.length
            );
        }

        validateEmbeddedLabelRecord(
                fields,
                dataPath,
                instanceIndex
        );

        int labelIndex =
                targetColumnIsFirst
                        ? 0
                        : fields.size() - 1;

        Object label =
                parseLabel(
                        fields.get(labelIndex)
                );

        double[] data =
                new double[fields.size() - 1];

        int outputIndex =
                0;

        for (int fieldIndex = 0;
             fieldIndex < fields.size();
             fieldIndex++) {

            if (fieldIndex == labelIndex) {
                continue;
            }

            data[outputIndex++] =
                    parseRequiredDouble(
                            fields.get(fieldIndex),
                            dataPath,
                            instanceIndex,
                            fieldIndex
                    );
        }

        return new ParsedInstance(
                label,
                data,
                data.length
        );
    }

    private ParsedInstance parseBoxedDoubleVector(
            List<String> fields,
            List<Object> labels,
            int instanceIndex,
            Path dataPath
    ) {
        boolean parseEmbeddedLabel =
                shouldParseEmbeddedLabel();

        if (!parseEmbeddedLabel) {
            Double[] data =
                    new Double[fields.size()];

            for (int fieldIndex = 0;
                 fieldIndex < fields.size();
                 fieldIndex++) {

                data[fieldIndex] =
                        parseNullableDouble(
                                fields.get(fieldIndex),
                                dataPath,
                                instanceIndex,
                                fieldIndex
                        );
            }

            return new ParsedInstance(
                    getSeparateLabel(
                            labels,
                            instanceIndex
                    ),
                    data,
                    data.length
            );
        }

        validateEmbeddedLabelRecord(
                fields,
                dataPath,
                instanceIndex
        );

        int labelIndex =
                targetColumnIsFirst
                        ? 0
                        : fields.size() - 1;

        Object label =
                parseLabel(
                        fields.get(labelIndex)
                );

        Double[] data =
                new Double[fields.size() - 1];

        int outputIndex =
                0;

        for (int fieldIndex = 0;
             fieldIndex < fields.size();
             fieldIndex++) {

            if (fieldIndex == labelIndex) {
                continue;
            }

            data[outputIndex++] =
                    parseNullableDouble(
                            fields.get(fieldIndex),
                            dataPath,
                            instanceIndex,
                            fieldIndex
                    );
        }

        return new ParsedInstance(
                label,
                data,
                data.length
        );
    }

    private double[][] parseDoubleMatrix(
            List<String> dimensions,
            Path dataPath,
            int instanceIndex
    ) {
        double[][] data =
                new double[dimensions.size()][];

        int expectedLength =
                -1;

        for (int dimension = 0;
             dimension < dimensions.size();
             dimension++) {

            String[] tokens =
                    splitLiteral(
                            dimensions.get(dimension),
                            innerSeparator
                    );

            double[] values =
                    new double[tokens.length];

            for (int timeIndex = 0;
                 timeIndex < tokens.length;
                 timeIndex++) {

                values[timeIndex] =
                        parseRequiredDouble(
                                tokens[timeIndex],
                                dataPath,
                                instanceIndex,
                                dimension,
                                timeIndex
                        );
            }

            expectedLength =
                    validateDimensionLength(
                            expectedLength,
                            values.length,
                            dataPath,
                            instanceIndex,
                            dimension
                    );

            data[dimension] =
                    values;
        }

        return data;
    }

    private Double[][] parseBoxedDoubleMatrix(
            List<String> dimensions,
            Path dataPath,
            int instanceIndex
    ) {
        Double[][] data =
                new Double[dimensions.size()][];

        int expectedLength =
                -1;

        for (int dimension = 0;
             dimension < dimensions.size();
             dimension++) {

            String[] tokens =
                    splitLiteral(
                            dimensions.get(dimension),
                            innerSeparator
                    );

            Double[] values =
                    new Double[tokens.length];

            for (int timeIndex = 0;
                 timeIndex < tokens.length;
                 timeIndex++) {

                values[timeIndex] =
                        parseNullableDouble(
                                tokens[timeIndex],
                                dataPath,
                                instanceIndex,
                                dimension,
                                timeIndex
                        );
            }

            expectedLength =
                    validateDimensionLength(
                            expectedLength,
                            values.length,
                            dataPath,
                            instanceIndex,
                            dimension
                    );

            data[dimension] =
                    values;
        }

        return data;
    }

    private Object[][] parseObjectMatrix(
            List<String> dimensions,
            Path dataPath,
            int instanceIndex
    ) {
        Object[][] data =
                new Object[dimensions.size()][];

        int expectedLength =
                -1;

        for (int dimension = 0;
             dimension < dimensions.size();
             dimension++) {

            String[] tokens =
                    splitLiteral(
                            dimensions.get(dimension),
                            innerSeparator
                    );

            Object[] values =
                    new Object[tokens.length];

            for (int timeIndex = 0;
                 timeIndex < tokens.length;
                 timeIndex++) {

                values[timeIndex] =
                        RowParser.parseValue(
                                tokens[timeIndex]
                        );
            }

            expectedLength =
                    validateDimensionLength(
                            expectedLength,
                            values.length,
                            dataPath,
                            instanceIndex,
                            dimension
                    );

            data[dimension] =
                    values;
        }

        return data;
    }

    private Object[] parseObjectVector(
            List<String> fields
    ) {
        Object[] data =
                new Object[fields.size()];

        for (int fieldIndex = 0;
             fieldIndex < fields.size();
             fieldIndex++) {

            data[fieldIndex] =
                    RowParser.parseValue(
                            fields.get(fieldIndex)
                    );
        }

        return data;
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
                    "Delimited record "
                            + instanceIndex
                            + " in file "
                            + dataPath
                            + " must contain at least one feature and one "
                            + "embedded label."
            );
        }
    }

    private Object parseLabel(
            String token
    ) {
        if (isRegression) {
            return Double.parseDouble(
                    requireNonemptyToken(
                            token,
                            "Embedded regression label"
                    )
            );
        }

        return RowParser.tryParseLabel(
                token
        );
    }

    private double parseRequiredDouble(
            String token,
            Path dataPath,
            int instanceIndex,
            int fieldIndex
    ) {
        String trimmed =
                token == null
                        ? ""
                        : token.trim();

        if (RowParser.isMissingToken(trimmed)) {
            throw new IllegalArgumentException(
                    "Encountered a missing value in file "
                            + dataPath
                            + " at instance "
                            + instanceIndex
                            + ", field "
                            + fieldIndex
                            + ", but hasMissingValues is false."
            );
        }

        try {
            //return Double.parseDouble(
            //        trimmed
            //);
            return JavaDoubleParser.parseDouble(trimmed);
        } catch (NumberFormatException e) {
            throw new IllegalArgumentException(
                    "Could not parse numeric value '"
                            + trimmed
                            + "' in file "
                            + dataPath
                            + " at instance "
                            + instanceIndex
                            + ", field "
                            + fieldIndex
                            + ".",
                    e
            );
        }
    }

    private double parseRequiredDouble(
            String token,
            Path dataPath,
            int instanceIndex,
            int dimension,
            int timeIndex
    ) {
        String trimmed =
                token == null
                        ? ""
                        : token.trim();

        if (RowParser.isMissingToken(trimmed)) {
            throw new IllegalArgumentException(
                    "Encountered a missing value in file "
                            + dataPath
                            + " at instance "
                            + instanceIndex
                            + ", dimension "
                            + dimension
                            + ", time index "
                            + timeIndex
                            + ", but hasMissingValues is false."
            );
        }

        try {
            //return Double.parseDouble(trimmed);
            return JavaDoubleParser.parseDouble(trimmed);
        } catch (NumberFormatException e) {
            throw new IllegalArgumentException(
                    "Could not parse numeric value '"
                            + trimmed
                            + "' in file "
                            + dataPath
                            + " at instance "
                            + instanceIndex
                            + ", dimension "
                            + dimension
                            + ", time index "
                            + timeIndex
                            + ".",
                    e
            );
        }
    }

    private Double parseNullableDouble(
            String token,
            Path dataPath,
            int instanceIndex,
            int fieldIndex
    ) {
        String trimmed =
                token == null
                        ? ""
                        : token.trim();

        if (RowParser.isMissingToken(trimmed)) {
            return null;
        }

        try {
            //return Double.valueOf(trimmed);
            return JavaDoubleParser.parseDouble(trimmed);
        } catch (NumberFormatException e) {
            throw new IllegalArgumentException(
                    "Could not parse numeric value '"
                            + trimmed
                            + "' in file "
                            + dataPath
                            + " at instance "
                            + instanceIndex
                            + ", field "
                            + fieldIndex
                            + ".",
                    e
            );
        }
    }

    private Double parseNullableDouble(
            String token,
            Path dataPath,
            int instanceIndex,
            int dimension,
            int timeIndex
    ) {
        String trimmed =
                token == null
                        ? ""
                        : token.trim();

        if (RowParser.isMissingToken(trimmed)) {
            return null;
        }

        try {
            //return Double.valueOf(trimmed);
            return JavaDoubleParser.parseDouble(trimmed);
        } catch (NumberFormatException e) {
            throw new IllegalArgumentException(
                    "Could not parse numeric value '"
                            + trimmed
                            + "' in file "
                            + dataPath
                            + " at instance "
                            + instanceIndex
                            + ", dimension "
                            + dimension
                            + ", time index "
                            + timeIndex
                            + ".",
                    e
            );
        }
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
                    "Row-encoded 2D instance "
                            + instanceIndex
                            + " in file "
                            + dataPath
                            + " has inconsistent dimension lengths. "
                            + "Expected "
                            + expectedLength
                            + " values but dimension "
                            + dimension
                            + " contains "
                            + actualLength
                            + "."
            );
        }

        return expectedLength;
    }

    private int requireMatrixLength(
            Object[] matrix,
            Path dataPath,
            int instanceIndex
    ) {
        if (matrix.length == 0) {
            throw new IllegalArgumentException(
                    "Row-encoded 2D instance "
                            + instanceIndex
                            + " in file "
                            + dataPath
                            + " contains no dimensions."
            );
        }

        Object firstDimension =
                matrix[0];

        if (firstDimension == null
                || !firstDimension.getClass().isArray()) {

            throw new IllegalArgumentException(
                    "Row-encoded 2D instance "
                            + instanceIndex
                            + " in file "
                            + dataPath
                            + " has an invalid first dimension."
            );
        }

        return java.lang.reflect.Array.getLength(
                firstDimension
        );
    }

    private int requireMatrixLength(
            double[][] matrix,
            Path dataPath,
            int instanceIndex
    ) {
        if (matrix.length == 0) {
            throw new IllegalArgumentException(
                    "Row-encoded 2D instance "
                            + instanceIndex
                            + " in file "
                            + dataPath
                            + " contains no dimensions."
            );
        }

        return matrix[0].length;
    }

    private Object getSeparateLabel(
            List<Object> labels,
            int index
    ) {
        if (labelFileName == null) {
            return null;
        }

        if (index >= labels.size()) {
            throw new IllegalArgumentException(
                    "The separate label file contains fewer labels than "
                            + "the data file. Missing label for instance "
                            + index
                            + "."
            );
        }

        return labels.get(
                index
        );
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
                    "Separate label count does not match the number of "
                            + "data instances. Labels="
                            + labels.size()
                            + ", instances="
                            + instanceCount
                            + "."
            );
        }
    }

    private void validateDataFile(
            Path dataPath
    ) throws IOException {
        if (!Files.exists(dataPath)) {
            throw new IOException(
                    "Delimited data file does not exist: "
                            + dataPath
            );
        }

        if (!Files.isRegularFile(dataPath)) {
            throw new IOException(
                    "Delimited data path is not a regular file: "
                            + dataPath
            );
        }

        if (!Files.isReadable(dataPath)) {
            throw new IOException(
                    "Delimited data file is not readable: "
                            + dataPath
            );
        }
    }

    private static String[] splitLiteral(
            String value,
            char separator
    ) {
        if (value == null) {
            return new String[]{null};
        }

        int fieldCount =
                1;

        for (int index = 0;
             index < value.length();
             index++) {

            if (value.charAt(index) == separator) {
                fieldCount++;
            }
        }

        String[] result =
                new String[fieldCount];

        int resultIndex =
                0;

        int fieldStart =
                0;

        for (int index = 0;
             index < value.length();
             index++) {

            if (value.charAt(index) != separator) {
                continue;
            }

            result[resultIndex++] =
                    value.substring(
                            fieldStart,
                            index
                    );

            fieldStart =
                    index + 1;
        }

        result[resultIndex] =
                value.substring(
                        fieldStart
                );

        return result;
    }

    private static String requireNonblank(
            String value,
            String argumentName
    ) {
        if (value == null || value.isBlank()) {
            throw new IllegalArgumentException(
                    argumentName
                            + " cannot be null or blank."
            );
        }

        return value.trim();
    }

    private static String requireNonemptyToken(
            String value,
            String role
    ) {
        if (value == null || value.trim().isEmpty()) {
            throw new IllegalArgumentException(
                    role
                            + " cannot be missing."
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

        String trimmed =
                value.trim();

        if (trimmed.isEmpty()
                || trimmed.equalsIgnoreCase("None")) {

            return null;
        }

        return trimmed;
    }

    private static String normalizeSeparator(
            String separator
    ) {
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
                            + argumentName
                            + "."
            );
        }

        if (separator.length() != 1) {
            throw new IllegalArgumentException(
                    "FastCSV-backed DelimitedFileReader requires "
                            + argumentName
                            + " to contain exactly one character. "
                            + "Received: '"
                            + separator
                            + "'."
            );
        }

        char value =
                separator.charAt(0);

        if (value == '\n' || value == '\r') {
            throw new IllegalArgumentException(
                    argumentName
                            + " cannot be a line-separator character."
            );
        }

        return value;
    }

    public static List<Integer> readLabels(
            String labelFileName,
            boolean hasHeader
    ) throws IOException {
        List<Integer> labels =
                new ArrayList<>();

        if (labelFileName == null) {
            return labels;
        }

        try (BufferedReader reader =
                     Files.newBufferedReader(
                             Path.of(labelFileName),
                             StandardCharsets.UTF_8
                     )) {

            if (hasHeader) {
                reader.readLine();
            }

            String line;

            while ((line = reader.readLine()) != null) {
                String trimmed =
                        line.trim();

                if (trimmed.isEmpty()) {
                    throw new IllegalArgumentException(
                            "Encountered an empty integer label in file: "
                                    + labelFileName
                    );
                }

                labels.add(
                        Integer.parseInt(
                                trimmed
                        )
                );
            }
        }

        return labels;
    }

    public static List<Object> readGenericLabels(
            String labelFileName,
            boolean hasHeader,
            boolean isRegression
    ) throws IOException {
        List<Object> labels =
                new ArrayList<>();

        if (labelFileName == null) {
            return labels;
        }

        try (BufferedReader reader =
                     Files.newBufferedReader(
                             Path.of(labelFileName),
                             StandardCharsets.UTF_8
                     )) {

            if (hasHeader) {
                reader.readLine();
            }

            String line;

            while ((line = reader.readLine()) != null) {
                String trimmed =
                        line.trim();

                if (trimmed.isEmpty()) {
                    throw new IllegalArgumentException(
                            "Encountered an empty label in file: "
                                    + labelFileName
                    );
                }

                if (isRegression) {
                    labels.add(
                            Double.parseDouble(
                                    trimmed
                            )
                    );
                } else {
                    labels.add(
                            RowParser.tryParseLabel(
                                    trimmed
                            )
                    );
                }
            }
        }

        return labels;
    }

    public static class ParsedDoubleRow {

        public final Object label;
        public final double[] features;

        public ParsedDoubleRow(
                Object label,
                double[] features
        ) {
            this.label =
                    label;

            this.features =
                    features;
        }
    }

    public static class ParsedBoxedDoubleRow {

        public final Object label;
        public final Double[] features;

        public ParsedBoxedDoubleRow(
                Object label,
                Double[] features
        ) {
            this.label =
                    label;

            this.features =
                    features;
        }
    }

    private static final class ParsedInstance {

        private final Object label;
        private final Object data;
        private final int length;

        private ParsedInstance(
                Object label,
                Object data,
                int length
        ) {
            this.label =
                    label;

            this.data =
                    data;

            this.length =
                    length;
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

    /**
     * Retained for compatibility with existing callers.
     *
     * <p>This helper still performs a complete record-count pass. It should
     * not be called before {@link #read()} merely to preallocate a dataset,
     * because that would read the data file twice.</p>
     */
    public static class FileInfoExtractor {

        public static int[] getFileInformation(
                String fileName,
                boolean hasHeader,
                String separator
        ) throws IOException {
            String normalized =
                    normalizeSeparator(
                            separator
                    );

            char fieldSeparator =
                    requireSingleCharacterSeparator(
                            normalized,
                            "separator"
                    );

            int recordCount =
                    0;

            int columnCount =
                    0;

            try (CsvReader<CsvRecord> csvReader =
                         CsvReader.builder()
                                 .fieldSeparator(fieldSeparator)
                                 .skipEmptyLines(false)
                                 .detectBomHeader(true)
                                 .ofCsvRecord(
                                         Path.of(fileName)
                                 )) {

                var iterator =
                        csvReader.iterator();

                if (hasHeader && iterator.hasNext()) {
                    CsvRecord header =
                            iterator.next();

                    columnCount =
                            header.getFields().size();
                }

                while (iterator.hasNext()) {
                    CsvRecord record =
                            iterator.next();

                    if (columnCount == 0) {
                        columnCount =
                                record.getFields().size();
                    }

                    recordCount++;
                }
            }

            return new int[]{
                    recordCount,
                    columnCount
            };
        }
    }

    /**
     * General row-parsing utilities retained for compatibility with existing
     * reader code, including PerFileDelimitedSeriesReader.
     */
    public static class RowParser {

        private static volatile Set<String> missingIndicators =
                normalizeMissingIndicators(
                        AppContext.MissingStrings
                );

        public static void setMissingIndicators(
                Set<String> indicators
        ) {
            missingIndicators =
                    normalizeMissingIndicators(
                            indicators
                    );
        }

        public static Set<String> getMissingIndicators() {
            return missingIndicators;
        }

        public static boolean isMissingToken(
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

        public static ParsedDoubleRow parseDoubleRow(
                String[] lineArray,
                boolean targetColumnIsFirst,
                boolean isRegression
        ) {
            int dataLength =
                    lineArray.length - 1;

            double[] features =
                    new double[dataLength];

            int labelIndex =
                    targetColumnIsFirst
                            ? 0
                            : dataLength;

            Object label =
                    isRegression
                            ? Double.parseDouble(
                            lineArray[labelIndex].trim()
                    )
                            : tryParseLabel(
                            lineArray[labelIndex]
                    );

            int outputIndex =
                    0;

            for (int index = 0;
                 index < lineArray.length;
                 index++) {

                if (index == labelIndex) {
                    continue;
                }

                //features[outputIndex++] = Double.parseDouble(lineArray[index].trim());
                features[outputIndex++] = JavaDoubleParser.parseDouble(lineArray[index].trim());
            }

            return new ParsedDoubleRow(
                    label,
                    features
            );
        }

        public static ParsedBoxedDoubleRow parseBoxedDoubleRow(
                String[] lineArray,
                boolean targetColumnIsFirst,
                boolean isRegression
        ) {
            int dataLength =
                    lineArray.length - 1;

            Double[] features =
                    new Double[dataLength];

            int labelIndex =
                    targetColumnIsFirst
                            ? 0
                            : dataLength;

            Object label =
                    isRegression
                            ? Double.parseDouble(
                            lineArray[labelIndex].trim()
                    )
                            : tryParseLabel(
                            lineArray[labelIndex]
                    );

            int outputIndex =
                    0;

            for (int index = 0;
                 index < lineArray.length;
                 index++) {

                if (index == labelIndex) {
                    continue;
                }

                features[outputIndex++] =
                        parseBoxedDoubleToken(
                                lineArray[index]
                        );
            }

            return new ParsedBoxedDoubleRow(
                    label,
                    features
            );
        }

        private static Double parseBoxedDoubleToken(
                String token
        ) {
            if (isMissingToken(token)) {
                return null;
            }

            //return Double.valueOf(token.trim());
            return JavaDoubleParser.parseDouble(token.trim());
        }

        public static Object tryParseLabel(
                String token
        ) {
            String trimmed =
                    token.trim();

            try {
                return Integer.parseInt(
                        trimmed
                );
            } catch (NumberFormatException ignored) {
                try {
                    double value =
                            Double.parseDouble(
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

        public static double[] parseDoubleArray(
                String row,
                String separator
        ) {
            char delimiter =
                    requireSingleCharacterSeparator(
                            normalizeSeparator(separator),
                            "separator"
                    );

            String[] tokens =
                    splitLiteral(
                            row,
                            delimiter
                    );

            double[] parsed =
                    new double[tokens.length];

            for (int index = 0;
                 index < tokens.length;
                 index++) {

                //parsed[index] = Double.parseDouble(tokens[index].trim());
                parsed[index] = JavaDoubleParser.parseDouble(tokens[index].trim());
            }

            return parsed;
        }

        public static double[][] parseDoubleMatrix(
                String row,
                String arraySeparator,
                String entrySeparator
        ) {
            char outer =
                    requireSingleCharacterSeparator(
                            normalizeSeparator(arraySeparator),
                            "arraySeparator"
                    );

            String[] rowStrings =
                    splitLiteral(
                            row,
                            outer
                    );

            double[][] matrix =
                    new double[rowStrings.length][];

            for (int index = 0;
                 index < rowStrings.length;
                 index++) {

                matrix[index] =
                        parseDoubleArray(
                                rowStrings[index],
                                entrySeparator
                        );
            }

            return matrix;
        }

        public static Double[] parseBoxedDoubleArray(
                String row,
                String separator
        ) {
            char delimiter =
                    requireSingleCharacterSeparator(
                            normalizeSeparator(separator),
                            "separator"
                    );

            String[] tokens =
                    splitLiteral(
                            row,
                            delimiter
                    );

            Double[] parsed =
                    new Double[tokens.length];

            for (int index = 0;
                 index < tokens.length;
                 index++) {

                parsed[index] =
                        parseBoxedDoubleToken(
                                tokens[index]
                        );
            }

            return parsed;
        }

        public static Double[][] parseBoxedDoubleMatrix(
                String row,
                String arraySeparator,
                String entrySeparator
        ) {
            char outer =
                    requireSingleCharacterSeparator(
                            normalizeSeparator(arraySeparator),
                            "arraySeparator"
                    );

            String[] rowStrings =
                    splitLiteral(
                            row,
                            outer
                    );

            Double[][] matrix =
                    new Double[rowStrings.length][];

            for (int index = 0;
                 index < rowStrings.length;
                 index++) {

                matrix[index] =
                        parseBoxedDoubleArray(
                                rowStrings[index],
                                entrySeparator
                        );
            }

            return matrix;
        }

        public static Object[] parse1DRow(
                String row,
                String separator
        ) {
            char delimiter =
                    requireSingleCharacterSeparator(
                            normalizeSeparator(separator),
                            "separator"
                    );

            String[] tokens =
                    splitLiteral(
                            row,
                            delimiter
                    );

            Object[] parsed =
                    new Object[tokens.length];

            for (int index = 0;
                 index < tokens.length;
                 index++) {

                parsed[index] =
                        parseValue(
                                tokens[index]
                        );
            }

            return parsed;
        }

        public static Object[][] parse2DRow(
                String row,
                String arraySeparator,
                String entrySeparator
        ) {
            char outer =
                    requireSingleCharacterSeparator(
                            normalizeSeparator(arraySeparator),
                            "arraySeparator"
                    );

            String[] rowStrings =
                    splitLiteral(
                            row,
                            outer
                    );

            Object[][] matrix =
                    new Object[rowStrings.length][];

            for (int index = 0;
                 index < rowStrings.length;
                 index++) {

                matrix[index] =
                        parse1DRow(
                                rowStrings[index],
                                entrySeparator
                        );
            }

            return matrix;
        }

        public static Object parseValue(
                String token
        ) {
            if (isMissingToken(token)) {
                return null;
            }

            String trimmed =
                    token.trim();

            try {
                return Double.parseDouble(
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

        private static Set<String> normalizeMissingIndicators(
                Set<String> indicators
        ) {
            if (indicators == null || indicators.isEmpty()) {
                return Set.of();
            }

            Set<String> normalized =
                    new HashSet<>();

            for (String indicator : indicators) {
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
    }
}
