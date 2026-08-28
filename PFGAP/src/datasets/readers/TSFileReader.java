package datasets.readers;

import core.AppContext;
import datasets.ListObjectDataset;
import org.apache.commons.lang3.time.DurationFormatUtils;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Set;
import java.util.stream.Collectors;

/**
 * Reader for .ts time-series files, including univariate and multivariate data.
 *
 * Expected .ts structure:
 *
 *      @problemName ...
 *      @timestamps true/false
 *      @univariate true/false
 *      @classLabel true labelA labelB ...
 *      @data
 *      dim1_values:dim2_values:...:label
 *
 * Examples:
 *
 * Univariate with labels:
 *
 *      1,2,3,4,5:A
 *      2,3,4,5,6:B
 *
 * Multivariate with labels:
 *
 *      1,2,3:4,5,6:A
 *      7,8,9:1,2,3:B
 *
 * Multivariate without labels:
 *
 *      1,2,3:4,5,6
 *      7,8,9:1,2,3
 *
 * Supported output shapes:
 *
 * 1. Numeric univariate, no missing values:
 *      double[]
 *
 * 2. Numeric univariate, with missing values:
 *      Double[]
 *
 * 3. Numeric multivariate, no missing values:
 *      double[][]
 *
 * 4. Numeric multivariate, with missing values:
 *      Double[][]
 *
 * 5. Generic univariate:
 *      Object[]
 *
 * 6. Generic multivariate:
 *      Object[][]
 *
 * Label handling:
 *
 * - If @classLabel true appears in the file, the final colon-separated field
 *   of each data row is treated as the label.
 * - If labelFileName is supplied, labels are read from that file instead.
 * - Supplying both embedded labels and a separate label file is treated as
 *   ambiguous and causes an IOException.
 *
 * Notes:
 *
 * - This reader supports basic timestamped .ts values of the form:
 *
 *      (timestamp,value),(timestamp,value)
 *
 *   In that case, the timestamp is ignored and only the value is retained.
 *
 * - It does not currently preserve timestamp coordinates.
 */
public class TSFileReader implements DatasetReader {

    private static final String DIMENSION_SEPARATOR = ":";
    private static final char VALUE_SEPARATOR = ',';

    private final String dataFileName;
    private final String labelFileName;
    private final boolean isNumeric;
    private final boolean hasMissingValues;
    private final boolean isRegression;

    public TSFileReader(
            String dataFileName,
            boolean isNumeric,
            boolean hasMissingValues,
            boolean isRegression
    ) {
        this(
                dataFileName,
                null,
                isNumeric,
                hasMissingValues,
                isRegression
        );
    }

    public TSFileReader(
            String dataFileName,
            String labelFileName,
            boolean isNumeric,
            boolean hasMissingValues,
            boolean isRegression
    ) {
        this.dataFileName = dataFileName;
        this.labelFileName = labelFileName;
        this.isNumeric = isNumeric;
        this.hasMissingValues = hasMissingValues;
        this.isRegression = isRegression;
    }

    @Override
    public ListObjectDataset read() throws IOException {

        ListObjectDataset dataset = new ListObjectDataset();

        TSMetadata metadata = inspectMetadata(dataFileName);

        if (metadata.hasEmbeddedLabels && labelFileName != null) {
            throw new IOException(
                    "Ambiguous label source: .ts file declares embedded labels, "
                            + "but a separate label file was also supplied."
            );
        }

        List<Object> separateLabels = new ArrayList<>();

        if (labelFileName != null) {
            separateLabels =
                    DelimitedFileReader.readGenericLabels(
                            labelFileName,
                            false,
                            isRegression
                    );
        }

        int i = 0;
        boolean inDataSection = false;
        long start = System.nanoTime();

        try (BufferedReader br = new BufferedReader(new FileReader(dataFileName))) {

            String line;

            while ((line = br.readLine()) != null) {

                line = line.trim();

                if (line.isEmpty()) {
                    continue;
                }

                if (line.startsWith("#")) {
                    continue;
                }

                String lower = line.toLowerCase();

                if (lower.startsWith("@data")) {
                    inDataSection = true;
                    continue;
                }

                if (!inDataSection) {
                    continue;
                }

                ParsedTSRow parsed =
                        parseTSRow(
                                line,
                                metadata.hasEmbeddedLabels,
                                metadata.isUnivariate,
                                separateLabels,
                                i
                        );

                dataset.add(parsed.label, parsed.data, i);

                updateGlobalLength(parsed.data);

                ProgressLogger.logProgress(i);
                i++;
            }
        }

        long end = System.nanoTime();
        ProgressLogger.logDuration(start, end);

        return dataset;
    }

    private ParsedTSRow parseTSRow(
            String line,
            boolean hasEmbeddedLabel,
            boolean metadataSaysUnivariate,
            List<Object> separateLabels,
            int rowIndex
    ) throws IOException {

        String[] parts = line.split(DIMENSION_SEPARATOR, -1);

        if (parts.length == 0) {
            throw new IOException("Invalid .ts row: empty data row.");
        }

        Object label = null;
        int dimensionCount = parts.length;

        if (hasEmbeddedLabel) {
            if (parts.length < 2) {
                throw new IOException(
                        "Invalid .ts row: embedded label was expected, "
                                + "but the row has no data-label separator."
                );
            }

            String labelToken = parts[parts.length - 1];
            label = parseLabel(labelToken, isRegression);
            dimensionCount = parts.length - 1;

        } else if (labelFileName != null) {
            label = separateLabels.get(rowIndex);
        }

        boolean isActuallyUnivariate =
                metadataSaysUnivariate || dimensionCount == 1;

        if (isActuallyUnivariate) {
            Object data = parseUnivariateDimension(parts[0]);
            return new ParsedTSRow(label, data);
        }

        Object data = parseMultivariateDimensions(parts, dimensionCount);
        return new ParsedTSRow(label, data);
    }

    private Object parseUnivariateDimension(String dimensionString) {

        List<String> tokens = splitValues(dimensionString);

        if (isNumeric) {
            if (hasMissingValues) {
                Double[] values = new Double[tokens.size()];

                for (int i = 0; i < tokens.size(); i++) {
                    values[i] = parseBoxedDoubleToken(extractValue(tokens.get(i)));
                }

                return values;
            }

            double[] values = new double[tokens.size()];

            for (int i = 0; i < tokens.size(); i++) {
                values[i] = Double.parseDouble(extractValue(tokens.get(i)).trim());
            }

            return values;
        }

        Object[] values = new Object[tokens.size()];

        for (int i = 0; i < tokens.size(); i++) {
            values[i] = parseGenericValue(extractValue(tokens.get(i)));
        }

        return values;
    }

    private Object parseMultivariateDimensions(
            String[] parts,
            int dimensionCount
    ) {

        if (isNumeric) {
            if (hasMissingValues) {
                Double[][] matrix = new Double[dimensionCount][];

                for (int d = 0; d < dimensionCount; d++) {
                    List<String> tokens = splitValues(parts[d]);
                    matrix[d] = new Double[tokens.size()];

                    for (int t = 0; t < tokens.size(); t++) {
                        matrix[d][t] =
                                parseBoxedDoubleToken(
                                        extractValue(tokens.get(t))
                                );
                    }
                }

                return matrix;
            }

            double[][] matrix = new double[dimensionCount][];

            for (int d = 0; d < dimensionCount; d++) {
                List<String> tokens = splitValues(parts[d]);
                matrix[d] = new double[tokens.size()];

                for (int t = 0; t < tokens.size(); t++) {
                    matrix[d][t] =
                            Double.parseDouble(
                                    extractValue(tokens.get(t)).trim()
                            );
                }
            }

            return matrix;
        }

        Object[][] matrix = new Object[dimensionCount][];

        for (int d = 0; d < dimensionCount; d++) {
            List<String> tokens = splitValues(parts[d]);
            matrix[d] = new Object[tokens.size()];

            for (int t = 0; t < tokens.size(); t++) {
                matrix[d][t] = parseGenericValue(extractValue(tokens.get(t)));
            }
        }

        return matrix;
    }

    /**
     * Splits a dimension string into values.
     *
     * Handles both:
     *
     *      1,2,3,4
     *
     * and timestamped:
     *
     *      (0,1),(1,2),(2,3)
     *
     * For timestamped data, commas inside parentheses are not treated
     * as value separators.
     */
    private static List<String> splitValues(String dimensionString) {

        List<String> values = new ArrayList<>();

        if (dimensionString == null || dimensionString.trim().isEmpty()) {
            return values;
        }

        int depth = 0;
        StringBuilder current = new StringBuilder();

        for (int i = 0; i < dimensionString.length(); i++) {

            char c = dimensionString.charAt(i);

            if (c == '(') {
                depth++;
                current.append(c);
            } else if (c == ')') {
                depth--;
                current.append(c);
            } else if (c == VALUE_SEPARATOR && depth == 0) {
                values.add(current.toString().trim());
                current.setLength(0);
            } else {
                current.append(c);
            }
        }

        values.add(current.toString().trim());

        return values;
    }

    /**
     * Extracts the actual data value from either a raw value token or a
     * timestamped tuple.
     *
     * Examples:
     *
     *      "3.14"      -> "3.14"
     *      "(10,3.14)" -> "3.14"
     */
    private static String extractValue(String token) {

        if (token == null) {
            return null;
        }

        String trimmed = token.trim();

        if (trimmed.startsWith("(") && trimmed.endsWith(")")) {
            String inner = trimmed.substring(1, trimmed.length() - 1);
            int commaIndex = inner.lastIndexOf(',');

            if (commaIndex >= 0 && commaIndex < inner.length() - 1) {
                return inner.substring(commaIndex + 1).trim();
            }
        }

        return trimmed;
    }

    private static Double parseBoxedDoubleToken(String token) {

        if (token == null) {
            return null;
        }

        String trimmed = token.trim();

        if (MissingValueParser.isMissing(trimmed)) {
            return null;
        }

        return Double.valueOf(trimmed);
    }

    private static Object parseGenericValue(String token) {

        if (token == null) {
            return null;
        }

        String trimmed = token.trim();

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

    private static Object parseLabel(
            String token,
            boolean isRegression
    ) {

        String trimmed = token.trim();

        if (isRegression) {
            return Double.parseDouble(trimmed);
        }

        try {
            return Integer.parseInt(trimmed);
        } catch (NumberFormatException ignored) {

            try {
                double value = Double.parseDouble(trimmed);

                if (value == Math.rint(value)) {
                    return (int) value;
                }

                return value;

            } catch (NumberFormatException ignoredAgain) {
                return trimmed;
            }
        }
    }

    private static void updateGlobalLength(Object data) {

        if (data instanceof double[]) {
            AppContext.length = ((double[]) data).length;

        } else if (data instanceof Double[]) {
            AppContext.length = ((Double[]) data).length;

        } else if (data instanceof double[][]) {
            double[][] matrix = (double[][]) data;

            if (matrix.length > 0) {
                AppContext.length = matrix[0].length;
            }

        } else if (data instanceof Double[][]) {
            Double[][] matrix = (Double[][]) data;

            if (matrix.length > 0) {
                AppContext.length = matrix[0].length;
            }

        } else if (data instanceof Object[][]) {
            Object[][] matrix = (Object[][]) data;

            if (matrix.length > 0) {
                AppContext.length = matrix[0].length;
            }

        } else if (data instanceof Object[]) {
            AppContext.length = ((Object[]) data).length;
        }
    }

    private static TSMetadata inspectMetadata(String dataFileName)
            throws IOException {

        TSMetadata metadata = new TSMetadata();

        try (BufferedReader br = new BufferedReader(new FileReader(dataFileName))) {

            String line;

            while ((line = br.readLine()) != null) {

                line = line.trim();

                if (line.isEmpty()) {
                    continue;
                }

                String lower = line.toLowerCase();

                if (lower.startsWith("@data")) {
                    break;
                }

                if (lower.startsWith("@classlabel")) {
                    metadata.hasEmbeddedLabels = parseClassLabelLine(line);
                } else if (lower.startsWith("@univariate")) {
                    metadata.isUnivariate = parseBooleanMetadataLine(line);
                } else if (lower.startsWith("@timestamps")) {
                    metadata.hasTimestamps = parseBooleanMetadataLine(line);
                }
            }
        }

        return metadata;
    }

    private static boolean parseClassLabelLine(String line) {

        String[] tokens = line.trim().split("\\s+");

        if (tokens.length < 2) {
            return false;
        }

        return Boolean.parseBoolean(tokens[1]);
    }

    private static boolean parseBooleanMetadataLine(String line) {

        String[] tokens = line.trim().split("\\s+");

        if (tokens.length < 2) {
            return false;
        }

        return Boolean.parseBoolean(tokens[1]);
    }

    private static class TSMetadata {

        private boolean hasEmbeddedLabels = false;
        private boolean isUnivariate = false;
        private boolean hasTimestamps = false;
    }

    private static class ParsedTSRow {

        private final Object label;
        private final Object data;

        private ParsedTSRow(Object label, Object data) {
            this.label = label;
            this.data = data;
        }
    }

    private static class MissingValueParser {

        private static Set<String> missingIndicators =
                AppContext.MissingStrings;

        public static void setMissingIndicators(Set<String> indicators) {
            missingIndicators =
                    indicators.stream()
                            .map(String::toUpperCase)
                            .collect(Collectors.toSet());
        }

        public static boolean isMissing(String token) {

            if (token == null) {
                return true;
            }

            return missingIndicators.contains(token.trim().toUpperCase());
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

                        System.out.print(i + ":" + usedMem / 1024 / 1024 + "mb\n");
                    }

                } else {
                    System.out.print(".");
                }
            }
        }

        public static void logDuration(long start, long end) {

            long elapsed = end - start;

            String timeDuration =
                    DurationFormatUtils.formatDuration(
                            (long) (elapsed / 1e6),
                            "H:m:s.SSS"
                    );

            System.out.println("finished in " + timeDuration);
        }
    }
}