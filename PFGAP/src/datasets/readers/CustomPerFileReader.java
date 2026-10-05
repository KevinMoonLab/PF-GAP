package datasets.readers;

import core.AppContext;
import datasets.ListObjectDataset;
import datasets.NumericStorageType;
import datasets.readers.api.CustomReaderContext;
import datasets.readers.interop.JavaReaderLoader;
import datasets.readers.interop.JavaSeriesReader;
import datasets.readers.lazy.LazySeriesRef;
import preprocessing.standardization.StandardizationStats;
import preprocessing.standardization.Standardizer;

import java.io.IOException;
import java.lang.reflect.Array;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.stream.Stream;

/**
 * Eager per-file dataset reader backed by a user-supplied Java plugin.
 *
 * <p>PFGAP discovers and orders instance files, constructs references,
 * invokes the plugin, validates conventional matrix outputs, applies optional
 * built-in standardization, assembles the dataset, and closes the plugin and
 * class loader.</p>
 *
 * <p>Standard per-file observations must preserve the dimension axis and
 * return {@code float[dimension][time]},
 * {@code double[dimension][time]}, or
 * {@code Object[dimension][time]}. A univariate series remains
 * {@code [1][time]}; the general one-dimensional PFGAP representations are
 * intentionally rejected by this reader. Proprietary nonstandard
 * representations remain valid when standardization is disabled and
 * downstream consumers understand them.</p>
 */
public final class CustomPerFileReader implements DatasetReader {
    private static final String EAGER_READER_KEY = "custom-eager";

    private final String dataPath;
    private final String filePattern;
    private final String customReaderDescriptor;
    private final Map<String, String> customReaderParameters;
    private final List<String> featureColumns;
    private final boolean isTest;
    private final boolean isRegression;
    private final boolean isNumeric;
    private final boolean hasMissingValues;
    private final boolean customReaderThreadSafe;
    private final NumericStorageType numericStorageType;
    private final StandardizationStats standardizationStats;

    public CustomPerFileReader(ReaderOptions options) {
        this(requireOptions(options).getDataPath(),
                options.getFilePattern(),
                options.getCustomReaderDescriptor(),
                options.getCustomReaderParameters(),
                options.getFeatureColumns(),
                options.isTest(),
                options.isRegression(),
                options.isNumeric(),
                options.hasMissingValues(),
                options.isCustomReaderThreadSafe(),
                options.getNumericStorageType(),
                options.getStandardizationStats());
    }

    public CustomPerFileReader(
            String dataPath,
            String filePattern,
            String customReaderDescriptor,
            Map<String, String> customReaderParameters,
            boolean isTest,
            boolean isRegression,
            boolean isNumeric,
            boolean hasMissingValues
    ) {
        this(dataPath, filePattern, customReaderDescriptor,
                customReaderParameters, List.of(), isTest, isRegression,
                isNumeric, hasMissingValues, false, NumericStorageType.AUTO,
                null);
    }

    public CustomPerFileReader(
            String dataPath,
            String filePattern,
            String customReaderDescriptor,
            Map<String, String> customReaderParameters,
            List<String> featureColumns,
            boolean isTest,
            boolean isRegression,
            boolean isNumeric,
            boolean hasMissingValues,
            boolean customReaderThreadSafe
    ) {
        this(dataPath, filePattern, customReaderDescriptor,
                customReaderParameters, featureColumns, isTest, isRegression,
                isNumeric, hasMissingValues, customReaderThreadSafe,
                NumericStorageType.AUTO, null);
    }

    public CustomPerFileReader(
            String dataPath,
            String filePattern,
            String customReaderDescriptor,
            Map<String, String> customReaderParameters,
            boolean isTest,
            boolean isRegression,
            boolean isNumeric,
            boolean hasMissingValues,
            boolean customReaderThreadSafe
    ) {
        this(dataPath, filePattern, customReaderDescriptor,
                customReaderParameters, List.of(), isTest, isRegression,
                isNumeric, hasMissingValues, customReaderThreadSafe,
                NumericStorageType.AUTO, null);
    }

    /** Complete constructor. */
    public CustomPerFileReader(
            String dataPath,
            String filePattern,
            String customReaderDescriptor,
            Map<String, String> customReaderParameters,
            List<String> featureColumns,
            boolean isTest,
            boolean isRegression,
            boolean isNumeric,
            boolean hasMissingValues,
            boolean customReaderThreadSafe,
            NumericStorageType numericStorageType,
            StandardizationStats standardizationStats
    ) {
        this.dataPath = requireNonblank(dataPath, "dataPath");
        this.filePattern = normalizeNullableString(filePattern);
        this.customReaderDescriptor = JavaReaderLoader.normalizeDescriptor(
                requireNonblank(customReaderDescriptor,
                        "customReaderDescriptor"));
        this.customReaderParameters = normalizeParameters(
                customReaderParameters);
        this.featureColumns = copyFeatureColumns(featureColumns);
        this.isTest = isTest;
        this.isRegression = isRegression;
        this.isNumeric = isNumeric;
        this.hasMissingValues = hasMissingValues;
        this.customReaderThreadSafe = customReaderThreadSafe;
        this.numericStorageType = numericStorageType == null
                ? NumericStorageType.AUTO : numericStorageType;
        this.standardizationStats = standardizationStats;

        if (standardizationStats != null) {
            if (!isNumeric) {
                throw new IllegalArgumentException(
                        "Custom-reader standardization requires numeric data.");
            }
            if (!this.featureColumns.isEmpty()) {
                standardizationStats.validateFeatureCompatibility(
                        this.featureColumns);
            }
        }
    }

    @Override
    public ListObjectDataset read() throws IOException {
        List<Path> files = discoverFiles();
        Path configuredDataPath = Paths.get(dataPath)
                .toAbsolutePath().normalize();
        CustomReaderContext context = new CustomReaderContext(
                configuredDataPath,
                isTest,
                isRegression,
                isNumeric,
                hasMissingValues,
                featureColumns,
                customReaderParameters,
                numericStorageType
        );

        JavaSeriesReader javaReader = loadJavaReader(context);
        Throwable readingFailure = null;
        try {
            return materializeDataset(files, javaReader);
        } catch (RuntimeException | Error failure) {
            readingFailure = failure;
            throw failure;
        } finally {
            closeJavaReader(javaReader, readingFailure);
        }
    }

    private ListObjectDataset materializeDataset(
            List<Path> files,
            JavaSeriesReader javaReader
    ) {
        ListObjectDataset dataset = new ListObjectDataset(files.size());
        int commonLength = -1;
        boolean unequalLengths = false;
        boolean unknownLength = false;

        for (int instanceIndex = 0;
             instanceIndex < files.size();
             instanceIndex++) {
            Path file = files.get(instanceIndex);
            LazySeriesRef reference = new LazySeriesRef(
                    EAGER_READER_KEY, instanceIndex, file);
            Object instance = javaReader.read(reference);
            validateInstance(instance, file, instanceIndex);

            if (standardizationStats != null) {
                validateStandardizableInstance(
                        instance, file, instanceIndex);
                Standardizer.transformInstanceInPlace(
                        instance, standardizationStats);
            }

            dataset.add(inferLabel(file, instanceIndex),
                    instance, instanceIndex);

            int instanceLength = inferInstanceLength(instance);
            if (instanceLength < 0) {
                unknownLength = true;
            } else if (commonLength < 0) {
                commonLength = instanceLength;
            } else if (instanceLength != commonLength) {
                unequalLengths = true;
            }

            if (instanceIndex > 0) {
                DelimitedFileReader.ProgressLogger.logProgress(instanceIndex);
            }
        }

        int datasetLength = unknownLength || unequalLengths || commonLength < 0
                ? 0 : commonLength;
        dataset.setLength(datasetLength);
        AppContext.length = datasetLength;
        return dataset;
    }

    private JavaSeriesReader loadJavaReader(CustomReaderContext context)
            throws IOException {
        try {
            return new JavaSeriesReader(customReaderDescriptor, context,
                    customReaderThreadSafe);
        } catch (ReflectiveOperationException exception) {
            throw new IOException(
                    "Could not load custom per-file reader from descriptor: "
                            + customReaderDescriptor,
                    exception);
        }
    }

    private void closeJavaReader(
            JavaSeriesReader javaReader,
            Throwable readingFailure
    ) throws IOException {
        try {
            javaReader.close();
        } catch (Throwable closeFailure) {
            if (readingFailure != null) {
                readingFailure.addSuppressed(closeFailure);
                return;
            }
            if (closeFailure instanceof Error error) {
                throw error;
            }
            throw new IOException(
                    "Failed to close custom per-file reader "
                            + javaReader.getImplementationClassName() + ".",
                    closeFailure);
        }
    }

    private void validateInstance(
            Object instance,
            Path file,
            int instanceIndex
    ) {
        if (instance == null) {
            throw invalidInstance("returned null", file, instanceIndex);
        }
        if (instance.getClass().isArray()
                && Array.getLength(instance) == 0) {
            throw invalidInstance("returned an empty array", file,
                    instanceIndex);
        }
        rejectOneDimensionalStandardRepresentation(
                instance, file, instanceIndex);
        validateRecognizedMatrixShape(instance, file, instanceIndex);
    }

    /**
     * Enforces the per-file time-series invariant for standard PFGAP arrays.
     *
     * <p>One-dimensional arrays are valid in the general dataset contract,
     * but a per-file observation always preserves the dimension axis. A
     * univariate series must therefore be represented as {@code [1][time]}.
     * Proprietary nonstandard representations remain available when built-in
     * standardization is disabled and downstream consumers support them.</p>
     */
    private void rejectOneDimensionalStandardRepresentation(
            Object instance,
            Path file,
            int instanceIndex
    ) {
        if (instance instanceof float[]
                || instance instanceof double[]
                || instance instanceof Object[]) {
            throw invalidInstance(
                    "returned a one-dimensional standard representation; "
                            + "per-file readers require a two-dimensional "
                            + "dimension-major array such as [1][time]",
                    file,
                    instanceIndex);
        }
    }

    private void validateStandardizableInstance(
            Object instance,
            Path file,
            int instanceIndex
    ) {
        if (!(instance instanceof float[][])
                && !(instance instanceof double[][])) {
            throw invalidInstance(
                    "must return float[][] or double[][] when built-in "
                            + "standardization is enabled",
                    file,
                    instanceIndex);
        }
    }

    private void validateRecognizedMatrixShape(
            Object instance,
            Path file,
            int instanceIndex
    ) {
        if (!(instance instanceof float[][])
                && !(instance instanceof double[][])
                && !(instance instanceof Object[][])) {
            return;
        }
        int dimensions = Array.getLength(instance);
        if (dimensions == 0) {
            throw invalidInstance("returned zero dimensions", file,
                    instanceIndex);
        }
        Object first = Array.get(instance, 0);
        if (first == null || Array.getLength(first) == 0) {
            throw invalidInstance(
                    "returned a null or empty first dimension", file,
                    instanceIndex);
        }
        int expectedLength = Array.getLength(first);
        for (int dimension = 1; dimension < dimensions; dimension++) {
            Object row = Array.get(instance, dimension);
            if (row == null) {
                throw invalidInstance(
                        "returned a null dimension " + dimension,
                        file, instanceIndex);
            }
            int actualLength = Array.getLength(row);
            if (actualLength != expectedLength) {
                throw invalidInstance(
                        "returned a nonrectangular array: expected dimension "
                                + "length " + expectedLength
                                + " but dimension " + dimension
                                + " has length " + actualLength,
                        file, instanceIndex);
            }
        }
    }

    private IllegalStateException invalidInstance(
            String problem,
            Path file,
            int instanceIndex
    ) {
        return new IllegalStateException(
                "Custom reader " + problem + " for instance "
                        + instanceIndex + " from file " + file + ".");
    }

    private int inferInstanceLength(Object instance) {
        if (instance instanceof float[][] values) {
            return values.length == 0 ? 0 : values[0].length;
        }
        if (instance instanceof double[][] values) {
            return values.length == 0 ? 0 : values[0].length;
        }
        if (instance instanceof Object[][] values) {
            return values.length == 0 ? 0 : values[0].length;
        }
        return -1;
    }

    private Object inferLabel(Path file, int instanceIndex) {
        return null;
    }

    private List<Path> discoverFiles() throws IOException {
        Path path = Paths.get(dataPath).toAbsolutePath().normalize();
        if (!Files.exists(path) || !Files.isReadable(path)) {
            throw new IOException(
                    "Custom per-file data path is not readable: " + path);
        }
        if (Files.isRegularFile(path)) {
            return List.of(path);
        }
        if (!Files.isDirectory(path)) {
            throw new IOException(
                    "Custom per-file data path must be a regular file or "
                            + "directory: " + path);
        }
        if (filePattern == null) {
            throw new IllegalArgumentException(
                    "CustomPerFileReader requires filePattern when dataPath "
                            + "is a directory.");
        }
        return discoverFromPattern(path, filePattern);
    }

    private List<Path> discoverFromPattern(
            Path directory,
            String patternText
    ) throws IOException {
        NumericPattern pattern = NumericPattern.from(patternText);
        List<IndexedPath> indexed = new ArrayList<>();
        try (Stream<Path> stream = Files.list(directory)) {
            stream.filter(Files::isRegularFile).forEach(path -> {
                String fileName = path.getFileName().toString();
                Long sequence = pattern.tryExtractNumber(fileName);
                if (sequence != null) {
                    indexed.add(new IndexedPath(path, fileName, sequence));
                }
            });
        }
        if (indexed.isEmpty()) {
            throw new IOException(
                    "No files in directory " + directory
                            + " matched custom per-file pattern: "
                            + patternText);
        }
        indexed.sort(Comparator
                .comparingLong(IndexedPath::sequenceNumber)
                .thenComparing(IndexedPath::fileName));
        List<Path> files = new ArrayList<>(indexed.size());
        for (IndexedPath value : indexed) {
            files.add(value.path());
        }
        return List.copyOf(files);
    }

    private static ReaderOptions requireOptions(ReaderOptions options) {
        if (options == null) {
            throw new IllegalArgumentException(
                    "CustomPerFileReader requires non-null ReaderOptions.");
        }
        return options;
    }

    private static String requireNonblank(
            String value,
            String argumentName
    ) {
        if (value == null || value.isBlank()) {
            throw new IllegalArgumentException(
                    "CustomPerFileReader requires " + argumentName + ".");
        }
        return value.trim();
    }

    private static List<String> copyFeatureColumns(List<String> values) {
        if (values == null || values.isEmpty()) {
            return List.of();
        }
        List<String> copy = new ArrayList<>(values.size());
        Set<String> used = new HashSet<>();
        for (String value : values) {
            if (value == null || value.isBlank()) {
                throw new IllegalArgumentException(
                        "Custom-reader feature columns cannot be blank.");
            }
            String normalized = value.trim();
            if (!used.add(normalized)) {
                throw new IllegalArgumentException(
                        "Duplicate custom-reader feature column: "
                                + normalized);
            }
            copy.add(normalized);
        }
        return List.copyOf(copy);
    }

    private static Map<String, String> normalizeParameters(
            Map<String, String> parameters
    ) {
        if (parameters == null || parameters.isEmpty()) {
            return Map.of();
        }
        java.util.LinkedHashMap<String, String> normalized =
                new java.util.LinkedHashMap<>();
        for (Map.Entry<String, String> entry : parameters.entrySet()) {
            if (entry.getKey() == null || entry.getKey().isBlank()) {
                throw new IllegalArgumentException(
                        "Custom-reader parameter names cannot be blank.");
            }
            String name = entry.getKey().trim().toLowerCase(
                    java.util.Locale.ROOT);
            if (normalized.putIfAbsent(name, entry.getValue()) != null) {
                throw new IllegalArgumentException(
                        "Duplicate custom-reader parameter after "
                                + "normalization: " + name);
            }
        }
        return Map.copyOf(normalized);
    }

    private static String normalizeNullableString(String value) {
        if (value == null) {
            return null;
        }
        String trimmed = value.trim();
        return trimmed.isEmpty() || trimmed.equalsIgnoreCase("None")
                ? null : trimmed;
    }

    private record IndexedPath(
            Path path,
            String fileName,
            long sequenceNumber
    ) {
    }

    private static final class NumericPattern {
        private static final Pattern PLACEHOLDER = Pattern.compile(
                "\\{([A-Za-z_][A-Za-z0-9_]*)(?::0?(\\d+)d)?}");

        private final Pattern regex;
        private final String fieldName;

        private NumericPattern(Pattern regex, String fieldName) {
            this.regex = regex;
            this.fieldName = fieldName;
        }

        private static NumericPattern from(String pattern) {
            if (pattern == null || pattern.isBlank()) {
                throw new IllegalArgumentException(
                        "Custom per-file pattern cannot be blank.");
            }
            Matcher matcher = PLACEHOLDER.matcher(pattern);
            if (!matcher.find()) {
                throw new IllegalArgumentException(
                        "Custom per-file pattern must contain exactly one "
                                + "numeric placeholder: " + pattern);
            }
            String field = matcher.group(1);
            String width = matcher.group(2);
            int start = matcher.start();
            int end = matcher.end();
            if (matcher.find()) {
                throw new IllegalArgumentException(
                        "Custom per-file pattern supports exactly one numeric "
                                + "placeholder: " + pattern);
            }
            String number = width == null
                    ? "(\\d+)"
                    : "(\\d{" + Integer.parseInt(width) + "})";
            String regex = "^"
                    + globFragment(pattern.substring(0, start))
                    + number
                    + globFragment(pattern.substring(end)) + "$";
            return new NumericPattern(Pattern.compile(regex), field);
        }

        private Long tryExtractNumber(String fileName) {
            Matcher matcher = regex.matcher(fileName);
            if (!matcher.matches()) {
                return null;
            }
            try {
                return Long.parseLong(matcher.group(1));
            } catch (NumberFormatException exception) {
                throw new IllegalArgumentException(
                        "Numeric field '" + fieldName
                                + "' is too large in filename: " + fileName,
                        exception);
            }
        }

        private static String globFragment(String fragment) {
            StringBuilder regex = new StringBuilder();
            StringBuilder literal = new StringBuilder();
            for (int index = 0; index < fragment.length(); index++) {
                char current = fragment.charAt(index);
                if (current == '*' || current == '?') {
                    appendLiteral(regex, literal);
                    regex.append(current == '*' ? ".*" : ".");
                } else {
                    literal.append(current);
                }
            }
            appendLiteral(regex, literal);
            return regex.toString();
        }

        private static void appendLiteral(
                StringBuilder regex,
                StringBuilder literal
        ) {
            if (literal.length() > 0) {
                regex.append(Pattern.quote(literal.toString()));
                literal.setLength(0);
            }
        }
    }
}
