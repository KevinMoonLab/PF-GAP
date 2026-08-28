package datasets.readers.api;

import java.io.Serial;
import java.io.Serializable;
import java.nio.file.Path;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;

/**
 * Immutable configuration supplied to a user-defined reader.
 *
 * <p>This context intentionally exposes only stable reader-facing
 * information rather than the complete internal ReaderOptions object.
 * Custom readers are therefore insulated from changes to PFGAP's internal
 * reader configuration.</p>
 *
 * <p>Format-specific settings are supplied through a string parameter map.
 * Typical parameters for a proprietary binary reader might include:</p>
 *
 * <pre>
 * byte_order   = LITTLE_ENDIAN
 * header_bytes = 128
 * data_type    = FLOAT32
 * dimensions   = 12
 * time_length  = 5000
 * </pre>
 *
 * <p>The context is serializable so that its configuration can be retained
 * as part of lazy-reader reconstruction metadata. Runtime plugin objects,
 * class loaders, open files, memory mappings, and caches should not be stored
 * in this object.</p>
 */
public final class CustomReaderContext
        implements Serializable {

    @Serial
    private static final long serialVersionUID =
            1L;

    private final Path dataPath;
    private final boolean test;
    private final boolean regression;
    private final boolean numeric;
    private final boolean hasMissingValues;
    private final Map<String, String> parameters;

    /**
     * Creates an immutable custom-reader context.
     *
     * @param dataPath          configured dataset path, or null when the
     *                          instance reference supplies all location
     *                          information
     * @param test              whether this context is for test data
     * @param regression        whether the experiment is a regression task
     * @param numeric           whether numerical output is expected
     * @param hasMissingValues  whether missing values are expected
     * @param parameters        reader-specific string parameters
     */
    public CustomReaderContext(
            Path dataPath,
            boolean test,
            boolean regression,
            boolean numeric,
            boolean hasMissingValues,
            Map<String, String> parameters
    ) {
        this.dataPath =
                dataPath;

        this.test =
                test;

        this.regression =
                regression;

        this.numeric =
                numeric;

        this.hasMissingValues =
                hasMissingValues;

        this.parameters =
                normalizeParameters(
                        parameters
                );
    }

    /**
     * Creates a context without reader-specific parameters.
     */
    public CustomReaderContext(
            Path dataPath,
            boolean test,
            boolean regression,
            boolean numeric,
            boolean hasMissingValues
    ) {
        this(
                dataPath,
                test,
                regression,
                numeric,
                hasMissingValues,
                Map.of()
        );
    }

    /**
     * Returns the configured dataset-level path.
     *
     * <p>For a per-file custom reader, the concrete instance file should
     * normally be obtained from the supplied LazySeriesRef instead. This path
     * is useful for dataset-relative resources such as metadata files,
     * dictionaries, or shared schemas.</p>
     *
     * @return dataset-level path, possibly null
     */
    public Path getDataPath() {
        return dataPath;
    }

    /**
     * Returns the configured dataset path or throws when none is available.
     *
     * @return non-null dataset path
     */
    public Path requireDataPath() {
        if (dataPath == null) {
            throw new IllegalStateException(
                    "This custom reader requires a dataset path, but no "
                            + "dataPath was supplied."
            );
        }

        return dataPath;
    }

    public boolean isTest() {
        return test;
    }

    public boolean isRegression() {
        return regression;
    }

    public boolean isNumeric() {
        return numeric;
    }

    public boolean hasMissingValues() {
        return hasMissingValues;
    }

    /**
     * Returns an unmodifiable map of all reader-specific parameters.
     *
     * @return immutable parameter map
     */
    public Map<String, String> getParameters() {
        return parameters;
    }

    /**
     * Returns whether a nonblank parameter is configured.
     *
     * @param name parameter name
     * @return true when the parameter exists and is nonblank
     */
    public boolean hasParameter(
            String name
    ) {
        String value =
                parameters.get(
                        normalizeParameterName(
                                name
                        )
                );

        return value != null
                && !value.isBlank();
    }

    /**
     * Returns a parameter value or null when it is absent.
     *
     * @param name parameter name
     * @return parameter value, possibly null
     */
    public String getParameter(
            String name
    ) {
        return parameters.get(
                normalizeParameterName(
                        name
                )
        );
    }

    /**
     * Returns a parameter value or the supplied default.
     *
     * @param name         parameter name
     * @param defaultValue value returned when the parameter is absent
     * @return configured or default value
     */
    public String getParameter(
            String name,
            String defaultValue
    ) {
        String value =
                getParameter(
                        name
                );

        return value == null
                ? defaultValue
                : value;
    }

    /**
     * Returns a required nonblank parameter.
     *
     * @param name parameter name
     * @return configured parameter value
     */
    public String requireParameter(
            String name
    ) {
        String normalizedName =
                normalizeParameterName(
                        name
                );

        String value =
                parameters.get(
                        normalizedName
                );

        if (value == null || value.isBlank()) {
            throw new IllegalArgumentException(
                    "Required custom-reader parameter is missing or blank: "
                            + normalizedName
            );
        }

        return value;
    }

    public int getIntParameter(
            String name,
            int defaultValue
    ) {
        String value =
                getParameter(
                        name
                );

        return value == null
                ? defaultValue
                : parseInt(
                name,
                value
        );
    }

    public int requireIntParameter(
            String name
    ) {
        return parseInt(
                name,
                requireParameter(name)
        );
    }

    public long getLongParameter(
            String name,
            long defaultValue
    ) {
        String value =
                getParameter(
                        name
                );

        return value == null
                ? defaultValue
                : parseLong(
                name,
                value
        );
    }

    public long requireLongParameter(
            String name
    ) {
        return parseLong(
                name,
                requireParameter(name)
        );
    }

    public double getDoubleParameter(
            String name,
            double defaultValue
    ) {
        String value =
                getParameter(
                        name
                );

        return value == null
                ? defaultValue
                : parseDouble(
                name,
                value
        );
    }

    public double requireDoubleParameter(
            String name
    ) {
        return parseDouble(
                name,
                requireParameter(name)
        );
    }

    public boolean getBooleanParameter(
            String name,
            boolean defaultValue
    ) {
        String value =
                getParameter(
                        name
                );

        return value == null
                ? defaultValue
                : parseBoolean(
                name,
                value
        );
    }

    public boolean requireBooleanParameter(
            String name
    ) {
        return parseBoolean(
                name,
                requireParameter(name)
        );
    }

    /**
     * Reads an enum parameter without making the context depend on a
     * specific custom-reader enum type.
     *
     * <p>Matching is case-insensitive and surrounding whitespace is ignored.
     * Hyphens are normalized to underscores. For example,
     * {@code little-endian} can match {@code LITTLE_ENDIAN}.</p>
     *
     * @param name         parameter name
     * @param enumType     enum class
     * @param defaultValue value returned when the parameter is absent
     * @param <E>          enum type
     * @return configured or default enum value
     */
    public <E extends Enum<E>> E getEnumParameter(
            String name,
            Class<E> enumType,
            E defaultValue
    ) {
        String value =
                getParameter(
                        name
                );

        if (value == null) {
            return defaultValue;
        }

        return parseEnum(
                name,
                value,
                enumType
        );
    }

    public <E extends Enum<E>> E requireEnumParameter(
            String name,
            Class<E> enumType
    ) {
        return parseEnum(
                name,
                requireParameter(name),
                enumType
        );
    }

    /**
     * Resolves a configured path parameter.
     *
     * <p>Absolute values are returned unchanged. Relative values are resolved
     * against {@code dataPath} when it represents a directory. When
     * {@code dataPath} represents a file-like path, the value is resolved
     * against its parent directory. If no data path is configured, the
     * relative value is returned as a normalized relative path.</p>
     *
     * @param name parameter containing a path
     * @return resolved path
     */
    public Path requirePathParameter(
            String name
    ) {
        String value =
                requireParameter(
                        name
                );

        Path path =
                Path.of(
                        value
                );

        if (path.isAbsolute()) {
            return path.normalize();
        }

        if (dataPath == null) {
            return path.normalize();
        }

        Path base =
                dataPath;

        /*
         * The path may not exist yet, so resolution cannot rely entirely on
         * Files.isDirectory(). A path with a parent and a filename extension
         * is still treated naturally by callers through explicit absolute
         * parameters when ambiguity matters.
         */
        if (dataPath.getParent() != null
                && dataPath.getFileName() != null
                && dataPath.getFileName()
                .toString()
                .contains(".")) {

            base =
                    dataPath.getParent();
        }

        return base.resolve(path)
                .normalize();
    }

    private static Map<String, String> normalizeParameters(
            Map<String, String> parameters
    ) {
        if (parameters == null || parameters.isEmpty()) {
            return Map.of();
        }

        Map<String, String> normalized =
                new LinkedHashMap<>(
                        Math.max(
                                16,
                                parameters.size() * 2
                        )
                );

        for (Map.Entry<String, String> entry
                : parameters.entrySet()) {

            String name =
                    normalizeParameterName(
                            entry.getKey()
                    );

            String value =
                    entry.getValue();

            if (normalized.containsKey(name)) {
                throw new IllegalArgumentException(
                        "Duplicate custom-reader parameter after name "
                                + "normalization: "
                                + name
                );
            }

            normalized.put(
                    name,
                    value == null
                            ? null
                            : value.trim()
            );
        }

        return Collections.unmodifiableMap(
                normalized
        );
    }

    /**
     * Parameter names are normalized to lower case so Python, JSON, and Java
     * callers do not have to coordinate exact capitalization.
     */
    private static String normalizeParameterName(
            String name
    ) {
        Objects.requireNonNull(
                name,
                "Custom-reader parameter name cannot be null."
        );

        String normalized =
                name.trim()
                        .toLowerCase(
                                Locale.ROOT
                        );

        if (normalized.isEmpty()) {
            throw new IllegalArgumentException(
                    "Custom-reader parameter name cannot be blank."
            );
        }

        return normalized;
    }

    private static int parseInt(
            String name,
            String value
    ) {
        try {
            return Integer.parseInt(
                    value.trim()
            );
        } catch (NumberFormatException e) {
            throw invalidParameter(
                    name,
                    value,
                    "integer",
                    e
            );
        }
    }

    private static long parseLong(
            String name,
            String value
    ) {
        try {
            return Long.parseLong(
                    value.trim()
            );
        } catch (NumberFormatException e) {
            throw invalidParameter(
                    name,
                    value,
                    "long integer",
                    e
            );
        }
    }

    private static double parseDouble(
            String name,
            String value
    ) {
        try {
            return Double.parseDouble(
                    value.trim()
            );
        } catch (NumberFormatException e) {
            throw invalidParameter(
                    name,
                    value,
                    "double",
                    e
            );
        }
    }

    private static boolean parseBoolean(
            String name,
            String value
    ) {
        String normalized =
                value.trim()
                        .toLowerCase(
                                Locale.ROOT
                        );

        return switch (normalized) {
            case "true", "yes", "1", "on" ->
                    true;

            case "false", "no", "0", "off" ->
                    false;

            default ->
                    throw invalidParameter(
                            name,
                            value,
                            "boolean",
                            null
                    );
        };
    }

    private static <E extends Enum<E>> E parseEnum(
            String name,
            String value,
            Class<E> enumType
    ) {
        Objects.requireNonNull(
                enumType,
                "Custom-reader enum type cannot be null."
        );

        String normalized =
                value.trim()
                        .replace(
                                '-',
                                '_'
                        )
                        .replace(
                                ' ',
                                '_'
                        )
                        .toUpperCase(
                                Locale.ROOT
                        );

        try {
            return Enum.valueOf(
                    enumType,
                    normalized
            );
        } catch (IllegalArgumentException e) {
            throw new IllegalArgumentException(
                    "Invalid value '"
                            + value
                            + "' for custom-reader parameter '"
                            + name
                            + "'. Expected one of "
                            + java.util.Arrays.toString(
                            enumType.getEnumConstants()
                    )
                            + ".",
                    e
            );
        }
    }

    private static IllegalArgumentException invalidParameter(
            String name,
            String value,
            String expectedType,
            Exception cause
    ) {
        String message =
                "Invalid value '"
                        + value
                        + "' for custom-reader parameter '"
                        + normalizeParameterName(name)
                        + "'. Expected "
                        + expectedType
                        + ".";

        return cause == null
                ? new IllegalArgumentException(
                message
        )
                : new IllegalArgumentException(
                message,
                cause
        );
    }

    @Override
    public String toString() {
        return "CustomReaderContext{"
                + "dataPath="
                + dataPath
                + ", test="
                + test
                + ", regression="
                + regression
                + ", numeric="
                + numeric
                + ", hasMissingValues="
                + hasMissingValues
                + ", parameterNames="
                + parameters.keySet()
                + '}';
    }
}