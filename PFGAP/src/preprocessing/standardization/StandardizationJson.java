package preprocessing.standardization;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import com.google.gson.JsonArray;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;

import java.io.IOException;
import java.io.Reader;
import java.io.Writer;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Locale;
import java.util.Objects;
import java.util.Set;

/**
 * Reads and writes fitted standardization statistics as JSON.
 *
 * The writer always uses PFGAP's canonical field names:
 *
 *     version
 *     method
 *     scope
 *     varianceConvention
 *     featureNames
 *     counts
 *     centers
 *     scales
 *
 * The reader accepts several common aliases, including:
 *
 *     mean / means       -> centers
 *     std / stds         -> scales
 *     columns / features -> featureNames
 *
 * Missing metadata can be supplied by StandardizationConfig and the current
 * ordered reader feature names.
 *
 * Positional alignment is permitted for PER_DIMENSION statistics when
 * neither the JSON file nor the current reader configuration contains
 * feature names. In that case, center and scale element i are assumed to
 * describe data dimension i.
 */
public final class StandardizationJson {

    private static final String[] CENTER_ALIASES = {
            "centers",
            "center",
            "means",
            "mean",
            "averages",
            "average"
    };

    private static final String[] SCALE_ALIASES = {
            "scales",
            "scale",
            "standardDeviations",
            "standardDeviation",
            "standard_deviations",
            "standard_deviation",
            "stds",
            "std"
    };

    private static final String[] FEATURE_ALIASES = {
            "featureNames",
            "feature_names",
            "features",
            "columns",
            "columnNames",
            "column_names"
    };

    private static final String[] COUNT_ALIASES = {
            "counts",
            "count",
            "sampleCounts",
            "sample_counts",
            "n"
    };

    private static final String[] VERSION_ALIASES = {
            "version",
            "formatVersion",
            "format_version"
    };

    private static final String[] METHOD_ALIASES = {
            "method",
            "standardization",
            "standardizationMethod",
            "standardization_method"
    };

    private static final String[] SCOPE_ALIASES = {
            "scope",
            "standardizationScope",
            "standardization_scope"
    };

    private static final String[] VARIANCE_ALIASES = {
            "varianceConvention",
            "variance_convention",
            "variance",
            "ddof"
    };

    private static final Gson PRETTY_GSON =
            new GsonBuilder()
                    .setPrettyPrinting()
                    .create();

    private StandardizationJson() {
    }

    /**
     * Reads statistics using the supplied standardization configuration.
     *
     * JSON metadata overrides nothing silently. If the JSON explicitly
     * declares a method, scope, or variance convention that conflicts with
     * the configuration, loading fails.
     *
     * @param path JSON statistics path
     * @param config requested standardization configuration
     * @param fallbackFeatureNames ordered reader feature names, or empty
     * @return validated standardization statistics
     * @throws IOException if the file cannot be read
     */
    public static StandardizationStats read(
            String path,
            StandardizationConfig config,
            List<String> fallbackFeatureNames
    ) throws IOException {
        if (path == null || path.isBlank()) {
            throw new IllegalArgumentException(
                    "Standardization statistics path cannot be null "
                            + "or blank."
            );
        }

        return read(
                Paths.get(path),
                config,
                fallbackFeatureNames
        );
    }

    /**
     * Reads statistics using the supplied standardization configuration.
     *
     * @param path JSON statistics path
     * @param config requested standardization configuration
     * @param fallbackFeatureNames ordered reader feature names, or empty
     * @return validated standardization statistics
     * @throws IOException if the file cannot be read
     */
    public static StandardizationStats read(
            Path path,
            StandardizationConfig config,
            List<String> fallbackFeatureNames
    ) throws IOException {
        Objects.requireNonNull(
                path,
                "Standardization statistics path cannot be null."
        );

        Objects.requireNonNull(
                config,
                "StandardizationConfig cannot be null."
        );

        if (config.isDisabled()) {
            throw new IllegalArgumentException(
                    "Cannot load standardization statistics when "
                            + "standardization is disabled."
            );
        }

        if (!Files.exists(path)) {
            throw new IOException(
                    "Standardization statistics file does not exist: "
                            + path
            );
        }

        if (!Files.isRegularFile(path)) {
            throw new IOException(
                    "Standardization statistics path is not a regular file: "
                            + path
            );
        }

        JsonObject root;

        try (Reader reader =
                     Files.newBufferedReader(
                             path,
                             StandardCharsets.UTF_8
                     )) {

            JsonElement parsed =
                    new JsonParser().parse(reader);

            if (parsed == null || !parsed.isJsonObject()) {
                throw new IllegalArgumentException(
                        "Standardization JSON root must be an object: "
                                + path
                );
            }

            root =
                    parsed.getAsJsonObject();
        }

        return parse(
                root,
                config,
                fallbackFeatureNames,
                path.toString()
        );
    }

    /**
     * Writes statistics using canonical PFGAP field names.
     *
     * @param path destination JSON path
     * @param stats fitted statistics
     * @throws IOException if the file cannot be written
     */
    public static void write(
            String path,
            StandardizationStats stats
    ) throws IOException {
        if (path == null || path.isBlank()) {
            throw new IllegalArgumentException(
                    "Standardization statistics output path cannot be "
                            + "null or blank."
            );
        }

        write(
                Paths.get(path),
                stats
        );
    }

    /**
     * Writes statistics using canonical PFGAP field names.
     *
     * @param path destination JSON path
     * @param stats fitted statistics
     * @throws IOException if the file cannot be written
     */
    public static void write(
            Path path,
            StandardizationStats stats
    ) throws IOException {
        Objects.requireNonNull(
                path,
                "Standardization statistics output path cannot be null."
        );

        Objects.requireNonNull(
                stats,
                "StandardizationStats cannot be null."
        );

        Path parent =
                path.toAbsolutePath()
                        .normalize()
                        .getParent();

        if (parent != null) {
            Files.createDirectories(parent);
        }

        JsonObject root =
                toJsonObject(stats);

        try (Writer writer =
                     Files.newBufferedWriter(
                             path,
                             StandardCharsets.UTF_8
                     )) {

            PRETTY_GSON.toJson(
                    root,
                    writer
            );
        }
    }

    /**
     * Converts fitted statistics to the canonical JSON object.
     */
    public static JsonObject toJsonObject(
            StandardizationStats stats
    ) {
        Objects.requireNonNull(
                stats,
                "StandardizationStats cannot be null."
        );

        JsonObject root =
                new JsonObject();

        root.addProperty(
                "version",
                stats.getFormatVersion()
        );

        root.addProperty(
                "method",
                stats.getMethod().name()
        );

        root.addProperty(
                "scope",
                stats.getScope().name()
        );

        root.addProperty(
                "varianceConvention",
                stats.getVarianceConvention().name()
        );

        if (stats.hasFeatureNames()) {
            root.add(
                    "featureNames",
                    stringArray(
                            stats.getFeatureNames()
                    )
            );
        }

        root.add(
                "counts",
                longArray(
                        stats.getCounts()
                )
        );

        root.add(
                "centers",
                doubleArray(
                        stats.getCenters()
                )
        );

        root.add(
                "scales",
                doubleArray(
                        stats.getScales()
                )
        );

        return root;
    }

    private static StandardizationStats parse(
            JsonObject root,
            StandardizationConfig config,
            List<String> fallbackFeatureNames,
            String sourceDescription
    ) {
        int version =
                parseVersion(
                        root
                );

        StandardizationMethod method =
                parseMethod(
                        root,
                        config
                );

        StandardizationScope scope =
                parseScope(
                        root,
                        config
                );

        VarianceConvention varianceConvention =
                parseVarianceConvention(
                        root,
                        config
                );

        double[] centers =
                requireDoubleValues(
                        root,
                        CENTER_ALIASES,
                        "centers/mean",
                        sourceDescription
                );

        double[] scales =
                requireDoubleValues(
                        root,
                        SCALE_ALIASES,
                        "scales/std",
                        sourceDescription
                );

        if (centers.length != scales.length) {
            throw new IllegalArgumentException(
                    "Standardization JSON center and scale lengths differ "
                            + "in "
                            + sourceDescription
                            + ": centers="
                            + centers.length
                            + ", scales="
                            + scales.length
                            + "."
            );
        }

        int statisticGroupCount =
                centers.length;

        validateGroupCount(
                scope,
                statisticGroupCount,
                sourceDescription
        );

        List<String> jsonFeatureNames =
                optionalStringValues(
                        root,
                        FEATURE_ALIASES,
                        "feature names",
                        sourceDescription
                );

        List<String> fallbackNames =
                normalizeFeatureNames(
                        fallbackFeatureNames
                );

        List<String> featureNames =
                reconcileFeatureNames(
                        jsonFeatureNames,
                        fallbackNames,
                        scope,
                        statisticGroupCount,
                        sourceDescription
                );

        long[] counts =
                optionalCounts(
                        root,
                        statisticGroupCount,
                        sourceDescription
                );

        StandardizationStats stats =
                new StandardizationStats(
                        version,
                        method,
                        scope,
                        varianceConvention,
                        featureNames,
                        counts,
                        centers,
                        scales
                );

        config.validateStatistics(stats);

        if (!fallbackNames.isEmpty()) {
            stats.validateFeatureCompatibility(
                    fallbackNames
            );
        }

        return stats;
    }

    private static int parseVersion(
            JsonObject root
    ) {
        JsonElement element =
                findUniqueAlias(
                        root,
                        VERSION_ALIASES,
                        "format version"
                );

        if (element == null || element.isJsonNull()) {
            return StandardizationStats.CURRENT_FORMAT_VERSION;
        }

        if (!element.isJsonPrimitive()
                || !element.getAsJsonPrimitive().isNumber()) {

            throw new IllegalArgumentException(
                    "Standardization JSON version must be an integer."
            );
        }

        return element.getAsInt();
    }

    private static StandardizationMethod parseMethod(
            JsonObject root,
            StandardizationConfig config
    ) {
        JsonElement element =
                findUniqueAlias(
                        root,
                        METHOD_ALIASES,
                        "standardization method"
                );

        if (element == null || element.isJsonNull()) {
            return config.getMethod();
        }

        StandardizationMethod parsed =
                StandardizationMethod.fromString(
                        element.getAsString()
                );

        if (parsed != config.getMethod()) {
            throw new IllegalArgumentException(
                    "Standardization JSON method "
                            + parsed
                            + " conflicts with configured method "
                            + config.getMethod()
                            + "."
            );
        }

        return parsed;
    }

    private static StandardizationScope parseScope(
            JsonObject root,
            StandardizationConfig config
    ) {
        JsonElement element =
                findUniqueAlias(
                        root,
                        SCOPE_ALIASES,
                        "standardization scope"
                );

        if (element == null || element.isJsonNull()) {
            return config.getScope();
        }

        StandardizationScope parsed =
                StandardizationScope.fromString(
                        element.getAsString()
                );

        if (parsed != config.getScope()) {
            throw new IllegalArgumentException(
                    "Standardization JSON scope "
                            + parsed
                            + " conflicts with configured scope "
                            + config.getScope()
                            + "."
            );
        }

        return parsed;
    }

    private static VarianceConvention parseVarianceConvention(
            JsonObject root,
            StandardizationConfig config
    ) {
        JsonElement element =
                findUniqueAlias(
                        root,
                        VARIANCE_ALIASES,
                        "variance convention"
                );

        if (element == null || element.isJsonNull()) {
            return config.getVarianceConvention();
        }

        String raw =
                element.getAsString();

        VarianceConvention parsed;

        if ("0".equals(raw.trim())) {
            parsed =
                    VarianceConvention.POPULATION;
        } else if ("1".equals(raw.trim())) {
            parsed =
                    VarianceConvention.SAMPLE;
        } else {
            parsed =
                    VarianceConvention.fromString(raw);
        }

        if (parsed != config.getVarianceConvention()) {
            throw new IllegalArgumentException(
                    "Standardization JSON variance convention "
                            + parsed
                            + " conflicts with configured convention "
                            + config.getVarianceConvention()
                            + "."
            );
        }

        return parsed;
    }

    private static double[] requireDoubleValues(
            JsonObject root,
            String[] aliases,
            String description,
            String sourceDescription
    ) {
        JsonElement element =
                findUniqueAlias(
                        root,
                        aliases,
                        description
                );

        if (element == null || element.isJsonNull()) {
            throw new IllegalArgumentException(
                    "Standardization JSON "
                            + sourceDescription
                            + " does not contain "
                            + description
                            + ". Accepted names are "
                            + Arrays.toString(aliases)
                            + "."
            );
        }

        return parseDoubleArrayOrScalar(
                element,
                description
        );
    }

    private static long[] optionalCounts(
            JsonObject root,
            int groupCount,
            String sourceDescription
    ) {
        JsonElement element =
                findUniqueAlias(
                        root,
                        COUNT_ALIASES,
                        "observation counts"
                );

        if (element == null || element.isJsonNull()) {
            long[] unknown =
                    new long[groupCount];

            Arrays.fill(
                    unknown,
                    StandardizationStats.UNKNOWN_COUNT
            );

            return unknown;
        }

        long[] parsed =
                parseLongArrayOrScalar(
                        element,
                        "observation counts"
                );

        if (parsed.length == 1 && groupCount > 1) {
            long[] repeated =
                    new long[groupCount];

            Arrays.fill(
                    repeated,
                    parsed[0]
            );

            return repeated;
        }

        if (parsed.length != groupCount) {
            throw new IllegalArgumentException(
                    "Standardization JSON "
                            + sourceDescription
                            + " contains "
                            + parsed.length
                            + " count value(s), but "
                            + groupCount
                            + " statistic group(s) were found."
            );
        }

        return parsed;
    }

    private static List<String> optionalStringValues(
            JsonObject root,
            String[] aliases,
            String description,
            String sourceDescription
    ) {
        JsonElement element =
                findUniqueAlias(
                        root,
                        aliases,
                        description
                );

        if (element == null || element.isJsonNull()) {
            return Collections.emptyList();
        }

        if (!element.isJsonArray()) {
            throw new IllegalArgumentException(
                    description
                            + " must be represented as a JSON array in "
                            + sourceDescription
                            + "."
            );
        }

        List<String> values =
                new ArrayList<>();

        for (JsonElement item : element.getAsJsonArray()) {
            if (item == null || item.isJsonNull()) {
                throw new IllegalArgumentException(
                        description
                                + " cannot contain null values."
                );
            }

            String value =
                    item.getAsString().trim();

            if (value.isEmpty()) {
                throw new IllegalArgumentException(
                        description
                                + " cannot contain blank values."
                );
            }

            values.add(value);
        }

        return validateUniqueFeatureNames(
                values,
                description
        );
    }

    private static List<String> reconcileFeatureNames(
            List<String> jsonNames,
            List<String> fallbackNames,
            StandardizationScope scope,
            int groupCount,
            String sourceDescription
    ) {
        if (!jsonNames.isEmpty()
                && !fallbackNames.isEmpty()
                && !jsonNames.equals(fallbackNames)) {

            throw new IllegalArgumentException(
                    "Feature names in standardization JSON "
                            + sourceDescription
                            + " do not match the current reader's feature "
                            + "names or order. JSON="
                            + jsonNames
                            + ", reader="
                            + fallbackNames
                            + "."
            );
        }

        List<String> selected =
                !jsonNames.isEmpty()
                        ? jsonNames
                        : fallbackNames;

        if (scope == StandardizationScope.PER_DIMENSION
                && !selected.isEmpty()
                && selected.size() != groupCount) {

            throw new IllegalArgumentException(
                    "PER_DIMENSION statistics contain "
                            + groupCount
                            + " groups, but "
                            + selected.size()
                            + " feature names were supplied."
            );
        }

        return selected;
    }

    private static List<String> normalizeFeatureNames(
            List<String> names
    ) {
        if (names == null || names.isEmpty()) {
            return Collections.emptyList();
        }

        List<String> normalized =
                new ArrayList<>();

        for (String name : names) {
            if (name == null || name.isBlank()) {
                throw new IllegalArgumentException(
                        "Fallback feature names cannot contain null or "
                                + "blank values."
                );
            }

            normalized.add(
                    name.trim()
            );
        }

        return validateUniqueFeatureNames(
                normalized,
                "fallback feature names"
        );
    }

    private static List<String> validateUniqueFeatureNames(
            List<String> names,
            String description
    ) {
        Set<String> unique =
                new LinkedHashSet<>(names);

        if (unique.size() != names.size()) {
            throw new IllegalArgumentException(
                    description
                            + " contain duplicate values: "
                            + names
                            + "."
            );
        }

        return Collections.unmodifiableList(
                new ArrayList<>(names)
        );
    }

    private static void validateGroupCount(
            StandardizationScope scope,
            int groupCount,
            String sourceDescription
    ) {
        if (groupCount <= 0) {
            throw new IllegalArgumentException(
                    "Standardization JSON "
                            + sourceDescription
                            + " contains no statistic groups."
            );
        }

        if (scope == StandardizationScope.GLOBAL
                && groupCount != 1) {

            throw new IllegalArgumentException(
                    "GLOBAL standardization requires exactly one center "
                            + "and one scale, but "
                            + groupCount
                            + " groups were supplied."
            );
        }
    }

    private static double[] parseDoubleArrayOrScalar(
            JsonElement element,
            String description
    ) {
        if (element.isJsonArray()) {
            JsonArray array =
                    element.getAsJsonArray();

            double[] values =
                    new double[array.size()];

            for (int index = 0;
                 index < array.size();
                 index++) {

                values[index] =
                        parseFiniteDouble(
                                array.get(index),
                                description,
                                index
                        );
            }

            return values;
        }

        return new double[] {
                parseFiniteDouble(
                        element,
                        description,
                        0
                )
        };
    }

    private static long[] parseLongArrayOrScalar(
            JsonElement element,
            String description
    ) {
        if (element.isJsonArray()) {
            JsonArray array =
                    element.getAsJsonArray();

            long[] values =
                    new long[array.size()];

            for (int index = 0;
                 index < array.size();
                 index++) {

                values[index] =
                        parseCount(
                                array.get(index),
                                description,
                                index
                        );
            }

            return values;
        }

        return new long[] {
                parseCount(
                        element,
                        description,
                        0
                )
        };
    }

    private static double parseFiniteDouble(
            JsonElement element,
            String description,
            int index
    ) {
        if (element == null
                || element.isJsonNull()
                || !element.isJsonPrimitive()) {

            throw new IllegalArgumentException(
                    description
                            + " value at index "
                            + index
                            + " must be numeric."
            );
        }

        double value =
                element.getAsDouble();

        if (!Double.isFinite(value)) {
            throw new IllegalArgumentException(
                    description
                            + " value at index "
                            + index
                            + " must be finite, but received "
                            + value
                            + "."
            );
        }

        return value;
    }

    private static long parseCount(
            JsonElement element,
            String description,
            int index
    ) {
        if (element == null
                || element.isJsonNull()
                || !element.isJsonPrimitive()) {

            throw new IllegalArgumentException(
                    description
                            + " value at index "
                            + index
                            + " must be an integer."
            );
        }

        long value =
                element.getAsLong();

        if (value <= 0L) {
            throw new IllegalArgumentException(
                    description
                            + " value at index "
                            + index
                            + " must be positive, but received "
                            + value
                            + "."
            );
        }

        return value;
    }

    /**
     * Finds a logical field using case-insensitive alias matching.
     *
     * Multiple matching aliases are accepted only when their JSON values
     * are identical.
     */
    private static JsonElement findUniqueAlias(
            JsonObject root,
            String[] aliases,
            String description
    ) {
        JsonElement selected =
                null;

        String selectedName =
                null;

        for (String propertyName : root.keySet()) {
            if (!matchesAlias(
                    propertyName,
                    aliases
            )) {
                continue;
            }

            JsonElement candidate =
                    root.get(propertyName);

            if (selected == null) {
                selected =
                        candidate;

                selectedName =
                        propertyName;

                continue;
            }

            if (!Objects.equals(
                    selected,
                    candidate
            )) {
                throw new IllegalArgumentException(
                        "Standardization JSON contains conflicting fields "
                                + "for "
                                + description
                                + ": '"
                                + selectedName
                                + "' and '"
                                + propertyName
                                + "'."
                );
            }
        }

        return selected;
    }

    private static boolean matchesAlias(
            String propertyName,
            String[] aliases
    ) {
        String normalizedProperty =
                normalizeFieldName(
                        propertyName
                );

        for (String alias : aliases) {
            if (normalizedProperty.equals(
                    normalizeFieldName(alias)
            )) {
                return true;
            }
        }

        return false;
    }

    private static String normalizeFieldName(
            String value
    ) {
        return value.trim()
                .toLowerCase(Locale.ROOT)
                .replace("_", "")
                .replace("-", "")
                .replace(" ", "");
    }

    private static JsonArray stringArray(
            List<String> values
    ) {
        JsonArray array =
                new JsonArray();

        for (String value : values) {
            array.add(value);
        }

        return array;
    }

    private static JsonArray longArray(
            long[] values
    ) {
        JsonArray array =
                new JsonArray();

        for (long value : values) {
            array.add(value);
        }

        return array;
    }

    private static JsonArray doubleArray(
            double[] values
    ) {
        JsonArray array =
                new JsonArray();

        for (double value : values) {
            array.add(value);
        }

        return array;
    }
}