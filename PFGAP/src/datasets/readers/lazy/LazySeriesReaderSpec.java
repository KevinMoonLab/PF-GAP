package datasets.readers.lazy;

import datasets.readers.ReaderType;
import preprocessing.standardization.StandardizationStats;

import java.io.Serial;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * Serializable configuration needed to reconstruct a LazySeriesReader.
 *
 * <p>Runtime reader objects are not serialized. A saved model stores these
 * specifications and uses LazySeriesReaderFactory to recreate the readers
 * after model loading.</p>
 *
 * <p>The initialTimeCapacity and custom-reader fields were added after the
 * original serialized form was established. Older serialized specifications
 * do not contain those fields. Java deserialization therefore assigns zero
 * to the missing primitive fields, false to missing Boolean fields, and null
 * to missing object fields. The public getters normalize those values to
 * backward-compatible defaults.</p>
 */
public final class LazySeriesReaderSpec
        implements Serializable {

    @Serial
    private static final long serialVersionUID =
            1L;

    /**
     * Default initial allocation capacity used by compatible per-file
     * readers.
     *
     * <p>This is an allocation hint only. It does not constrain the final
     * number of time points in a materialized series.</p>
     */
    public static final int DEFAULT_INITIAL_TIME_CAPACITY =
            256;

    private final String readerKey;
    private final ReaderType readerType;
    private final String timeColumn;
    private final List<String> featureColumns;
    private final boolean numeric;
    private final boolean hasMissingValues;
    private final String entrySeparator;
    private final boolean hasHeader;
    private final StandardizationStats standardizationStats;

    /**
     * Initial allocation capacity for each materialized dimension.
     *
     * <p>Older serialized specifications receive zero. Callers should use
     * getInitialTimeCapacity(), which converts nonpositive stored values to
     * DEFAULT_INITIAL_TIME_CAPACITY.</p>
     */
    private final int initialTimeCapacity;

    /*
     * Custom-reader reconstruction fields.
     *
     * These fields describe how to reload the plugin. Runtime plugin
     * instances and URLClassLoaders are never serialized.
     */
    private final String customReaderDescriptor;
    private final Map<String, String> customReaderParameters;
    private final boolean customReaderThreadSafe;
    private final String customReaderDataPath;
    private final boolean customReaderTest;
    private final boolean customReaderRegression;

    /**
     * Complete constructor, including custom-reader reconstruction
     * information.
     *
     * <p>LazyCustomPerFileReader should use this constructor.</p>
     */
    public LazySeriesReaderSpec(
            String readerKey,
            ReaderType readerType,
            String timeColumn,
            List<String> featureColumns,
            boolean numeric,
            boolean hasMissingValues,
            String entrySeparator,
            boolean hasHeader,
            StandardizationStats standardizationStats,
            int initialTimeCapacity,
            String customReaderDescriptor,
            Map<String, String> customReaderParameters,
            boolean customReaderThreadSafe,
            String customReaderDataPath,
            boolean customReaderTest,
            boolean customReaderRegression
    ) {
        if (readerKey == null || readerKey.isBlank()) {
            throw new IllegalArgumentException(
                    "LazySeriesReaderSpec requires a non-empty readerKey."
            );
        }

        this.readerType =
                Objects.requireNonNull(
                        readerType,
                        "LazySeriesReaderSpec requires readerType."
                );

        if (initialTimeCapacity < 1) {
            throw new IllegalArgumentException(
                    "LazySeriesReaderSpec initialTimeCapacity must be "
                            + "at least 1. Received: "
                            + initialTimeCapacity
                            + "."
            );
        }

        this.readerKey =
                readerKey.trim();

        this.timeColumn =
                normalizeNullableString(
                        timeColumn
                );

        this.featureColumns =
                copyFeatureColumns(
                        featureColumns
                );

        this.numeric =
                numeric;

        this.hasMissingValues =
                hasMissingValues;

        this.entrySeparator =
                entrySeparator;

        this.hasHeader =
                hasHeader;

        this.standardizationStats =
                standardizationStats;

        this.initialTimeCapacity =
                initialTimeCapacity;

        this.customReaderDescriptor =
                normalizeNullableString(
                        customReaderDescriptor
                );

        this.customReaderParameters =
                copyParameters(
                        customReaderParameters
                );

        this.customReaderThreadSafe =
                customReaderThreadSafe;

        this.customReaderDataPath =
                normalizeNullableString(
                        customReaderDataPath
                );

        this.customReaderTest =
                customReaderTest;

        this.customReaderRegression =
                customReaderRegression;
    }

    /**
     * Capacity-aware constructor for built-in readers.
     *
     * <p>This retains the constructor introduced for the optimized numeric
     * readers. Custom-reader fields receive safe defaults.</p>
     */
    public LazySeriesReaderSpec(
            String readerKey,
            ReaderType readerType,
            String timeColumn,
            List<String> featureColumns,
            boolean numeric,
            boolean hasMissingValues,
            String entrySeparator,
            boolean hasHeader,
            StandardizationStats standardizationStats,
            int initialTimeCapacity
    ) {
        this(
                readerKey,
                readerType,
                timeColumn,
                featureColumns,
                numeric,
                hasMissingValues,
                entrySeparator,
                hasHeader,
                standardizationStats,
                initialTimeCapacity,
                null,
                Map.of(),
                false,
                null,
                false,
                false
        );
    }

    /**
     * Backward-compatible constructor retaining the original full signature.
     */
    public LazySeriesReaderSpec(
            String readerKey,
            ReaderType readerType,
            String timeColumn,
            List<String> featureColumns,
            boolean numeric,
            boolean hasMissingValues,
            String entrySeparator,
            boolean hasHeader,
            StandardizationStats standardizationStats
    ) {
        this(
                readerKey,
                readerType,
                timeColumn,
                featureColumns,
                numeric,
                hasMissingValues,
                entrySeparator,
                hasHeader,
                standardizationStats,
                DEFAULT_INITIAL_TIME_CAPACITY,
                null,
                Map.of(),
                false,
                null,
                false,
                false
        );
    }

    /**
     * Backward-compatible constructor for specifications that do not require
     * delimited-reader configuration.
     */
    public LazySeriesReaderSpec(
            String readerKey,
            ReaderType readerType,
            String timeColumn,
            List<String> featureColumns,
            boolean numeric,
            boolean hasMissingValues,
            StandardizationStats standardizationStats
    ) {
        this(
                readerKey,
                readerType,
                timeColumn,
                featureColumns,
                numeric,
                hasMissingValues,
                null,
                false,
                standardizationStats,
                DEFAULT_INITIAL_TIME_CAPACITY,
                null,
                Map.of(),
                false,
                null,
                false,
                false
        );
    }

    /**
     * Backward-compatible constructor without delimited-reader or
     * standardization configuration.
     */
    public LazySeriesReaderSpec(
            String readerKey,
            ReaderType readerType,
            String timeColumn,
            List<String> featureColumns,
            boolean numeric,
            boolean hasMissingValues
    ) {
        this(
                readerKey,
                readerType,
                timeColumn,
                featureColumns,
                numeric,
                hasMissingValues,
                null,
                false,
                null,
                DEFAULT_INITIAL_TIME_CAPACITY,
                null,
                Map.of(),
                false,
                null,
                false,
                false
        );
    }

    public String getReaderKey() {
        return readerKey;
    }

    public ReaderType getReaderType() {
        return readerType;
    }

    public String getTimeColumn() {
        return timeColumn;
    }

    public List<String> getFeatureColumns() {
        /*
         * Handles old serialized objects defensively, although the field
         * existed in the original serialized form.
         */
        if (featureColumns == null) {
            return List.of();
        }

        return Collections.unmodifiableList(
                featureColumns
        );
    }

    public boolean isNumeric() {
        return numeric;
    }

    public boolean hasMissingValues() {
        return hasMissingValues;
    }

    public String getEntrySeparator() {
        return entrySeparator;
    }

    public boolean hasHeader() {
        return hasHeader;
    }

    public StandardizationStats getStandardizationStats() {
        return standardizationStats;
    }

    public boolean hasStandardizationStats() {
        return standardizationStats != null;
    }

    /**
     * Returns the effective initial time capacity.
     *
     * <p>Older serialized specifications receive zero for the newly added
     * field. Zero is converted to the backward-compatible default.</p>
     */
    public int getInitialTimeCapacity() {
        return initialTimeCapacity > 0
                ? initialTimeCapacity
                : DEFAULT_INITIAL_TIME_CAPACITY;
    }

    public boolean hasExplicitInitialTimeCapacity() {
        return initialTimeCapacity > 0;
    }

    /**
     * Returns the descriptor needed to reload a custom reader plugin.
     *
     * @return descriptor, or null for a built-in reader
     */
    public String getCustomReaderDescriptor() {
        return customReaderDescriptor;
    }

    public boolean hasCustomReaderDescriptor() {
        return customReaderDescriptor != null
                && !customReaderDescriptor.isBlank();
    }

    /**
     * Returns the custom-reader parameter map.
     *
     * <p>Older serialized specifications receive null for this newly added
     * field. Such values are exposed as an empty map.</p>
     */
    public Map<String, String> getCustomReaderParameters() {
        if (customReaderParameters == null
                || customReaderParameters.isEmpty()) {

            return Map.of();
        }

        return Collections.unmodifiableMap(
                customReaderParameters
        );
    }

    public boolean isCustomReaderThreadSafe() {
        return customReaderThreadSafe;
    }

    /**
     * Returns the dataset-level path stored for CustomReaderContext.
     *
     * @return configured path string, or null
     */
    public String getCustomReaderDataPath() {
        return customReaderDataPath;
    }

    public boolean isCustomReaderTest() {
        return customReaderTest;
    }

    public boolean isCustomReaderRegression() {
        return customReaderRegression;
    }

    private static List<String> copyFeatureColumns(
            List<String> featureColumns
    ) {
        if (featureColumns == null || featureColumns.isEmpty()) {
            return new ArrayList<>();
        }

        return new ArrayList<>(
                featureColumns
        );
    }

    private static Map<String, String> copyParameters(
            Map<String, String> parameters
    ) {
        if (parameters == null || parameters.isEmpty()) {
            return new LinkedHashMap<>();
        }

        Map<String, String> copy =
                new LinkedHashMap<>(
                        Math.max(
                                16,
                                parameters.size() * 2
                        )
                );

        for (Map.Entry<String, String> entry
                : parameters.entrySet()) {

            String name =
                    entry.getKey();

            if (name == null || name.isBlank()) {
                throw new IllegalArgumentException(
                        "Custom-reader parameter names cannot be null "
                                + "or blank."
                );
            }

            copy.put(
                    name.trim(),
                    entry.getValue()
            );
        }

        return copy;
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

    @Override
    public String toString() {
        return "LazySeriesReaderSpec{"
                + "readerKey='"
                + readerKey
                + '\''
                + ", readerType="
                + readerType
                + ", timeColumn='"
                + timeColumn
                + '\''
                + ", featureColumns="
                + getFeatureColumns()
                + ", numeric="
                + numeric
                + ", hasMissingValues="
                + hasMissingValues
                + ", entrySeparator='"
                + printableSeparator(entrySeparator)
                + '\''
                + ", hasHeader="
                + hasHeader
                + ", hasStandardizationStats="
                + hasStandardizationStats()
                + ", initialTimeCapacity="
                + getInitialTimeCapacity()
                + ", explicitInitialTimeCapacity="
                + hasExplicitInitialTimeCapacity()
                + ", customReader="
                + hasCustomReaderDescriptor()
                + ", customReaderClass='"
                + extractCustomReaderClassName(
                customReaderDescriptor
        )
                + '\''
                + ", customReaderParameterNames="
                + getCustomReaderParameters().keySet()
                + ", customReaderThreadSafe="
                + customReaderThreadSafe
                + ", customReaderTest="
                + customReaderTest
                + ", customReaderRegression="
                + customReaderRegression
                + '}';
    }

    private static String printableSeparator(
            String separator
    ) {
        if (separator == null) {
            return null;
        }

        return switch (separator) {
            case "\t" -> "\\t";
            case "\n" -> "\\n";
            case "\r" -> "\\r";
            default -> separator;
        };
    }

    /**
     * Returns only the plugin class name for diagnostics, avoiding disclosure
     * of the complete filesystem path in toString().
     */
    private static String extractCustomReaderClassName(
            String descriptor
    ) {
        if (descriptor == null || descriptor.isBlank()) {
            return null;
        }

        int separator =
                descriptor.lastIndexOf(':');

        if (separator < 0
                || separator == descriptor.length() - 1) {

            return descriptor;
        }

        return descriptor.substring(
                separator + 1
        );
    }
}