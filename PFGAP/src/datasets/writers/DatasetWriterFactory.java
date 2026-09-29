package datasets.writers;

import datasets.ListObjectDataset;
import datasets.readers.ReaderType;

import java.util.Locale;
import java.util.Objects;

/**
 * Creates the mirror dataset writer for an input {@link ReaderType}.
 *
 * <p>The factory is the single format-selection boundary between experiment
 * orchestration and concrete output encoders. {@code ExperimentOutputCoordinator}
 * should ask this factory for a writer rather than depend directly on every
 * writer implementation.</p>
 *
 * <p>Writer support is intentionally being introduced incrementally. The
 * factory currently enables only numeric row-oriented delimited output. Known
 * but not-yet-implemented reader formats fail with an explicit message instead
 * of silently falling back to a lossy generic representation.</p>
 *
 * <p>Selection uses the stable enum name rather than a compile-time switch over
 * every enum constant. This allows writer implementations to be added in a
 * staged branch without forcing this class to reference reader constants that
 * may not exist in older builds.</p>
 */
public final class DatasetWriterFactory {

    private DatasetWriterFactory() {
        // Utility class.
    }

    /**
     * Creates the mirror writer for one reader type.
     *
     * @param readerType reader that produced the source dataset
     * @return stateless writer for the corresponding external format
     * @throws UnsupportedOperationException when the format is recognized by
     *         the reader layer but its mirror writer is not implemented yet
     */
    public static DatasetWriter create(
            ReaderType readerType
    ) {
        Objects.requireNonNull(
                readerType,
                "ReaderType cannot be null when selecting a dataset writer."
        );
        return create(readerType.name());
    }

    /**
     * Creates a writer from a reader-type name.
     *
     * <p>This overload is useful for manifests, restored configurations, and
     * focused tests. Parsing is case-insensitive and treats spaces and hyphens
     * as underscores.</p>
     *
     * @param readerTypeName reader-type name
     * @return stateless mirror writer
     */
    public static DatasetWriter create(
            String readerTypeName
    ) {
        String normalized = normalizeReaderTypeName(readerTypeName);

        return switch (normalized) {
            case "NUMERIC_DELIMITED",
                    "NUMERIC_DELIMITED_FILE",
                    "NUMERICDELIMITED" ->
                    new NumericDelimitedFileWriter();

            case "DELIMITED", "DELIMITED_FILE", "GENERIC_DELIMITED" ->
                    new DelimitedFileWriter();

            case "NUMERIC_LONG_DELIMITED",
                    "NUMERIC_LONG_FORM_DELIMITED",
                    "NUMERIC_LONG_FORM", "NUMERIC_LONG_FORMAT" ->
                    new NumericLongFormatWriter();

            case "LONG_DELIMITED", "LONG_FORM_DELIMITED",
                    "GENERIC_LONG_DELIMITED", "LONG_FORM", "LONG_FORMAT" ->
                    new LongFormatWriter();

            case "NUMERIC_PER_FILE_DELIMITED_SERIES",
                    "NUMERIC_PER_FILE_DELIMITED",
                    "NUMERIC_PER_FILE_SERIES" ->
                    unavailable(
                            normalized,
                            "NumericPerFileDelimitedSeriesWriter"
                    );

            case "PER_FILE_DELIMITED_SERIES",
                    "PER_FILE_DELIMITED",
                    "GENERIC_PER_FILE_DELIMITED_SERIES" ->
                    unavailable(
                            normalized,
                            "PerFileDelimitedSeriesWriter"
                    );

            case "NPY",
                    "NUMPY",
                    "NUMPY_ARRAY" ->
                    unavailable(
                            normalized,
                            "NpyDatasetWriter"
                    );

            case "PARQUET",
                    "HARDWOOD_PARQUET" ->
                    throw new UnsupportedOperationException(
                            "Dataset output for reader type "
                                    + normalized
                                    + " is not yet available because the "
                                    + "Hardwood writer integration is pending."
                    );

            default ->
                    throw new UnsupportedOperationException(
                            "No mirror dataset writer is registered for "
                                    + "reader type "
                                    + normalized
                                    + ". Add the writer to datasets.writers "
                                    + "and register it in DatasetWriterFactory."
                    );
        };
    }

    /**
     * Creates a writer and verifies that it supports the supplied materialized
     * representation and options.
     *
     * @param readerType reader that produced the source dataset
     * @param dataset materialized dataset to encode
     * @param options output options
     * @return validated writer
     */
    public static DatasetWriter createFor(
            ReaderType readerType,
            ListObjectDataset dataset,
            DatasetWriteOptions options
    ) {
        Objects.requireNonNull(dataset, "Dataset cannot be null.");
        Objects.requireNonNull(options, "DatasetWriteOptions cannot be null.");

        DatasetWriter writer = create(readerType);
        if (!writer.supports(dataset, options)) {
            Object first = dataset.getData() == null
                    || dataset.getData().isEmpty()
                    ? null
                    : dataset.getData().get(0);
            throw new IllegalArgumentException(
                    "Writer "
                            + writer.formatName()
                            + " does not support the requested layout "
                            + options.getDataLayout()
                            + " and materialized instance type "
                            + (first == null
                            ? "unavailable"
                            : first.getClass().getTypeName())
                            + "."
            );
        }
        return writer;
    }

    /**
     * Returns whether a writer has been implemented for the reader type.
     *
     * <p>This method does not inspect a dataset representation. Use
     * {@link #createFor(ReaderType, ListObjectDataset, DatasetWriteOptions)}
     * for complete preflight validation.</p>
     */
    public static boolean isImplemented(
            ReaderType readerType
    ) {
        Objects.requireNonNull(readerType, "ReaderType cannot be null.");
        return isImplemented(readerType.name());
    }

    /** Returns whether a writer has been implemented for a reader-type name. */
    public static boolean isImplemented(
            String readerTypeName
    ) {
        String n = normalizeReaderTypeName(readerTypeName);
        return switch (n) {
            case "NUMERIC_DELIMITED", "NUMERIC_DELIMITED_FILE",
                    "NUMERICDELIMITED", "DELIMITED", "DELIMITED_FILE",
                    "GENERIC_DELIMITED", "NUMERIC_LONG_DELIMITED",
                    "NUMERIC_LONG_FORM_DELIMITED", "NUMERIC_LONG_FORM",
                    "NUMERIC_LONG_FORMAT", "LONG_DELIMITED",
                    "LONG_FORM_DELIMITED", "GENERIC_LONG_DELIMITED",
                    "LONG_FORM", "LONG_FORMAT" -> true;
            default -> false;
        };
    }

    private static DatasetWriter unavailable(
            String readerType,
            String plannedWriter
    ) {
        throw new UnsupportedOperationException(
                "Dataset output for reader type "
                        + readerType
                        + " is planned but not yet implemented. Expected "
                        + "mirror writer: datasets.writers."
                        + plannedWriter
                        + "."
        );
    }

    private static String normalizeReaderTypeName(
            String readerTypeName
    ) {
        if (readerTypeName == null || readerTypeName.isBlank()) {
            throw new IllegalArgumentException(
                    "Reader-type name cannot be null or blank when selecting "
                            + "a dataset writer."
            );
        }
        return readerTypeName.trim()
                .toUpperCase(Locale.ROOT)
                .replace('-', '_')
                .replace(' ', '_');
    }
}
