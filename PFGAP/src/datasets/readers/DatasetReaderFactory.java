package datasets.readers;

import datasets.readers.lazy.*;

import java.util.Objects;

/**
 * Factory for constructing DatasetReader implementations from ReaderOptions.
 *
 * ExperimentRunner and other high-level execution code should use this
 * factory instead of directly instantiating concrete reader classes.
 *
 * This keeps reader-selection logic centralized and makes it easier to add
 * new readers without modifying downstream experiment code.
 */
public final class DatasetReaderFactory {

    private DatasetReaderFactory() {
        // Utility class.
    }

    /**
     * Creates a DatasetReader based on the supplied ReaderOptions.
     *
     * @param options reader configuration
     * @return concrete DatasetReader implementation
     */
    public static DatasetReader create(ReaderOptions options) {

        Objects.requireNonNull(options, "ReaderOptions cannot be null.");

        ReaderType readerType = options.getReaderType();

        if (readerType == null) {
            throw new IllegalArgumentException("ReaderType cannot be null.");
        }

        switch (readerType) {

            case DELIMITED:
                return createDelimitedReader(options);

            case TS:
                return createTSReader(options);

            case LONG_FORMAT_DELIMITED:
                return createLongFormatDelimitedReader(options);

            case LONG_FORMAT_PARQUET:
                return createLongFormatParquetReader(options);

            case DIRECTORY:
                throw new UnsupportedOperationException(
                        "DIRECTORY reader has not been implemented yet."
                );

            case HDF5:
                return createHDF5Reader(options);

            case NESTED_PARQUET:
                throw new UnsupportedOperationException(
                        "NESTED_PARQUET reader has not been implemented yet."
                );

            case LAZY_PER_FILE_PARQUET:
                return new LazyPerFileParquetReader(options);

            case PER_FILE_PARQUET:
                return new PerFileParquetReader(options);

            case PER_FILE_DELIMITED:
                return new PerFileDelimitedReader(
                        options
                );

            case LAZY_PER_FILE_DELIMITED:
                return new LazyPerFileDelimitedReader(
                        options
                );

            case LAZY_PER_FILE_NUMERIC_DELIMITED:
                return new LazyPerFileNumericDelimitedReader(options);

            case PER_FILE_NUMERIC_DELIMITED:
                return new NumericPerFileDelimitedReader(options);

            case NUMERIC_LONG_FORMAT:
                return new NumericLongFormatReader(options);

            case PER_FILE_NUMERIC_PARQUET:
                return new NumericPerFileParquetReader(options);

            case LAZY_PER_FILE_NUMERIC_PARQUET:
                return new LazyPerFileNumericParquetReader(options);

            case PARQUET_COLUMN_FILE_READER:
                return new ParquetColumnSeriesReader(options);

            case PER_FILE_CUSTOM:
                return new CustomPerFileReader(options);

            case LAZY_PER_FILE_CUSTOM:
                return new LazyCustomPerFileReader(options);

            default:
                throw new IllegalArgumentException(
                        "Unsupported reader type: " + readerType
                );
        }
    }

    private static DatasetReader createDelimitedReader(ReaderOptions options) {

        requireNonNullOrEmpty(
                options.getDataPath(),
                "Delimited reader requires dataPath."
        );

        requireNonNullOrEmpty(
                options.getEntrySeparator(),
                "Delimited reader requires entrySeparator."
        );

        return new DelimitedFileReader(
                options.getDataPath(),
                options.getLabelPath(),
                options.getEntrySeparator(),
                options.getArraySeparator(),
                options.hasHeader(),
                options.is2D(),
                options.isNumeric(),
                options.hasMissingValues(),
                options.targetColumnIsFirst(),
                options.isTest(),
                options.isRegression()
        );
    }

    private static DatasetReader createTSReader(ReaderOptions options) {

        requireNonNullOrEmpty(
                options.getDataPath(),
                "TS reader requires dataPath."
        );

        return new TSFileReader(
                options.getDataPath(),
                options.getLabelPath(),
                options.isNumeric(),
                options.hasMissingValues(),
                options.isRegression()
        );
    }

    private static DatasetReader createLongFormatDelimitedReader(
            ReaderOptions options
    ) {

        requireNonNullOrEmpty(
                options.getDataPath(),
                "Long-format delimited reader requires dataPath."
        );

        requireNonNullOrEmpty(
                options.getEntrySeparator(),
                "Long-format delimited reader requires entrySeparator."
        );

        if (options.getFeatureColumns().isEmpty()) {
            throw new IllegalArgumentException(
                    "Long-format delimited reader requires at least one feature column."
            );
        }

        return new LongFormatReader(options);
    }

    private static DatasetReader createLongFormatParquetReader(
            ReaderOptions options
    ) {

        // if no idColumn is supplied, we assume that each row is an independent data instance

        requireNonNullOrEmpty(
                options.getDataPath(),
                "Long-format Parquet reader requires dataPath."
        );

        if (options.getFeatureColumns().isEmpty()) {
            throw new IllegalArgumentException(
                    "Long-format Parquet reader requires at least one feature column."
            );
        }

        return new LongFormatParquetReader(options);
    }

    private static DatasetReader createHDF5Reader(
            ReaderOptions options
    ) {

        requireNonNullOrEmpty(
                options.getDataPath(),
                "HDF5 reader requires dataPath."
        );

        requireNonNullOrEmpty(
                options.getHdf5DatasetPath(),
                "HDF5 reader requires hdf5DatasetPath."
        );

        return new HDF5Reader(options);
    }

    private static void requireNonNullOrEmpty(
            String value,
            String message
    ) {

        if (value == null || value.trim().isEmpty()) {
            throw new IllegalArgumentException(message);
        }
    }
}