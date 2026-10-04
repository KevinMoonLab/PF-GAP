package datasets.readers.lazy;

import datasets.readers.NumericPerFileDelimitedSeriesReader;
import datasets.readers.NumericPerFileParquetSeriesReader;
import datasets.readers.PerFileDelimitedSeriesReader;
import datasets.readers.PerFileParquetSeriesReader;
import datasets.readers.ReaderType;
import datasets.readers.api.CustomReaderContext;
import datasets.readers.interop.JavaSeriesReader;
import preprocessing.standardization.StandardizationStats;

import java.io.IOException;
import java.nio.file.Path;

/**
 * Reconstructs reusable runtime lazy-series readers from serializable reader
 * specifications.
 *
 * <p>The factory is invoked during reader registration or saved-model
 * restoration, not for each materialization. The returned reader therefore
 * retains immutable configuration and reusable projections/delegates while
 * allocating only per-call file readers and materialization buffers.</p>
 */
public final class LazySeriesReaderFactory {

    private LazySeriesReaderFactory() {
        // Utility class.
    }

    public static LazySeriesReader create(LazySeriesReaderSpec spec) {
        if (spec == null) {
            throw new IllegalArgumentException(
                    "LazySeriesReaderSpec cannot be null.");
        }

        ReaderType type = spec.getReaderType();
        return switch (type) {
            case LAZY_PER_FILE_PARQUET ->
                    new PerFileParquetSeriesReader(
                            spec.getTimeColumn(),
                            spec.getFeatureColumns(),
                            spec.isNumeric(),
                            spec.hasMissingValues(),
                            spec.getStandardizationStats(),
                            spec.getInitialTimeCapacity(),
                            spec.getNumericStorageType()
                    );

            case LAZY_PER_FILE_DELIMITED ->
                    new PerFileDelimitedSeriesReader(
                            spec.getEntrySeparator(),
                            spec.hasHeader(),
                            spec.getTimeColumn(),
                            spec.getFeatureColumns(),
                            spec.isNumeric(),
                            spec.hasMissingValues(),
                            spec.getStandardizationStats(),
                            spec.getInitialTimeCapacity(),
                            spec.getNumericStorageType()
                    );

            case LAZY_PER_FILE_NUMERIC_DELIMITED ->
                    new NumericPerFileDelimitedSeriesReader(
                            spec.getEntrySeparator(),
                            spec.hasHeader(),
                            spec.getTimeColumn(),
                            spec.getFeatureColumns(),
                            spec.getStandardizationStats(),
                            spec.getInitialTimeCapacity(),
                            spec.getNumericStorageType()
                    );

            case LAZY_PER_FILE_NUMERIC_PARQUET ->
                    new NumericPerFileParquetSeriesReader(
                            spec.getTimeColumn(),
                            spec.getFeatureColumns(),
                            spec.hasMissingValues(),
                            spec.getStandardizationStats(),
                            spec.getInitialTimeCapacity(),
                            spec.getParquetTimeOrderPolicy(),
                            spec.getNumericStorageType()
                    );

            case LAZY_PER_FILE_CUSTOM -> createCustomReader(spec);

            default -> throw new IllegalArgumentException(
                    "Reader type does not define a lazy per-file series "
                            + "reader: " + type);
        };
    }

    private static LazySeriesReader createCustomReader(
            LazySeriesReaderSpec spec
    ) {
        String descriptor = spec.getCustomReaderDescriptor();
        if (descriptor == null || descriptor.isBlank()) {
            throw new IllegalArgumentException(
                    "Custom lazy reader specification requires a non-empty "
                            + "custom reader descriptor.");
        }

        Path dataPath = spec.getCustomReaderDataPath() == null
                ? null : Path.of(spec.getCustomReaderDataPath());

        CustomReaderContext context = new CustomReaderContext(
                dataPath,
                spec.isCustomReaderTest(),
                spec.isCustomReaderRegression(),
                spec.isNumeric(),
                spec.hasMissingValues(),
                spec.getFeatureColumns(),
                spec.getCustomReaderParameters(),
                spec.getNumericStorageType()
        );

        JavaSeriesReader customReader;
        try {
            customReader = new JavaSeriesReader(
                    descriptor,
                    context,
                    spec.isCustomReaderThreadSafe()
            );
        } catch (IOException | ReflectiveOperationException e) {
            throw new IllegalStateException(
                    "Could not reconstruct custom lazy series reader from "
                            + "descriptor: " + descriptor,
                    e);
        }

        PerFileValidatingLazySeriesReader perFileReader;
        try {
            perFileReader = new PerFileValidatingLazySeriesReader(customReader);
        } catch (RuntimeException | Error constructionFailure) {
            closeAfterConstructionFailure(customReader, constructionFailure);
            throw constructionFailure;
        }

        StandardizationStats stats = spec.getStandardizationStats();
        if (stats == null) {
            return perFileReader;
        }

        try {
            return new StandardizingLazySeriesReader(perFileReader, stats);
        } catch (RuntimeException | Error constructionFailure) {
            closeAfterConstructionFailure(perFileReader, constructionFailure);
            throw constructionFailure;
        }
    }

    private static void closeAfterConstructionFailure(
            AutoCloseable reader,
            Throwable constructionFailure
    ) {
        try {
            reader.close();
        } catch (Throwable closeFailure) {
            constructionFailure.addSuppressed(closeFailure);
        }
    }

}
