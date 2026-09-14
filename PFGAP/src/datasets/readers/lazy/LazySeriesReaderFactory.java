package datasets.readers.lazy;

import datasets.readers.*;
import datasets.readers.api.CustomReaderContext;
import datasets.readers.interop.JavaSeriesReader;
import preprocessing.standardization.StandardizationStats;

import java.io.IOException;
import java.nio.file.Path;

/**
 * Reconstructs runtime lazy-series readers from serializable reader
 * specifications.
 */
public final class LazySeriesReaderFactory {

    private LazySeriesReaderFactory() {
        // Utility class.
    }

    public static LazySeriesReader create(
            LazySeriesReaderSpec spec
    ) {
        if (spec == null) {
            throw new IllegalArgumentException(
                    "LazySeriesReaderSpec cannot be null."
            );
        }

        ReaderType readerType =
                spec.getReaderType();

        return switch (readerType) {
            case LAZY_PER_FILE_PARQUET ->
                    new PerFileParquetSeriesReader(
                            spec.getTimeColumn(),
                            spec.getFeatureColumns(),
                            spec.isNumeric(),
                            spec.hasMissingValues(),
                            spec.getStandardizationStats()
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
                            spec.getInitialTimeCapacity()
                    );

            case LAZY_PER_FILE_NUMERIC_DELIMITED ->
                    new NumericPerFileDelimitedSeriesReader(
                            spec.getEntrySeparator(),
                            spec.hasHeader(),
                            spec.getTimeColumn(),
                            spec.getFeatureColumns(),
                            spec.getStandardizationStats(),
                            spec.getInitialTimeCapacity()
                    );

            case LAZY_PER_FILE_NUMERIC_PARQUET ->
                    new NumericPerFileParquetSeriesReader(
                            spec.getTimeColumn(),
                            spec.getFeatureColumns(),
                            spec.hasMissingValues(),
                            spec.getStandardizationStats(),
                            spec.getInitialTimeCapacity(),
                            NumericPerFileParquetSeriesReader
                                    .TimeOrderPolicy
                                    .FILE_ORDER
                    );

            case LAZY_PER_FILE_CUSTOM ->
                    createCustomReader(spec);

            default ->
                    throw new UnsupportedOperationException(
                            "Reader type "
                                    + readerType
                                    + " cannot currently be restored "
                                    + "as a LazySeriesReader."
                    );
        };
    }

    /**
     * Reconstructs a custom plugin reader and conditionally decorates it with
     * PFGAP standardization.
     *
     * <p>The custom plugin always materializes raw instances. When the saved
     * specification contains prepared statistics, the returned decorator
     * transforms each conventional numeric array exactly once after the
     * plugin returns it.</p>
     */
    private static LazySeriesReader createCustomReader(
            LazySeriesReaderSpec spec
    ) {
        String descriptor =
                spec.getCustomReaderDescriptor();

        if (descriptor == null || descriptor.isBlank()) {
            throw new IllegalArgumentException(
                    "LAZY_PER_FILE_CUSTOM requires a custom reader "
                            + "descriptor."
            );
        }

        StandardizationStats standardizationStats =
                spec.getStandardizationStats();

        if (standardizationStats != null && !spec.isNumeric()) {
            throw new IllegalArgumentException(
                    "Custom lazy standardization requires isNumeric=true."
            );
        }

        Path dataPath =
                spec.getCustomReaderDataPath() == null
                        ? null
                        : Path.of(
                        spec.getCustomReaderDataPath()
                );

        CustomReaderContext context =
                new CustomReaderContext(
                        dataPath,
                        spec.isCustomReaderTest(),
                        spec.isCustomReaderRegression(),
                        spec.isNumeric(),
                        spec.hasMissingValues(),
                        spec.getFeatureColumns(),
                        spec.getCustomReaderParameters()
                );

        JavaSeriesReader customReader;

        try {
            customReader =
                    new JavaSeriesReader(
                            descriptor,
                            context,
                            spec.isCustomReaderThreadSafe()
                    );
        } catch (IOException | ReflectiveOperationException e) {
            throw new IllegalStateException(
                    "Could not reconstruct custom lazy series reader from "
                            + "descriptor: "
                            + descriptor,
                    e
            );
        }

        if (standardizationStats == null) {
            return customReader;
        }

        try {
            return new StandardizingLazySeriesReader(
                    customReader,
                    standardizationStats
            );
        } catch (RuntimeException | Error constructionFailure) {
            /*
             * Ownership transfers to the decorator only after successful
             * construction. If decoration fails, close the plugin adapter and
             * preserve any cleanup failure as suppressed context.
             */
            try {
                customReader.close();
            } catch (Exception closeFailure) {
                constructionFailure.addSuppressed(closeFailure);
            }

            throw constructionFailure;
        }
    }
}