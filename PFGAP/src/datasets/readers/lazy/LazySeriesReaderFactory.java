package datasets.readers.lazy;

import datasets.readers.*;
import datasets.readers.api.CustomReaderContext;
import datasets.readers.interop.JavaSeriesReader;

import java.io.IOException;
import java.nio.file.Path;

public final class LazySeriesReaderFactory {

    private LazySeriesReaderFactory() {
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
                            spec.getStandardizationStats()
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

            case LAZY_PER_FILE_CUSTOM -> {
                String descriptor =
                        spec.getCustomReaderDescriptor();

                if (descriptor == null || descriptor.isBlank()) {
                    throw new IllegalArgumentException(
                            "LAZY_PER_FILE_CUSTOM requires a custom reader "
                                    + "descriptor."
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
                                spec.getCustomReaderParameters()
                        );

                try {
                    yield new JavaSeriesReader(
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
            }

            default ->
                    throw new UnsupportedOperationException(
                            "Reader type "
                                    + readerType
                                    + " cannot currently be restored "
                                    + "as a LazySeriesReader."
                    );
        };
    }
}