package datasets.readers;

import datasets.ListObjectDataset;
import datasets.readers.api.CustomDatasetReader;
import datasets.readers.api.CustomReaderContext;
import datasets.readers.interop.JavaReaderLoader;
import datasets.readers.interop.LoadedCustomDatasetReader;

import java.io.IOException;
import java.nio.file.InvalidPathException;
import java.nio.file.Path;
import java.util.Objects;

/**
 * Eager {@link DatasetReader} adapter for an externally supplied
 * {@link CustomDatasetReader} plugin.
 *
 * <p>The adapter translates internal {@link ReaderOptions} into the stable
 * public {@link CustomReaderContext}, invokes the plugin once to materialize
 * the complete dataset, and closes the plugin and its class loader before
 * returning. Standardization and subsequent preprocessing remain part of the
 * ordinary PFGAP dataset-preparation pipeline.</p>
 */
public final class CustomDatasetReaderAdapter implements DatasetReader {

    private final ReaderOptions options;

    public CustomDatasetReaderAdapter(ReaderOptions options) {
        this.options = Objects.requireNonNull(
                options,
                "ReaderOptions cannot be null."
        ).copy();
    }

    @Override
    public ListObjectDataset read() throws IOException {
        String descriptor = requireDescriptor(
                options.getCustomReaderDescriptor()
        );
        CustomReaderContext context = createContext(options);

        LoadedCustomDatasetReader loaded;
        try {
            loaded = JavaReaderLoader.loadDatasetReader(descriptor);
        } catch (ReflectiveOperationException failure) {
            throw new IOException(
                    "Failed to load custom dataset reader from descriptor: "
                            + descriptor,
                    failure
            );
        }

        Throwable primaryFailure = null;
        try {
            ListObjectDataset dataset = loaded.getReader().read(context);
            if (dataset == null) {
                throw new IOException(
                        "Custom dataset reader returned null for descriptor: "
                                + descriptor
                );
            }
            return dataset;
        } catch (IOException | RuntimeException | Error failure) {
            primaryFailure = failure;
            throw failure;
        } finally {
            try {
                loaded.close();
            } catch (Throwable closeFailure) {
                if (primaryFailure != null) {
                    primaryFailure.addSuppressed(closeFailure);
                } else {
                    rethrowCloseFailure(
                            closeFailure,
                            descriptor
                    );
                }
            }
        }
    }

    private static CustomReaderContext createContext(
            ReaderOptions options
    ) throws IOException {
        return new CustomReaderContext(
                toPath(options.getDataPath(), "data"),
                toPath(options.getLabelPath(), "label"),
                options.isTest(),
                options.isRegression(),
                options.isNumeric(),
                options.hasMissingValues(),
                options.getFeatureColumns(),
                options.getLabelColumns(),
                options.getCustomReaderParameters(),
                options.getNumericStorageType()
        );
    }

    private static Path toPath(
            String value,
            String role
    ) throws IOException {
        if (value == null || value.isBlank()) {
            return null;
        }
        try {
            return Path.of(value.trim()).normalize();
        } catch (InvalidPathException failure) {
            throw new IOException(
                    "Invalid custom dataset " + role + " path: " + value,
                    failure
            );
        }
    }

    private static String requireDescriptor(String descriptor) {
        if (descriptor == null || descriptor.isBlank()) {
            throw new IllegalArgumentException(
                    "CUSTOM_DATASET requires a nonblank custom reader "
                            + "descriptor."
            );
        }
        return JavaReaderLoader.normalizeDescriptor(descriptor);
    }

    private static void rethrowCloseFailure(
            Throwable failure,
            String descriptor
    ) throws IOException {
        if (failure instanceof IOException ioFailure) {
            throw ioFailure;
        }
        if (failure instanceof RuntimeException runtimeFailure) {
            throw runtimeFailure;
        }
        if (failure instanceof Error error) {
            throw error;
        }
        throw new IOException(
                "Failed to close custom dataset reader loaded from descriptor: "
                        + descriptor,
                failure
        );
    }
}
