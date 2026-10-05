package datasets.readers.interop;

import datasets.readers.api.CustomSeriesReader;

import java.io.IOException;
import java.net.URLClassLoader;
import java.util.Objects;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * Owns a dynamically loaded custom series reader and its class loader.
 *
 * <p>The class loader remains open for the lifetime of the reader because a
 * plugin may load helper classes, resources, or service providers after its
 * constructor returns. This object must therefore be retained and closed by
 * the runtime adapter rather than discarded after plugin construction.</p>
 *
 * <p>When closed, an {@link AutoCloseable} plugin is closed before its
 * {@link URLClassLoader}. Cleanup is idempotent, and a class-loader cleanup
 * failure is suppressed onto an earlier plugin cleanup failure.</p>
 *
 * <p>This is a runtime resource and must not be serialized. Saved models store
 * the normalized descriptor and serializable reader configuration, then load
 * a new plugin instance in the destination JVM.</p>
 */
public final class LoadedCustomReader implements AutoCloseable {
    private final CustomSeriesReader reader;
    private final URLClassLoader classLoader;
    private final String descriptor;
    private final String implementationClassName;
    private final AtomicBoolean closed = new AtomicBoolean(false);

    /**
     * Creates a validated loaded-reader resource.
     *
     * <p>This constructor is package-private because
     * {@link JavaReaderLoader} owns descriptor validation, class loading, type
     * checking, and reflective construction.</p>
     */
    LoadedCustomReader(
            CustomSeriesReader reader,
            URLClassLoader classLoader,
            String descriptor,
            String implementationClassName
    ) {
        this.reader = Objects.requireNonNull(
                reader, "LoadedCustomReader requires a reader instance.");
        this.classLoader = Objects.requireNonNull(
                classLoader,
                "LoadedCustomReader requires a URLClassLoader.");
        this.descriptor = requireNonblank(descriptor, "descriptor");
        this.implementationClassName = requireNonblank(
                implementationClassName, "implementation class name");
    }

    /**
     * Returns the loaded plugin instance.
     *
     * @throws IllegalStateException if ownership has already been released
     */
    public CustomSeriesReader getReader() {
        requireOpen();
        return reader;
    }

    /**
     * Returns the plugin class loader for diagnostics or plugin-resource use.
     *
     * @throws IllegalStateException if ownership has already been released
     */
    public ClassLoader getClassLoader() {
        requireOpen();
        return classLoader;
    }

    public String getDescriptor() {
        return descriptor;
    }

    public String getImplementationClassName() {
        return implementationClassName;
    }

    /**
     * Returns whether closure has started.
     *
     * <p>A true value means cleanup was attempted. It does not imply that
     * every cleanup operation succeeded.</p>
     */
    public boolean isClosed() {
        return closed.get();
    }

    /**
     * Closes the plugin and then its class loader.
     *
     * <p>Closure is idempotent. The class loader is always given a cleanup
     * attempt, including when plugin cleanup throws an exception or error.</p>
     *
     * @throws Exception if plugin or class-loader cleanup fails
     */
    @Override
    public void close() throws Exception {
        if (!closed.compareAndSet(false, true)) {
            return;
        }

        Throwable primaryFailure = null;
        if (reader instanceof AutoCloseable closeableReader) {
            try {
                closeableReader.close();
            } catch (Throwable failure) {
                primaryFailure = failure;
            }
        }

        try {
            classLoader.close();
        } catch (IOException classLoaderFailure) {
            if (primaryFailure == null) {
                primaryFailure = classLoaderFailure;
            } else {
                primaryFailure.addSuppressed(classLoaderFailure);
            }
        }

        rethrow(primaryFailure);
    }

    /**
     * Closes this resource while converting checked cleanup failures into an
     * {@link IllegalStateException}. Runtime exceptions and errors retain
     * their original type.
     */
    public void closeUnchecked() {
        try {
            close();
        } catch (RuntimeException | Error failure) {
            throw failure;
        } catch (Exception failure) {
            throw new IllegalStateException(
                    "Failed to close custom reader '"
                            + implementationClassName
                            + "' loaded from descriptor: "
                            + descriptor,
                    failure
            );
        }
    }

    private void requireOpen() {
        if (closed.get()) {
            throw new IllegalStateException(
                    "Custom reader has already been closed: "
                            + implementationClassName);
        }
    }

    private static String requireNonblank(String value, String role) {
        if (value == null || value.isBlank()) {
            throw new IllegalArgumentException(
                    "LoadedCustomReader requires a nonempty " + role + ".");
        }
        return value.trim();
    }

    private static void rethrow(Throwable failure) throws Exception {
        if (failure == null) {
            return;
        }
        if (failure instanceof Exception exception) {
            throw exception;
        }
        if (failure instanceof Error error) {
            throw error;
        }
        throw new IllegalStateException(
                "Unexpected custom-reader cleanup failure.", failure);
    }

    @Override
    public String toString() {
        return "LoadedCustomReader{"
                + "implementationClassName='" + implementationClassName + '\''
                + ", descriptor='" + descriptor + '\''
                + ", closed=" + closed.get()
                + '}';
    }
}
