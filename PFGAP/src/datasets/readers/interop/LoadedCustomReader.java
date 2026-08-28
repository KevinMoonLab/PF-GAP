package datasets.readers.interop;

import datasets.readers.api.CustomSeriesReader;

import java.io.IOException;
import java.net.URLClassLoader;
import java.util.Objects;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * Owns a dynamically loaded custom series reader and the class loader used
 * to load it.
 *
 * <p>The class loader must remain open for the lifetime of the reader. A
 * custom-reader implementation may load helper classes, resources, service
 * providers, or other JAR contents after its constructor has returned.
 * Closing the class loader immediately after instantiation could therefore
 * break later reader operations.</p>
 *
 * <p>When this object is closed, its URLClassLoader is closed as well. The
 * reader instance itself does not need to implement AutoCloseable, but if it
 * does, its close method is invoked before the class loader is closed.</p>
 *
 * <p>This object is a runtime resource and must not be serialized. Saved lazy
 * models should store the custom-reader descriptor and configuration
 * parameters, then reconstruct a new LoadedCustomReader in the new JVM.</p>
 */
public final class LoadedCustomReader
        implements AutoCloseable {

    private final CustomSeriesReader reader;
    private final URLClassLoader classLoader;
    private final String descriptor;
    private final String implementationClassName;
    private final AtomicBoolean closed;

    /**
     * Creates a loaded-reader resource.
     *
     * <p>This constructor is package-private because instances should be
     * created by {@link JavaReaderLoader}, which is responsible for validating
     * the descriptor, loading the class, checking the required interface, and
     * constructing the plugin.</p>
     *
     * @param reader                  loaded custom reader instance
     * @param classLoader             loader that owns the plugin classes
     * @param descriptor              original plugin descriptor
     * @param implementationClassName fully qualified implementation class name
     */
    LoadedCustomReader(
            CustomSeriesReader reader,
            URLClassLoader classLoader,
            String descriptor,
            String implementationClassName
    ) {
        this.reader =
                Objects.requireNonNull(
                        reader,
                        "LoadedCustomReader requires a reader instance."
                );

        this.classLoader =
                Objects.requireNonNull(
                        classLoader,
                        "LoadedCustomReader requires a URLClassLoader."
                );

        if (descriptor == null || descriptor.isBlank()) {
            throw new IllegalArgumentException(
                    "LoadedCustomReader requires a nonempty descriptor."
            );
        }

        if (implementationClassName == null
                || implementationClassName.isBlank()) {

            throw new IllegalArgumentException(
                    "LoadedCustomReader requires a nonempty "
                            + "implementation class name."
            );
        }

        this.descriptor =
                descriptor.trim();

        this.implementationClassName =
                implementationClassName.trim();

        this.closed =
                new AtomicBoolean(
                        false
                );
    }

    /**
     * Returns the loaded custom reader.
     *
     * @return custom reader instance
     * @throws IllegalStateException if this resource has already been closed
     */
    public CustomSeriesReader getReader() {
        requireOpen();

        return reader;
    }

    /**
     * Returns the class loader that owns the custom reader.
     *
     * <p>Most callers should not need direct access to the loader. It is
     * exposed for diagnostics and future plugin-resource use.</p>
     *
     * @return plugin class loader
     * @throws IllegalStateException if this resource has already been closed
     */
    public ClassLoader getClassLoader() {
        requireOpen();

        return classLoader;
    }

    /**
     * Returns the descriptor used to load this plugin.
     *
     * @return original normalized descriptor
     */
    public String getDescriptor() {
        return descriptor;
    }

    /**
     * Returns the fully qualified implementation class name.
     *
     * @return plugin implementation class name
     */
    public String getImplementationClassName() {
        return implementationClassName;
    }

    /**
     * Returns whether this resource has been closed.
     *
     * @return true after successful or attempted closure
     */
    public boolean isClosed() {
        return closed.get();
    }

    /**
     * Closes the plugin and its class loader.
     *
     * <p>Closure is idempotent. Calling this method more than once has no
     * effect after the first call.</p>
     *
     * <p>If the custom reader implements AutoCloseable, it is closed before
     * its class loader. If both operations fail, the class-loader exception is
     * added as a suppressed exception to the reader-close exception.</p>
     *
     * @throws Exception if plugin or class-loader cleanup fails
     */
    @Override
    public void close()
            throws Exception {

        if (!closed.compareAndSet(
                false,
                true
        )) {
            return;
        }

        Exception readerFailure =
                null;

        if (reader instanceof AutoCloseable closeableReader) {
            try {
                closeableReader.close();
            } catch (Exception e) {
                readerFailure =
                        e;
            }
        }

        IOException classLoaderFailure =
                null;

        try {
            classLoader.close();
        } catch (IOException e) {
            classLoaderFailure =
                    e;
        }

        if (readerFailure != null) {
            if (classLoaderFailure != null) {
                readerFailure.addSuppressed(
                        classLoaderFailure
                );
            }

            throw readerFailure;
        }

        if (classLoaderFailure != null) {
            throw classLoaderFailure;
        }
    }

    /**
     * Closes this resource while converting checked cleanup failures into an
     * IllegalStateException.
     *
     * <p>This is useful from application-context cleanup code that cannot
     * conveniently propagate checked exceptions.</p>
     */
    public void closeUnchecked() {
        try {
            close();
        } catch (Exception e) {
            throw new IllegalStateException(
                    "Failed to close custom reader '"
                            + implementationClassName
                            + "' loaded from descriptor: "
                            + descriptor,
                    e
            );
        }
    }

    private void requireOpen() {
        if (closed.get()) {
            throw new IllegalStateException(
                    "Custom reader has already been closed: "
                            + implementationClassName
            );
        }
    }

    @Override
    public String toString() {
        return "LoadedCustomReader{"
                + "implementationClassName='"
                + implementationClassName
                + '\''
                + ", descriptor='"
                + descriptor
                + '\''
                + ", closed="
                + closed.get()
                + '}';
    }
}