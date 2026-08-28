package datasets.readers.interop;

import datasets.readers.api.CustomReaderContext;
import datasets.readers.api.CustomSeriesReader;
import datasets.readers.lazy.LazySeriesReader;
import datasets.readers.lazy.LazySeriesRef;

import java.io.IOException;
import java.util.Objects;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * Runtime adapter that exposes a dynamically loaded
 * {@link CustomSeriesReader} through PFGAP's internal
 * {@link LazySeriesReader} interface.
 *
 * <p>The same adapter can be used by both eager and lazy custom per-file
 * readers:</p>
 *
 * <ul>
 *     <li>
 *         An eager reader invokes {@link #read(LazySeriesRef)} while building
 *         the complete dataset.
 *     </li>
 *     <li>
 *         A lazy reader registers this adapter and invokes it only when an
 *         instance is requested by a distance calculation.
 *     </li>
 * </ul>
 *
 * <p>This class owns the associated {@link LoadedCustomReader}. Closing this
 * adapter closes both the custom plugin, when it implements
 * {@link AutoCloseable}, and the URLClassLoader used to load its JAR.</p>
 *
 * <p>The adapter itself is a runtime resource and must not be serialized.
 * Saved lazy models should retain the plugin descriptor and custom-reader
 * parameters, then reconstruct a new adapter in the target JVM.</p>
 *
 * <h2>Thread safety</h2>
 *
 * <p>When {@code threadSafe=true}, calls are delegated without
 * synchronization. The plugin implementation is then responsible for safely
 * handling concurrent reads.</p>
 *
 * <p>When {@code threadSafe=false}, calls are synchronized on a private lock.
 * This permits stateful or otherwise non-thread-safe plugins to be used
 * safely, at the cost of serializing their read operations.</p>
 */
public final class JavaSeriesReader
        implements LazySeriesReader, AutoCloseable {

    private final LoadedCustomReader loadedReader;
    private final CustomReaderContext context;
    private final boolean threadSafe;
    private final Object invocationLock;
    private final AtomicBoolean closed;

    /**
     * Loads a custom reader from a descriptor and creates its PFGAP adapter.
     *
     * @param descriptor plugin descriptor in the form
     *                   {@code javareader:path/to/plugin.jar:package.Class}
     * @param context immutable custom-reader configuration
     * @param threadSafe whether the plugin supports concurrent read calls
     * @throws IOException if the plugin JAR cannot be accessed
     * @throws ReflectiveOperationException if the plugin class cannot be
     *                                      loaded or instantiated
     */
    public JavaSeriesReader(
            String descriptor,
            CustomReaderContext context,
            boolean threadSafe
    ) throws IOException, ReflectiveOperationException {
        this(
                JavaReaderLoader.load(
                        descriptor
                ),
                context,
                threadSafe
        );
    }

    /**
     * Loads a custom reader using synchronized invocation by default.
     *
     * <p>This is the safer default for third-party plugins. A known
     * thread-safe plugin can use the three-argument constructor to permit
     * concurrent reads.</p>
     *
     * @param descriptor plugin descriptor
     * @param context immutable custom-reader configuration
     * @throws IOException if the plugin JAR cannot be accessed
     * @throws ReflectiveOperationException if the plugin cannot be loaded
     */
    public JavaSeriesReader(
            String descriptor,
            CustomReaderContext context
    ) throws IOException, ReflectiveOperationException {
        this(
                descriptor,
                context,
                false
        );
    }

    /**
     * Creates an adapter around an already loaded custom reader.
     *
     * <p>Ownership of {@code loadedReader} transfers to this adapter. The
     * caller must not close it independently while this adapter remains in
     * use.</p>
     *
     * @param loadedReader loaded plugin resource
     * @param context immutable custom-reader configuration
     * @param threadSafe whether the plugin supports concurrent read calls
     */
    public JavaSeriesReader(
            LoadedCustomReader loadedReader,
            CustomReaderContext context,
            boolean threadSafe
    ) {
        this.loadedReader =
                Objects.requireNonNull(
                        loadedReader,
                        "JavaSeriesReader requires a loaded custom reader."
                );

        this.context =
                Objects.requireNonNull(
                        context,
                        "JavaSeriesReader requires a CustomReaderContext."
                );

        if (loadedReader.isClosed()) {
            throw new IllegalArgumentException(
                    "JavaSeriesReader cannot use a LoadedCustomReader that "
                            + "has already been closed."
            );
        }

        this.threadSafe =
                threadSafe;

        this.invocationLock =
                new Object();

        this.closed =
                new AtomicBoolean(
                        false
                );
    }

    /**
     * Creates a synchronized adapter around an already loaded reader.
     *
     * @param loadedReader loaded plugin resource
     * @param context immutable custom-reader configuration
     */
    public JavaSeriesReader(
            LoadedCustomReader loadedReader,
            CustomReaderContext context
    ) {
        this(
                loadedReader,
                context,
                false
        );
    }

    /**
     * Materializes one instance through the custom plugin.
     *
     * @param reference instance reference
     * @return non-null instance representation
     * @throws IllegalStateException if this adapter is closed, the plugin
     *                               fails, or the plugin returns null
     */
    @Override
    public Object read(
            LazySeriesRef reference
    ) {
        requireOpen();

        if (reference == null) {
            throw new IllegalArgumentException(
                    "JavaSeriesReader cannot read a null LazySeriesRef."
            );
        }

        if (threadSafe) {
            return invokeReader(
                    reference
            );
        }

        synchronized (invocationLock) {
            /*
             * The adapter may have been closed while this invocation was
             * waiting to acquire the lock.
             */
            requireOpen();

            return invokeReader(
                    reference
            );
        }
    }

    private Object invokeReader(
            LazySeriesRef reference
    ) {
        CustomSeriesReader reader =
                loadedReader.getReader();

        Object result;

        try {
            result =
                    reader.read(
                            reference,
                            context
                    );
        } catch (IOException e) {
            throw new IllegalStateException(
                    buildFailureMessage(
                            "failed to read the referenced instance",
                            reference
                    ),
                    e
            );
        } catch (RuntimeException e) {
            throw new IllegalStateException(
                    buildFailureMessage(
                            "threw a runtime exception",
                            reference
                    ),
                    e
            );
        } catch (Error e) {
            /*
             * Errors indicate JVM-level or linkage failures and should not be
             * converted into ordinary exceptions. Add location context and
             * preserve the original Error category.
             */
            e.addSuppressed(
                    new IllegalStateException(
                            buildFailureMessage(
                                    "failed with an Error",
                                    reference
                            )
                    )
            );

            throw e;
        }

        if (result == null) {
            throw new IllegalStateException(
                    buildFailureMessage(
                            "returned null",
                            reference
                    )
            );
        }

        return result;
    }

    private String buildFailureMessage(
            String action,
            LazySeriesRef reference
    ) {
        StringBuilder message =
                new StringBuilder();

        message.append(
                "Custom series reader "
        );

        message.append(
                loadedReader.getImplementationClassName()
        );

        message.append(
                " "
        );

        message.append(
                action
        );

        message.append(
                " for instance index "
        );

        message.append(
                //reference.getInstanceIndex()
                reference.getIndex()
        );

        if (reference.getFile() != null) {
            message.append(
                    " from file "
            );

            message.append(
                    reference.getFile()
            );
        }

        message.append(
                ". Descriptor: "
        );

        message.append(
                loadedReader.getDescriptor()
        );

        return message.toString();
    }

    /**
     * Returns the immutable context supplied to the plugin.
     *
     * @return custom-reader context
     */
    public CustomReaderContext getContext() {
        return context;
    }

    /**
     * Returns the loaded plugin implementation.
     *
     * <p>This method is primarily intended for diagnostics and carefully
     * controlled integration code. Normal callers should use
     * {@link #read(LazySeriesRef)}.</p>
     *
     * @return loaded custom reader implementation
     * @throws IllegalStateException if this adapter is closed
     */
    public CustomSeriesReader getCustomReader() {
        requireOpen();

        return loadedReader.getReader();
    }

    /**
     * Returns the plugin descriptor.
     *
     * @return normalized plugin descriptor
     */
    public String getDescriptor() {
        return loadedReader.getDescriptor();
    }

    /**
     * Returns the plugin implementation class name.
     *
     * @return fully qualified implementation class name
     */
    public String getImplementationClassName() {
        return loadedReader.getImplementationClassName();
    }

    /**
     * Returns whether unsynchronized concurrent plugin invocation is enabled.
     *
     * @return true when the plugin is declared thread-safe
     */
    public boolean isThreadSafe() {
        return threadSafe;
    }

    /**
     * Returns whether this adapter has been closed.
     *
     * @return true after closure begins
     */
    public boolean isClosed() {
        return closed.get();
    }

    /**
     * Closes the custom plugin and its owning class loader.
     *
     * <p>Closure is idempotent. For non-thread-safe plugins, closure acquires
     * the same lock used for invocation so it cannot close the class loader
     * while a synchronized read is active.</p>
     *
     * <p>For plugins declared thread-safe, the caller must ensure that no
     * concurrent read remains active before closing the adapter. This avoids
     * adding synchronization overhead to the thread-safe read path.</p>
     *
     * @throws Exception if plugin or class-loader cleanup fails
     */
    @Override
    public void close()
            throws Exception {

        if (threadSafe) {
            closeInternal();

            return;
        }

        synchronized (invocationLock) {
            closeInternal();
        }
    }

    private void closeInternal()
            throws Exception {

        if (!closed.compareAndSet(
                false,
                true
        )) {
            return;
        }

        loadedReader.close();
    }

    /**
     * Closes this adapter while converting checked cleanup failures into an
     * IllegalStateException.
     */
    public void closeUnchecked() {
        try {
            close();
        } catch (Exception e) {
            throw new IllegalStateException(
                    "Failed to close custom series reader "
                            + loadedReader.getImplementationClassName()
                            + " loaded from descriptor: "
                            + loadedReader.getDescriptor(),
                    e
            );
        }
    }

    private void requireOpen() {
        if (closed.get()) {
            throw new IllegalStateException(
                    "JavaSeriesReader has already been closed for plugin: "
                            + loadedReader.getImplementationClassName()
            );
        }

        if (loadedReader.isClosed()) {
            throw new IllegalStateException(
                    "The loaded custom reader resource has already been "
                            + "closed for plugin: "
                            + loadedReader.getImplementationClassName()
            );
        }
    }

    @Override
    public String toString() {
        return "JavaSeriesReader{"
                + "implementationClassName='"
                + loadedReader.getImplementationClassName()
                + '\''
                + ", descriptor='"
                + loadedReader.getDescriptor()
                + '\''
                + ", threadSafe="
                + threadSafe
                + ", closed="
                + closed.get()
                + '}';
    }
}