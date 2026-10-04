package datasets.readers.interop;

import datasets.readers.api.CustomReaderContext;
import datasets.readers.api.CustomSeriesReader;
import datasets.readers.lazy.LazySeriesReader;
import datasets.readers.lazy.LazySeriesRef;

import java.io.IOException;
import java.io.InterruptedIOException;
import java.util.Objects;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.locks.ReentrantReadWriteLock;

/**
 * Runtime adapter exposing a dynamically loaded {@link CustomSeriesReader}
 * through PFGAP's {@link LazySeriesReader} contract.
 *
 * <p>The adapter owns its {@link LoadedCustomReader}. Closing the adapter
 * closes an {@link AutoCloseable} plugin and then its class loader.</p>
 *
 * <h2>Concurrency</h2>
 *
 * <p>A lifecycle read lock protects every plugin invocation from concurrent
 * closure. Thread-safe plugins may still execute reads concurrently. Readers
 * declared non-thread-safe are additionally serialized on a private invocation
 * lock. Closure takes the lifecycle write lock and therefore waits for all
 * active reads, regardless of the plugin's thread-safety declaration.</p>
 */
public final class JavaSeriesReader
        implements LazySeriesReader, AutoCloseable {
    private final LoadedCustomReader loadedReader;
    private final CustomReaderContext context;
    private final boolean threadSafe;
    private final Object invocationLock = new Object();
    private final AtomicBoolean closed = new AtomicBoolean(false);
    private final ReentrantReadWriteLock lifecycleLock =
            new ReentrantReadWriteLock();

    public JavaSeriesReader(
            String descriptor,
            CustomReaderContext context,
            boolean threadSafe
    ) throws IOException, ReflectiveOperationException {
        this(JavaReaderLoader.load(descriptor), context, threadSafe);
    }

    /**
     * Loads a custom reader using synchronized plugin invocation by default.
     */
    public JavaSeriesReader(
            String descriptor,
            CustomReaderContext context
    ) throws IOException, ReflectiveOperationException {
        this(descriptor, context, false);
    }

    /**
     * Creates an adapter and assumes ownership of {@code loadedReader}.
     */
    public JavaSeriesReader(
            LoadedCustomReader loadedReader,
            CustomReaderContext context,
            boolean threadSafe
    ) {
        this.loadedReader = Objects.requireNonNull(
                loadedReader,
                "JavaSeriesReader requires a loaded custom reader.");
        this.context = Objects.requireNonNull(
                context,
                "JavaSeriesReader requires a CustomReaderContext.");
        if (loadedReader.isClosed()) {
            throw new IllegalArgumentException(
                    "JavaSeriesReader cannot use a LoadedCustomReader that "
                            + "has already been closed.");
        }
        this.threadSafe = threadSafe;
    }

    public JavaSeriesReader(
            LoadedCustomReader loadedReader,
            CustomReaderContext context
    ) {
        this(loadedReader, context, false);
    }

    /**
     * Materializes one observation through the custom plugin.
     */
    @Override
    public Object read(LazySeriesRef reference) {
        if (reference == null) {
            throw new IllegalArgumentException(
                    "JavaSeriesReader cannot read a null LazySeriesRef.");
        }

        ReentrantReadWriteLock.ReadLock readLock = lifecycleLock.readLock();
        readLock.lock();
        try {
            requireOpen();
            if (threadSafe) {
                return invokeReader(reference);
            }
            synchronized (invocationLock) {
                requireOpen();
                return invokeReader(reference);
            }
        } finally {
            readLock.unlock();
        }
    }

    private Object invokeReader(LazySeriesRef reference) {
        CustomSeriesReader reader = loadedReader.getReader();
        Object result;
        try {
            result = reader.read(reference, context);
        } catch (InterruptedIOException exception) {
            Thread.currentThread().interrupt();
            throw pluginFailure(
                    "was interrupted while reading the referenced instance",
                    reference,
                    exception);
        } catch (IOException exception) {
            throw pluginFailure(
                    "failed to read the referenced instance",
                    reference,
                    exception);
        } catch (RuntimeException exception) {
            throw pluginFailure(
                    "threw a runtime exception",
                    reference,
                    exception);
        } catch (Error error) {
            error.addSuppressed(new IllegalStateException(
                    buildFailureMessage("failed with an Error", reference)));
            throw error;
        }

        if (result == null) {
            throw new IllegalStateException(
                    buildFailureMessage("returned null", reference));
        }
        return result;
    }

    private IllegalStateException pluginFailure(
            String action,
            LazySeriesRef reference,
            Exception cause
    ) {
        return new IllegalStateException(
                buildFailureMessage(action, reference), cause);
    }

    private String buildFailureMessage(
            String action,
            LazySeriesRef reference
    ) {
        StringBuilder message = new StringBuilder()
                .append("Custom series reader ")
                .append(loadedReader.getImplementationClassName())
                .append(' ')
                .append(action)
                .append(" for reader key '")
                .append(reference.getReaderKey())
                .append("', instance index ")
                .append(reference.getIndex());

        String filePath = reference.getFilePath();
        if (filePath != null && !filePath.isBlank()) {
            message.append(" from file ").append(filePath);
        }
        return message.append(". Descriptor: ")
                .append(loadedReader.getDescriptor())
                .toString();
    }

    public CustomReaderContext getContext() {
        return context;
    }

    public CustomSeriesReader getCustomReader() {
        ReentrantReadWriteLock.ReadLock readLock = lifecycleLock.readLock();
        readLock.lock();
        try {
            requireOpen();
            return loadedReader.getReader();
        } finally {
            readLock.unlock();
        }
    }

    public String getDescriptor() {
        return loadedReader.getDescriptor();
    }

    public String getImplementationClassName() {
        return loadedReader.getImplementationClassName();
    }

    public boolean isThreadSafe() {
        return threadSafe;
    }

    public boolean isClosed() {
        return closed.get();
    }

    /**
     * Waits for all active reads, then closes the plugin and class loader.
     */
    @Override
    public void close() throws Exception {
        ReentrantReadWriteLock.WriteLock writeLock = lifecycleLock.writeLock();
        writeLock.lock();
        try {
            if (!closed.compareAndSet(false, true)) {
                return;
            }
            loadedReader.close();
        } finally {
            writeLock.unlock();
        }
    }

    public void closeUnchecked() {
        try {
            close();
        } catch (RuntimeException | Error failure) {
            throw failure;
        } catch (Exception failure) {
            throw new IllegalStateException(
                    "Failed to close custom series reader "
                            + loadedReader.getImplementationClassName()
                            + " loaded from descriptor: "
                            + loadedReader.getDescriptor(),
                    failure);
        }
    }

    private void requireOpen() {
        if (closed.get() || loadedReader.isClosed()) {
            throw new IllegalStateException(
                    "JavaSeriesReader has already been closed for plugin: "
                            + loadedReader.getImplementationClassName());
        }
    }

    @Override
    public String toString() {
        return "JavaSeriesReader{"
                + "implementationClassName='"
                + loadedReader.getImplementationClassName() + '\''
                + ", descriptor='" + loadedReader.getDescriptor() + '\''
                + ", threadSafe=" + threadSafe
                + ", closed=" + closed.get()
                + '}';
    }
}
