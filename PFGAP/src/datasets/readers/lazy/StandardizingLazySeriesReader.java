package datasets.readers.lazy;

import preprocessing.standardization.StandardizationStats;
import preprocessing.standardization.Standardizer;

import java.util.Objects;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * Decorates a lazy series reader with PFGAP standardization.
 *
 * <p>The delegate is responsible for materializing one raw instance. This
 * decorator applies already prepared training statistics exactly once after
 * materialization and before returning the instance to the caller.</p>
 *
 * <p>This class does not fit, load, save, or mutate the statistics. Those
 * workflow responsibilities belong to the standardization pipeline. The
 * statistics supplied here must therefore already be final and compatible
 * with the materialized numeric representation.</p>
 *
 * <p>Built-in standardization currently supports these realized forms:</p>
 *
 * <pre>
 * double[]
 * Double[]
 * double[][]
 * Double[][]
 * </pre>
 *
 * <p>A custom reader may still return a proprietary representation when
 * standardization is disabled. When this decorator is present, an unsupported
 * representation is rejected by {@link Standardizer} rather than guessed at.</p>
 *
 * <h2>Resource ownership</h2>
 *
 * <p>This decorator owns its delegate. Closing it closes the delegate when the
 * delegate implements {@link AutoCloseable}. Closure is idempotent.</p>
 *
 * <h2>Thread safety</h2>
 *
 * <p>The decorator adds no synchronization to read operations. Its effective
 * read concurrency is therefore the same as the delegate's. Standardization
 * modifies only the newly materialized instance returned by that invocation;
 * the same instance object must not be shared concurrently by the delegate.</p>
 */
public final class StandardizingLazySeriesReader
        implements LazySeriesReader, AutoCloseable {

    private final LazySeriesReader delegate;
    private final StandardizationStats standardizationStats;
    private final AtomicBoolean closed;

    /**
     * Creates a standardizing decorator and transfers ownership of the
     * delegate to it.
     *
     * @param delegate reader that returns one raw materialized instance
     * @param standardizationStats prepared training statistics
     */
    public StandardizingLazySeriesReader(
            LazySeriesReader delegate,
            StandardizationStats standardizationStats
    ) {
        this.delegate =
                Objects.requireNonNull(
                        delegate,
                        "StandardizingLazySeriesReader requires a delegate."
                );

        this.standardizationStats =
                Objects.requireNonNull(
                        standardizationStats,
                        "StandardizingLazySeriesReader requires "
                                + "standardization statistics."
                );

        this.closed =
                new AtomicBoolean(false);
    }

    /**
     * Materializes one raw instance and standardizes that same object in
     * place.
     *
     * @param reference instance reference
     * @return the delegate's materialized instance after standardization
     */
    @Override
    public Object read(
            LazySeriesRef reference
    ) {
        requireOpen();

        Objects.requireNonNull(
                reference,
                "StandardizingLazySeriesReader cannot read a null reference."
        );

        Object series =
                delegate.read(reference);

        if (series == null) {
            throw new IllegalStateException(
                    "Lazy-series delegate returned null for instance index "
                            + reference.getIndex()
                            + "."
            );
        }

        return Standardizer.transformInstanceInPlace(
                series,
                standardizationStats
        );
    }

    /**
     * Returns the wrapped reader for controlled diagnostics.
     *
     * @return non-null delegate
     * @throws IllegalStateException after this decorator is closed
     */
    public LazySeriesReader getDelegate() {
        requireOpen();
        return delegate;
    }

    /**
     * Returns the prepared statistics used by this decorator.
     *
     * @return non-null prepared statistics
     */
    public StandardizationStats getStandardizationStats() {
        return standardizationStats;
    }

    /**
     * Returns whether closure has begun.
     *
     * @return true after the first close call
     */
    public boolean isClosed() {
        return closed.get();
    }

    /**
     * Closes the owned delegate when it is closeable.
     *
     * @throws Exception if delegate cleanup fails
     */
    @Override
    public void close() throws Exception {
        if (!closed.compareAndSet(false, true)) {
            return;
        }

        if (delegate instanceof AutoCloseable closeableDelegate) {
            closeableDelegate.close();
        }
    }

    /**
     * Closes this decorator while converting checked cleanup failures to an
     * unchecked lifecycle exception.
     */
    public void closeUnchecked() {
        try {
            close();
        } catch (Exception e) {
            throw new IllegalStateException(
                    "Failed to close standardizing lazy-series reader.",
                    e
            );
        }
    }

    private void requireOpen() {
        if (closed.get()) {
            throw new IllegalStateException(
                    "StandardizingLazySeriesReader has already been closed."
            );
        }
    }

    @Override
    public String toString() {
        return "StandardizingLazySeriesReader{"
                + "delegate="
                + delegate
                + ", method="
                + standardizationStats.getMethod()
                + ", scope="
                + standardizationStats.getScope()
                + ", closed="
                + closed.get()
                + '}';
    }
}