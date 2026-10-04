package datasets.readers.lazy;

import preprocessing.standardization.StandardizationStats;
import preprocessing.standardization.Standardizer;

import java.util.Objects;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.locks.ReentrantReadWriteLock;

/**
 * Decorates a lazy series reader with PFGAP standardization.
 *
 * <p>The delegate materializes one raw observation. This decorator applies
 * prepared training statistics exactly once, in place, before returning that
 * observation. It does not fit, load, save, or mutate the statistics.</p>
 *
 * <p>Built-in standardization supports the standard primitive numeric
 * observation representations:</p>
 *
 * <pre>
 * float[time]
 * double[time]
 * float[dimension][time]
 * double[dimension][time]
 * </pre>
 *
 * <p>Numeric missing values remain primitive NaN and are handled by
 * {@link Standardizer} according to its configured implementation. Reader
 * families may impose a narrower representation contract. In particular,
 * per-file time-series readers preserve the dimension axis and return a
 * two-dimensional matrix.</p>
 *
 * <h2>Vectorization</h2>
 *
 * <p>This class deliberately contains no scalar or Vector API loop. It routes
 * the realized primitive matrix directly to
 * {@link Standardizer#transformInstanceInPlace(Object, StandardizationStats)},
 * which is the single optimization point for scalar, parallel, and Vector API
 * implementations. The decorator therefore adds no representation conversion
 * and cannot bypass the optimized standardization path.</p>
 *
 * <h2>Resource ownership and concurrency</h2>
 *
 * <p>This decorator owns its delegate. A lifecycle read lock protects an
 * entire materialize-and-standardize operation. Closure takes the write lock,
 * waits for active reads to finish, and then closes a closeable delegate.
 * Concurrent reads remain permitted when the delegate permits them.</p>
 */
public final class StandardizingLazySeriesReader
        implements LazySeriesReader, AutoCloseable {
    private final LazySeriesReader delegate;
    private final StandardizationStats standardizationStats;
    private final AtomicBoolean closed = new AtomicBoolean(false);
    private final ReentrantReadWriteLock lifecycleLock =
            new ReentrantReadWriteLock();

    public StandardizingLazySeriesReader(
            LazySeriesReader delegate,
            StandardizationStats standardizationStats
    ) {
        this.delegate = Objects.requireNonNull(
                delegate,
                "StandardizingLazySeriesReader requires a delegate.");
        this.standardizationStats = Objects.requireNonNull(
                standardizationStats,
                "StandardizingLazySeriesReader requires standardization "
                        + "statistics.");
    }

    /**
     * Materializes and standardizes one observation while preventing
     * concurrent delegate closure.
     */
    @Override
    public Object read(LazySeriesRef reference) {
        Objects.requireNonNull(
                reference,
                "StandardizingLazySeriesReader cannot read a null reference.");

        ReentrantReadWriteLock.ReadLock readLock = lifecycleLock.readLock();
        readLock.lock();
        try {
            requireOpen();
            Object series = delegate.read(reference);
            if (series == null) {
                throw new IllegalStateException(
                        "Lazy-series delegate returned null for reader key '"
                                + reference.getReaderKey()
                                + "', instance index "
                                + reference.getIndex() + ".");
            }
            validateRepresentation(series, reference);
            Standardizer.transformInstanceInPlace(
                    series, standardizationStats);
            return series;
        } finally {
            readLock.unlock();
        }
    }

    private static void validateRepresentation(
            Object series,
            LazySeriesRef reference
    ) {
        if (!(series instanceof float[])
                && !(series instanceof double[])
                && !(series instanceof float[][])
                && !(series instanceof double[][])) {
            throw new IllegalStateException(
                    "Built-in lazy standardization requires float[], "
                            + "double[], float[][], or double[][], but "
                            + "reader key '"
                            + reference.getReaderKey()
                            + "' returned "
                            + series.getClass().getTypeName()
                            + " for instance index "
                            + reference.getIndex() + ".");
        }
    }

    public LazySeriesReader getDelegate() {
        ReentrantReadWriteLock.ReadLock readLock = lifecycleLock.readLock();
        readLock.lock();
        try {
            requireOpen();
            return delegate;
        } finally {
            readLock.unlock();
        }
    }

    public StandardizationStats getStandardizationStats() {
        return standardizationStats;
    }

    public boolean isClosed() {
        return closed.get();
    }

    /**
     * Waits for active materializations and then closes the owned delegate.
     */
    @Override
    public void close() throws Exception {
        ReentrantReadWriteLock.WriteLock writeLock = lifecycleLock.writeLock();
        writeLock.lock();
        try {
            if (!closed.compareAndSet(false, true)) {
                return;
            }
            if (delegate instanceof AutoCloseable closeableDelegate) {
                closeableDelegate.close();
            }
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
                    "Failed to close standardizing lazy-series reader.",
                    failure);
        }
    }

    private void requireOpen() {
        if (closed.get()) {
            throw new IllegalStateException(
                    "StandardizingLazySeriesReader has already been closed.");
        }
    }

    @Override
    public String toString() {
        return "StandardizingLazySeriesReader{"
                + "delegate=" + delegate
                + ", method=" + standardizationStats.getMethod()
                + ", scope=" + standardizationStats.getScope()
                + ", closed=" + closed.get()
                + '}';
    }
}
