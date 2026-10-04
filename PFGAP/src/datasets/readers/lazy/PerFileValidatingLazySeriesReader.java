package datasets.readers.lazy;

import java.util.Objects;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.locks.ReentrantReadWriteLock;

/**
 * Enforces the standard per-file time-series representation contract around
 * an arbitrary lazy series reader.
 *
 * <p>Standard PFGAP one-dimensional arrays are valid generally, but per-file
 * readers preserve the dimension axis. This decorator therefore rejects
 * {@code float[]}, {@code double[]}, and {@code Object[]}; a univariate
 * per-file observation must remain {@code [1][time]}.</p>
 *
 * <p>Recognized standard per-file representations are {@code float[][]},
 * {@code double[][]}, and {@code Object[][]}. They must be nonempty,
 * rectangular, and contain no null dimension arrays. Proprietary non-array
 * representations remain available for custom distances.</p>
 */
public final class PerFileValidatingLazySeriesReader
        implements LazySeriesReader, AutoCloseable {
    private final LazySeriesReader delegate;
    private final AtomicBoolean closed = new AtomicBoolean(false);
    private final ReentrantReadWriteLock lifecycleLock =
            new ReentrantReadWriteLock();

    public PerFileValidatingLazySeriesReader(LazySeriesReader delegate) {
        this.delegate = Objects.requireNonNull(
                delegate,
                "PerFileValidatingLazySeriesReader requires a delegate.");
    }

    @Override
    public Object read(LazySeriesRef reference) {
        Objects.requireNonNull(reference, "reference cannot be null.");
        ReentrantReadWriteLock.ReadLock readLock = lifecycleLock.readLock();
        readLock.lock();
        try {
            requireOpen();
            Object result = delegate.read(reference);
            validate(result, reference);
            return result;
        } finally {
            readLock.unlock();
        }
    }

    private static void validate(Object result, LazySeriesRef reference) {
        if (result == null) {
            throw failure("returned null", reference);
        }
        // Check the more-specific 2D array types first. Every Java 2D
        // array is also an Object[], so testing Object[] first would reject
        // valid per-file matrices before they reach matrix validation.
        if (result instanceof float[][] values) {
            validateRows(values, reference);
            return;
        }
        if (result instanceof double[][] values) {
            validateRows(values, reference);
            return;
        }
        if (result instanceof Object[][] values) {
            validateRows(values, reference);
            return;
        }
        if (result instanceof float[]
                || result instanceof double[]
                || result instanceof Object[]) {
            throw failure(
                    "returned a one-dimensional standard representation; "
                            + "per-file readers require a two-dimensional "
                            + "dimension-major array such as [1][time]",
                    reference);
        }
    }

    private static void validateRows(Object[] rows, LazySeriesRef reference) {
        if (rows.length == 0) {
            throw failure("returned zero dimensions", reference);
        }
        Object first = rows[0];
        if (first == null || java.lang.reflect.Array.getLength(first) == 0) {
            throw failure("returned a null or empty first dimension", reference);
        }
        int expectedLength = java.lang.reflect.Array.getLength(first);
        for (int dimension = 1; dimension < rows.length; dimension++) {
            Object row = rows[dimension];
            if (row == null) {
                throw failure("returned null dimension " + dimension,
                        reference);
            }
            int actualLength = java.lang.reflect.Array.getLength(row);
            if (actualLength != expectedLength) {
                throw failure(
                        "returned a nonrectangular matrix: dimension "
                                + dimension + " has length " + actualLength
                                + " instead of " + expectedLength,
                        reference);
            }
        }
    }

    private static IllegalStateException failure(
            String problem,
            LazySeriesRef reference
    ) {
        return new IllegalStateException(
                "Lazy per-file reader " + problem + " for reader key '"
                        + reference.getReaderKey() + "', instance index "
                        + reference.getIndex() + ".");
    }

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

    private void requireOpen() {
        if (closed.get()) {
            throw new IllegalStateException(
                    "PerFileValidatingLazySeriesReader is closed.");
        }
    }
}
