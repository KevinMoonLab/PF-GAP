package core.parallel;

import java.io.Serial;
import java.util.Objects;
import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.ForkJoinTask;
import java.util.concurrent.RecursiveAction;
import java.util.concurrent.RecursiveTask;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * Application-owned, bounded work-stealing runtime for PFGAP.
 *
 * <p>One runtime represents one worker budget. Sequential mode creates no
 * threads. Parallel mode owns one private {@link ForkJoinPool}; PFGAP code
 * should neither create nested CPU executors nor use the common pool.</p>
 *
 * <p>The runtime supports both per-index ranges and coarse terminal ranges.
 * The latter permit a caller to create one worker-local evaluator or workspace
 * per terminal task and reuse it for every index in that task.</p>
 */
public final class ParallelRuntime implements AutoCloseable {

    private static final long SHUTDOWN_TIMEOUT_SECONDS = 30L;

    private final int workerCount;
    private final ForkJoinPool pool;
    private volatile boolean closed;

    public ParallelRuntime(int requestedWorkers) {
        this.workerCount = resolveWorkerCount(requestedWorkers);
        this.pool = workerCount == 1
                ? null
                : createPool(workerCount);
    }

    public static int resolveWorkerCount(int requestedWorkers) {
        if (requestedWorkers == -1) {
            return Math.max(
                    1,
                    Runtime.getRuntime().availableProcessors()
            );
        }
        if (requestedWorkers < 1) {
            throw new IllegalArgumentException(
                    "num_workers must be -1 or a positive integer. Received: "
                            + requestedWorkers
                            + "."
            );
        }
        return requestedWorkers;
    }

    public int getWorkerCount() {
        return workerCount;
    }

    public boolean isParallel() {
        return workerCount > 1;
    }

    public boolean isSequential() {
        return workerCount == 1;
    }

    public boolean isClosed() {
        return closed;
    }

    public void run(CheckedRunnable action) throws Exception {
        Objects.requireNonNull(action, "Parallel action cannot be null.");
        call(() -> {
            action.run();
            return null;
        });
    }

    public <T> T call(CheckedSupplier<T> computation) throws Exception {
        Objects.requireNonNull(
                computation,
                "Parallel computation cannot be null."
        );
        ensureOpen();
        failIfInterrupted();

        if (isSequential()) {
            return computation.get();
        }

        return invokeTask(
                new CheckedRecursiveTask<>(computation)
        );
    }

    /**
     * Applies an operation exactly once to every index in the half-open range.
     */
    public void forRange(
            int startInclusive,
            int endExclusive,
            int minimumLeafSize,
            CheckedIntConsumer operation
    ) throws Exception {
        Objects.requireNonNull(operation, "Range operation cannot be null.");

        forRanges(
                startInclusive,
                endExclusive,
                minimumLeafSize,
                (rangeStart, rangeEnd) -> {
                    for (int index = rangeStart;
                         index < rangeEnd;
                         index++) {
                        failIfInterrupted();
                        operation.accept(index);
                    }
                }
        );
    }

    /**
     * Partitions a half-open integer range into coarse terminal ranges.
     *
     * <p>The callback is invoked once for each terminal range, not once per
     * index. Terminal ranges are disjoint and collectively cover the original
     * range exactly. A caller may therefore create task-local mutable state in
     * the callback and reuse it throughout that range without synchronization.</p>
     *
     * @param startInclusive first index, inclusive
     * @param endExclusive final index, exclusive
     * @param minimumLeafSize smallest preferred terminal range size
     * @param operation terminal-range operation
     */
    public void forRanges(
            int startInclusive,
            int endExclusive,
            int minimumLeafSize,
            CheckedRangeConsumer operation
    ) throws Exception {
        validateRange(
                startInclusive,
                endExclusive,
                minimumLeafSize,
                operation
        );
        ensureOpen();
        failIfInterrupted();

        int size = endExclusive - startInclusive;
        if (size == 0) {
            return;
        }

        if (isSequential() || size <= minimumLeafSize) {
            operation.accept(startInclusive, endExclusive);
            return;
        }

        int targetLeafSize = effectiveLeafSize(
                size,
                minimumLeafSize
        );

        invokeAction(
                new CheckedRangeTask(
                        startInclusive,
                        endExclusive,
                        targetLeafSize,
                        operation
                )
        );
    }

    /**
     * Returns the private pool for specialized fork-join task graphs.
     * General range work should use {@link #forRange} or {@link #forRanges}.
     */
    public ForkJoinPool getPool() {
        ensureOpen();
        if (pool == null) {
            throw new IllegalStateException(
                    "Sequential ParallelRuntime does not own a ForkJoinPool."
            );
        }
        return pool;
    }

    public boolean isCurrentPoolWorker() {
        return pool != null
                && ForkJoinTask.inForkJoinPool()
                && ForkJoinTask.getPool() == pool;
    }

    public RuntimeSnapshot snapshot() {
        ensureOpen();
        if (pool == null) {
            return new RuntimeSnapshot(
                    workerCount,
                    0,
                    0,
                    0L,
                    0L,
                    true
            );
        }
        return new RuntimeSnapshot(
                workerCount,
                pool.getActiveThreadCount(),
                pool.getRunningThreadCount(),
                pool.getQueuedTaskCount(),
                pool.getStealCount(),
                pool.isQuiescent()
        );
    }

    @Override
    public void close() {
        if (closed) {
            return;
        }
        closed = true;

        if (pool == null) {
            return;
        }

        pool.shutdown();
        try {
            if (!pool.awaitTermination(
                    SHUTDOWN_TIMEOUT_SECONDS,
                    TimeUnit.SECONDS
            )) {
                pool.shutdownNow();
                if (!pool.awaitTermination(
                        SHUTDOWN_TIMEOUT_SECONDS,
                        TimeUnit.SECONDS
                )) {
                    System.err.println(
                            "PFGAP parallel runtime did not terminate cleanly."
                    );
                }
            }
        } catch (InterruptedException interrupted) {
            pool.shutdownNow();
            Thread.currentThread().interrupt();
        }
    }

    private int effectiveLeafSize(
            int size,
            int minimumLeafSize
    ) {
        /*
         * Generate only a modest multiple of the worker count. This supplies
         * enough tasks for stealing while preventing tiny task explosions when
         * callers request a minimum leaf size of one for a very large range.
         */
        long targetTaskCount = Math.max(1L, (long) workerCount * 4L);
        long balancedLeafSize =
                (size + targetTaskCount - 1L) / targetTaskCount;

        return (int) Math.max(
                minimumLeafSize,
                Math.min((long) Integer.MAX_VALUE, balancedLeafSize)
        );
    }

    private <T> T invokeTask(
            CheckedRecursiveTask<T> task
    ) throws Exception {
        try {
            if (isCurrentPoolWorker()) {
                return task.invoke();
            }
            return pool.invoke(task);
        } catch (TaskExecutionException failure) {
            return rethrow(failure.getCause());
        }
    }

    private void invokeAction(
            CheckedRangeTask task
    ) throws Exception {
        try {
            if (isCurrentPoolWorker()) {
                task.invoke();
            } else {
                pool.invoke(task);
            }
        } catch (TaskExecutionException failure) {
            rethrowVoid(failure.getCause());
        }
    }

    private static ForkJoinPool createPool(int workerCount) {
        AtomicInteger workerIds = new AtomicInteger();

        ForkJoinPool.ForkJoinWorkerThreadFactory threadFactory = pool -> {
            var worker =
                    ForkJoinPool.defaultForkJoinWorkerThreadFactory
                            .newThread(pool);
            worker.setName(
                    "pfgap-worker-" + workerIds.incrementAndGet()
            );
            return worker;
        };

        Thread.UncaughtExceptionHandler exceptionHandler =
                (thread, failure) -> System.err.println(
                        "Uncaught failure in "
                                + thread.getName()
                                + ": "
                                + failure
                );

        return new ForkJoinPool(
                workerCount,
                threadFactory,
                exceptionHandler,
                false
        );
    }

    private void ensureOpen() {
        if (closed) {
            throw new IllegalStateException(
                    "ParallelRuntime has already been closed."
            );
        }
    }

    private static void failIfInterrupted()
            throws InterruptedException {
        if (Thread.currentThread().isInterrupted()) {
            throw new InterruptedException(
                    "Parallel execution was interrupted."
            );
        }
    }

    private static void validateRange(
            int startInclusive,
            int endExclusive,
            int minimumLeafSize,
            CheckedRangeConsumer operation
    ) {
        Objects.requireNonNull(operation, "Range operation cannot be null.");
        if (startInclusive < 0) {
            throw new IllegalArgumentException(
                    "Range start cannot be negative: " + startInclusive
            );
        }
        if (endExclusive < startInclusive) {
            throw new IllegalArgumentException(
                    "Range end cannot be less than its start. Start="
                            + startInclusive
                            + ", end="
                            + endExclusive
                            + "."
            );
        }
        if (minimumLeafSize < 1) {
            throw new IllegalArgumentException(
                    "minimumLeafSize must be positive. Received: "
                            + minimumLeafSize
                            + "."
            );
        }
    }

    private static <T> T rethrow(Throwable cause) throws Exception {
        if (cause instanceof InterruptedException interrupted) {
            Thread.currentThread().interrupt();
            throw interrupted;
        }
        if (cause instanceof Exception exception) {
            throw exception;
        }
        if (cause instanceof Error error) {
            throw error;
        }
        throw new IllegalStateException(
                "Parallel task failed with a non-Exception throwable.",
                cause
        );
    }

    private static void rethrowVoid(Throwable cause) throws Exception {
        rethrow(cause);
    }

    @FunctionalInterface
    public interface CheckedRunnable {
        void run() throws Exception;
    }

    @FunctionalInterface
    public interface CheckedSupplier<T> {
        T get() throws Exception;
    }

    @FunctionalInterface
    public interface CheckedIntConsumer {
        void accept(int value) throws Exception;
    }

    @FunctionalInterface
    public interface CheckedRangeConsumer {
        void accept(
                int startInclusive,
                int endExclusive
        ) throws Exception;
    }

    public record RuntimeSnapshot(
            int workerCount,
            int activeThreadCount,
            int runningThreadCount,
            long queuedTaskCount,
            long stealCount,
            boolean quiescent
    ) {
    }

    private static final class CheckedRecursiveTask<T>
            extends RecursiveTask<T> {

        @Serial
        private static final long serialVersionUID = 1L;

        private final CheckedSupplier<T> computation;

        private CheckedRecursiveTask(
                CheckedSupplier<T> computation
        ) {
            this.computation = computation;
        }

        @Override
        protected T compute() {
            try {
                return computation.get();
            } catch (Throwable failure) {
                throw TaskExecutionException.wrap(failure);
            }
        }
    }

    private static final class CheckedRangeTask
            extends RecursiveAction {

        @Serial
        private static final long serialVersionUID = 1L;

        private final int startInclusive;
        private final int endExclusive;
        private final int leafSize;
        private final CheckedRangeConsumer operation;

        private CheckedRangeTask(
                int startInclusive,
                int endExclusive,
                int leafSize,
                CheckedRangeConsumer operation
        ) {
            this.startInclusive = startInclusive;
            this.endExclusive = endExclusive;
            this.leafSize = leafSize;
            this.operation = operation;
        }

        @Override
        protected void compute() {
            if (Thread.currentThread().isInterrupted()) {
                throw TaskExecutionException.wrap(
                        new InterruptedException(
                                "Parallel range execution was interrupted."
                        )
                );
            }

            int size = endExclusive - startInclusive;
            if (size <= leafSize) {
                try {
                    operation.accept(
                            startInclusive,
                            endExclusive
                    );
                    return;
                } catch (Throwable failure) {
                    throw TaskExecutionException.wrap(failure);
                }
            }

            int middle = startInclusive + (size >>> 1);
            CheckedRangeTask left =
                    new CheckedRangeTask(
                            startInclusive,
                            middle,
                            leafSize,
                            operation
                    );
            CheckedRangeTask right =
                    new CheckedRangeTask(
                            middle,
                            endExclusive,
                            leafSize,
                            operation
                    );

            left.fork();
            try {
                right.compute();
                left.join();
            } catch (Throwable failure) {
                left.cancel(true);
                right.cancel(true);
                throw failure;
            }
        }
    }

    private static final class TaskExecutionException
            extends RuntimeException {

        @Serial
        private static final long serialVersionUID = 1L;

        private TaskExecutionException(Throwable cause) {
            super(cause);
        }

        private static TaskExecutionException wrap(Throwable failure) {
            if (failure instanceof TaskExecutionException wrapped) {
                return wrapped;
            }
            return new TaskExecutionException(failure);
        }
    }
}
