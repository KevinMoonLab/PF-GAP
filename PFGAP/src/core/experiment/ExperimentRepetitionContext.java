package core.experiment;

import core.parallel.ParallelRuntime;

import java.util.LinkedHashMap;
import java.util.Map;

/**
 * Owns all mutable execution state associated with one experiment repetition.
 *
 * <p>Each repetition receives an independent {@link ParallelRuntime}. Closing
 * this context waits for and shuts down that runtime before the next
 * repetition begins.</p>
 *
 * <p>The accumulated metrics, counts, timings, and artifact paths are retained
 * after execution so they can be passed directly to the experiment-result
 * assembler.</p>
 */
public final class ExperimentRepetitionContext
        implements AutoCloseable {

    private final int repetition;
    private final ParallelRuntime parallelRuntime;

    private final Map<String, String> artifacts;
    private final Map<String, Double> additionalMetrics;
    private final Map<String, Long> additionalCounts;
    private final Map<String, Double> additionalTimings;

    private boolean closed;

    public ExperimentRepetitionContext(
            int repetition,
            int requestedWorkers
    ) {
        if (repetition < 0) {
            throw new IllegalArgumentException(
                    "Repetition index cannot be negative. Received: "
                            + repetition
                            + "."
            );
        }

        this.repetition =
                repetition;

        this.parallelRuntime =
                new ParallelRuntime(
                        requestedWorkers
                );

        this.artifacts =
                new LinkedHashMap<>();

        this.additionalMetrics =
                new LinkedHashMap<>();

        this.additionalCounts =
                new LinkedHashMap<>();

        this.additionalTimings =
                new LinkedHashMap<>();
    }

    public int getRepetition() {
        return repetition;
    }

    public int getDisplayRepetition() {
        return repetition + 1;
    }

    public ParallelRuntime getParallelRuntime() {
        ensureOpen();
        return parallelRuntime;
    }

    public int getWorkerCount() {
        ensureOpen();
        return parallelRuntime.getWorkerCount();
    }

    public boolean isParallel() {
        ensureOpen();
        return parallelRuntime.isParallel();
    }

    public Map<String, String> getArtifacts() {
        return artifacts;
    }

    public Map<String, Double> getAdditionalMetrics() {
        return additionalMetrics;
    }

    public Map<String, Long> getAdditionalCounts() {
        return additionalCounts;
    }

    public Map<String, Double> getAdditionalTimings() {
        return additionalTimings;
    }

    public void addArtifact(
            String name,
            String path
    ) {
        requireName(
                name,
                "Artifact"
        );

        artifacts.put(
                name,
                path
        );
    }

    public void addMetric(
            String name,
            double value
    ) {
        requireName(
                name,
                "Metric"
        );

        additionalMetrics.put(
                name,
                value
        );
    }

    public void addCount(
            String name,
            long value
    ) {
        requireName(
                name,
                "Count"
        );

        additionalCounts.put(
                name,
                value
        );
    }

    public void addTimingMilliseconds(
            String name,
            double milliseconds
    ) {
        requireName(
                name,
                "Timing"
        );

        if (!Double.isFinite(milliseconds)
                || milliseconds < 0.0) {

            throw new IllegalArgumentException(
                    "Timing must be finite and nonnegative. "
                            + "Received "
                            + milliseconds
                            + " for "
                            + name
                            + "."
            );
        }

        additionalTimings.put(
                name,
                milliseconds
        );
    }

    public ParallelRuntime.RuntimeSnapshot runtimeSnapshot() {
        ensureOpen();
        return parallelRuntime.snapshot();
    }

    @Override
    public void close() {
        if (closed) {
            return;
        }

        closed =
                true;

        parallelRuntime.close();
    }

    private void ensureOpen() {
        if (closed) {
            throw new IllegalStateException(
                    "Experiment repetition context "
                            + getDisplayRepetition()
                            + " has already been closed."
            );
        }
    }

    private static void requireName(
            String name,
            String valueType
    ) {
        if (name == null || name.isBlank()) {
            throw new IllegalArgumentException(
                    valueType
                            + " name cannot be null or blank."
            );
        }
    }
}