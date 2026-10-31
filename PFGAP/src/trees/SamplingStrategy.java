package trees;

import java.util.Locale;

/**
 * Whole-tree training-data sampling strategy.
 */
public enum SamplingStrategy {
    /** Use every training observation exactly once, in source order. */
    FULL,

    /** Sample training observations with replacement. */
    BOOTSTRAP,

    /** Sample distinct training observations without replacement. */
    SUBSAMPLE;

    /**
     * Parses a user-facing sampling strategy.
     *
     * @param raw configured strategy name
     * @return parsed strategy
     * @throws IllegalArgumentException if the value is null, blank, or unknown
     */
    public static SamplingStrategy parse(String raw) {
        if (raw == null || raw.isBlank()) {
            throw new IllegalArgumentException(
                    "sampling cannot be null or blank. Expected full, bootstrap, or subsample."
            );
        }
        String normalized = raw
                .trim()
                .replace('-', '_')
                .toUpperCase(Locale.ROOT);
        try {
            return SamplingStrategy.valueOf(normalized);
        } catch (IllegalArgumentException exception) {
            throw new IllegalArgumentException(
                    "Invalid sampling strategy '" + raw
                            + "'. Expected full, bootstrap, or subsample.",
                    exception
            );
        }
    }
}
