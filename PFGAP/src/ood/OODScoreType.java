package ood;

import java.util.Locale;

/**
 * Identifies the out-of-distribution scoring method used during
 * distance-observing forest evaluation.
 *
 * <p>The enum is intentionally small. New scoring methods can be added without
 * changing tree traversal because concrete scorers consume the existing
 * {@link SplitDistanceObserver} callback.</p>
 */
public enum OODScoreType {

    /**
     * Averages branch-local relative support exceedance along each tree path,
     * then averages the resulting tree scores across the forest.
     *
     * <p>For a finite test winning distance {@code d} and finite branch training
     * maximum {@code m}, the node contribution is:</p>
     *
     * <pre>
     * max(0, (d - m) / max(m, scaleFloor))
     * </pre>
     *
     * <p>A branch with an infinite training maximum contributes zero because it
     * does not establish a finite upper support boundary. A positive-infinite
     * test distance against a finite boundary is handled by the scorer's
     * documented finite proxy policy.</p>
     */
    RELATIVE_SUPPORT_EXCEEDANCE;

    /**
     * Parses a user-facing score type without case sensitivity.
     *
     * <p>Hyphens and spaces are accepted as alternatives to underscores. For
     * example, {@code relative-support-exceedance} and
     * {@code relative support exceedance} both resolve to
     * {@link #RELATIVE_SUPPORT_EXCEEDANCE}.</p>
     *
     * @param value score type text
     * @return parsed score type
     * @throws IllegalArgumentException if the value is null, blank, or unknown
     */
    public static OODScoreType parse(String value) {
        if (value == null || value.isBlank()) {
            throw new IllegalArgumentException(
                    "OOD score type cannot be null or blank."
            );
        }

        String normalized = value
                .trim()
                .replace('-', '_')
                .replace(' ', '_')
                .toUpperCase(Locale.ROOT);

        try {
            return valueOf(normalized);
        } catch (IllegalArgumentException exception) {
            throw new IllegalArgumentException(
                    "Unknown OOD score type '" + value
                            + "'. Supported values: "
                            + supportedValues() + ".",
                    exception
            );
        }
    }

    /** Returns the canonical user-facing configuration value. */
    public String configValue() {
        return name().toLowerCase(Locale.ROOT);
    }

    /** Returns a stable comma-separated list of supported configuration values. */
    public static String supportedValues() {
        StringBuilder values = new StringBuilder();
        for (OODScoreType type : values()) {
            if (!values.isEmpty()) {
                values.append(", ");
            }
            values.append(type.configValue());
        }
        return values.toString();
    }
}
