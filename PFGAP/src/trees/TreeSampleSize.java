package trees;

import java.io.Serial;
import java.io.Serializable;
import java.util.Objects;

/**
 * User-configured whole-tree sample size.
 *
 * <p>A missing value means the full training-set size. An integer value is an
 * exact observation count, while a floating-point value is a fraction of the
 * training-set size. Keeping the two forms distinct ensures that {@code 1}
 * means one observation and {@code 1.0} means the full dataset.</p>
 */
public final class TreeSampleSize implements Serializable {
    @Serial
    private static final long serialVersionUID = 1L;

    private enum Kind {
        FULL_SIZE,
        COUNT,
        FRACTION
    }

    private static final TreeSampleSize FULL_SIZE =
            new TreeSampleSize(Kind.FULL_SIZE, 0, Double.NaN);

    private final Kind kind;
    private final int count;
    private final double fraction;

    private TreeSampleSize(
            Kind kind,
            int count,
            double fraction
    ) {
        this.kind = Objects.requireNonNull(kind, "Sample-size kind cannot be null.");
        this.count = count;
        this.fraction = fraction;
    }

    /** Returns the default size specification, which resolves to population size. */
    public static TreeSampleSize fullSize() {
        return FULL_SIZE;
    }

    /** Creates an exact-count sample-size specification. */
    public static TreeSampleSize count(int count) {
        if (count < 1) {
            throw new IllegalArgumentException(
                    "Integer sample_size must be positive, but received " + count + "."
            );
        }
        return new TreeSampleSize(Kind.COUNT, count, Double.NaN);
    }

    /** Creates a fractional sample-size specification. */
    public static TreeSampleSize fraction(double fraction) {
        if (!Double.isFinite(fraction)
                || fraction <= 0.0
                || fraction > 1.0) {
            throw new IllegalArgumentException(
                    "Floating-point sample_size must be finite and within (0, 1], "
                            + "but received " + fraction + "."
            );
        }
        return new TreeSampleSize(Kind.FRACTION, 0, fraction);
    }

    /**
     * Parses the command-line representation emitted by the Python wrapper.
     *
     * <p>{@code None}, {@code null}, and blank values select the default full
     * population size. Values containing a decimal point or exponent are
     * interpreted as fractions; other values are interpreted as integer
     * counts.</p>
     */
    public static TreeSampleSize parse(String raw) {
        if (raw == null
                || raw.isBlank()
                || raw.trim().equalsIgnoreCase("none")
                || raw.trim().equalsIgnoreCase("null")) {
            return fullSize();
        }
        String normalized = raw.trim();
        boolean fractionalSyntax = normalized.indexOf('.') >= 0
                || normalized.indexOf('e') >= 0
                || normalized.indexOf('E') >= 0;
        try {
            if (fractionalSyntax) {
                return fraction(Double.parseDouble(normalized));
            }
            return count(Integer.parseInt(normalized));
        } catch (NumberFormatException exception) {
            throw new IllegalArgumentException(
                    "Invalid sample_size '" + raw
                            + "'. Expected None, a positive integer, or a fraction within (0, 1].",
                    exception
            );
        }
    }

    /** Returns whether this is the default population-sized specification. */
    public boolean isFullSize() {
        return kind == Kind.FULL_SIZE;
    }

    /**
     * Resolves this specification for a concrete population and strategy.
     *
     * @param populationSize number of available training observations
     * @param strategy sampling strategy
     * @return number of observations or draws for one tree
     */
    public int resolve(
            int populationSize,
            SamplingStrategy strategy
    ) {
        if (populationSize < 1) {
            throw new IllegalArgumentException(
                    "Sampling requires a positive population size, but received "
                            + populationSize + "."
            );
        }
        Objects.requireNonNull(strategy, "Sampling strategy cannot be null.");
        if (strategy == SamplingStrategy.FULL && !isFullSize()) {
            throw new IllegalArgumentException(
                    "sample_size must be None when sampling='full'."
            );
        }

        int resolved = switch (kind) {
            case FULL_SIZE -> populationSize;
            case COUNT -> count;
            case FRACTION -> Math.max(
                    1,
                    (int) Math.ceil(fraction * populationSize)
            );
        };

        if (strategy == SamplingStrategy.SUBSAMPLE
                && resolved > populationSize) {
            throw new IllegalArgumentException(
                    "subsample sample_size cannot exceed the training-set size. "
                            + "Resolved sample size " + resolved
                            + " exceeds population size " + populationSize + "."
            );
        }
        return resolved;
    }

    @Override
    public String toString() {
        return switch (kind) {
            case FULL_SIZE -> "None";
            case COUNT -> Integer.toString(count);
            case FRACTION -> Double.toString(fraction);
        };
    }
}
