package distance.missing;

import core.AppContext;
import core.contracts.ObjectDataset;
import util.Pair;

import java.io.Serial;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Objects;
import java.util.Random;

/**
 * Faithful univariate DTW-AROW for primitive numeric time series containing
 * missing values represented by NaN.
 *
 * <p>The implementation preserves the published DTW-AROW algorithm:</p>
 * <ul>
 *     <li>a comparison involving a missing sample has local cost zero;</li>
 *     <li>horizontal and vertical steps are prohibited when either endpoint
 *         of that step involves a missing sample;</li>
 *     <li>the dynamic-programming zeroth row and column are initialized to
 *         zero;</li>
 *     <li>the final distance is {@code sqrt(gamma * cumulativeCost)}, where
 *         {@code gamma = (length1 + length2) / (available1 + available2)}.</li>
 * </ul>
 *
 * <p>A negative-one window reproduces unconstrained DTW-AROW. A nonnegative
 * window is an explicit additional Sakoe-Chiba constraint and is never silently
 * widened. If it cannot connect the endpoint pair, the result is unavailable.</p>
 *
 * <p>Distance-only evaluation uses two cost rows. Path evaluation additionally
 * stores one predecessor byte per matrix cell. A finite {@code bestSoFar} is
 * converted exactly to a cumulative-cost cutoff, allowing safe cell pruning
 * without changing any result capable of beating the bound.</p>
 */
public final class DTWAROW implements Serializable {

    @Serial
    private static final long serialVersionUID = 1L;

    private static final double INF = Double.POSITIVE_INFINITY;
    private static final byte STEP_NONE = 0;
    private static final byte STEP_DIAGONAL = 1;
    private static final byte STEP_HORIZONTAL = 2;
    private static final byte STEP_VERTICAL = 3;

    public DTWAROW() {
    }

    public double distance(Object first, Object second) {
        return distance(first, second, Double.POSITIVE_INFINITY, -1);
    }

    public double distance(
            Object first,
            Object second,
            double bestSoFar
    ) {
        return distance(first, second, bestSoFar, -1);
    }

    public double distance(
            Object first,
            Object second,
            double bestSoFar,
            int windowSize
    ) {
        validateBestSoFar(bestSoFar);
        validateWindow(windowSize);
        AccessorPair pair = makeMatchingAccessors(first, second);
        return computeDistance(
                pair.first(),
                pair.second(),
                bestSoFar,
                windowSize
        );
    }

    public List<Pair<Integer, Integer>> getAlignmentPath(
            Object first,
            Object second,
            int windowSize
    ) {
        validateWindow(windowSize);
        AccessorPair pair = makeMatchingAccessors(first, second);
        return computePath(pair.first(), pair.second(), windowSize);
    }

    private static double computeDistance(
            SeriesAccessor first,
            SeriesAccessor second,
            double bestSoFar,
            int windowSize
    ) {
        Preparation preparation = prepare(first, second, windowSize);
        if (!preparation.available()) {
            return INF;
        }

        int firstLength = first.length();
        int secondLength = second.length();
        int window = preparation.window();
        double gamma = preparation.gamma();
        double rawCutoff = rawCutoff(bestSoFar, gamma);

        double[] previous = new double[secondLength + 1];
        double[] current = new double[secondLength + 1];

        // Published DTW-AROW boundary convention: C[j,0] = C[0,j'] = 0.
        Arrays.fill(previous, 0.0);

        for (int firstDp = 1; firstDp <= firstLength; firstDp++) {
            Arrays.fill(current, INF);
            current[0] = 0.0;

            int secondStart = Math.max(1, firstDp - window);
            int secondStop = Math.min(secondLength, firstDp + window);

            for (int secondDp = secondStart;
                 secondDp <= secondStop;
                 secondDp++) {
                int firstIndex = firstDp - 1;
                int secondIndex = secondDp - 1;

                double diagonal = previous[secondDp - 1];
                double horizontal = invalidHorizontalStep(
                        first,
                        firstIndex,
                        second,
                        secondIndex
                ) ? INF : current[secondDp - 1];
                double vertical = invalidVerticalStep(
                        first,
                        firstIndex,
                        second,
                        secondIndex
                ) ? INF : previous[secondDp];

                double minimumPrevious = minimum(
                        diagonal,
                        horizontal,
                        vertical
                );
                double cumulative = safeAdd(
                        extendedSquaredDifference(
                                first,
                                firstIndex,
                                second,
                                secondIndex
                        ),
                        minimumPrevious
                );
                current[secondDp] = cumulative > rawCutoff
                        ? INF
                        : cumulative;
            }

            double[] temporary = previous;
            previous = current;
            current = temporary;
        }

        double finalCost = previous[secondLength];
        if (Double.isInfinite(finalCost)) {
            return INF;
        }
        return Math.sqrt(gamma * finalCost);
    }

    private static List<Pair<Integer, Integer>> computePath(
            SeriesAccessor first,
            SeriesAccessor second,
            int windowSize
    ) {
        Preparation preparation = prepare(first, second, windowSize);
        if (!preparation.available()) {
            return Collections.emptyList();
        }

        int firstLength = first.length();
        int secondLength = second.length();
        int window = preparation.window();
        double[] previous = new double[secondLength + 1];
        double[] current = new double[secondLength + 1];
        byte[][] steps = new byte[firstLength + 1][secondLength + 1];

        Arrays.fill(previous, 0.0);

        for (int firstDp = 1; firstDp <= firstLength; firstDp++) {
            Arrays.fill(current, INF);
            current[0] = 0.0;

            int secondStart = Math.max(1, firstDp - window);
            int secondStop = Math.min(secondLength, firstDp + window);

            for (int secondDp = secondStart;
                 secondDp <= secondStop;
                 secondDp++) {
                int firstIndex = firstDp - 1;
                int secondIndex = secondDp - 1;

                double diagonal = previous[secondDp - 1];
                double horizontal = invalidHorizontalStep(
                        first,
                        firstIndex,
                        second,
                        secondIndex
                ) ? INF : current[secondDp - 1];
                double vertical = invalidVerticalStep(
                        first,
                        firstIndex,
                        second,
                        secondIndex
                ) ? INF : previous[secondDp];

                double minimumPrevious = diagonal;
                byte selectedStep = STEP_DIAGONAL;
                if (horizontal < minimumPrevious) {
                    minimumPrevious = horizontal;
                    selectedStep = STEP_HORIZONTAL;
                }
                if (vertical < minimumPrevious) {
                    minimumPrevious = vertical;
                    selectedStep = STEP_VERTICAL;
                }

                current[secondDp] = safeAdd(
                        extendedSquaredDifference(
                                first,
                                firstIndex,
                                second,
                                secondIndex
                        ),
                        minimumPrevious
                );
                if (!Double.isInfinite(current[secondDp])) {
                    steps[firstDp][secondDp] = selectedStep;
                }
            }

            double[] temporary = previous;
            previous = current;
            current = temporary;
        }

        if (Double.isInfinite(previous[secondLength])) {
            return Collections.emptyList();
        }
        return backtrack(steps, firstLength, secondLength);
    }

    private static Preparation prepare(
            SeriesAccessor first,
            SeriesAccessor second,
            int windowSize
    ) {
        int firstLength = first.length();
        int secondLength = second.length();
        if (firstLength == 0 || secondLength == 0) {
            return Preparation.UNAVAILABLE;
        }

        int availableFirst = first.countAvailable();
        int availableSecond = second.countAvailable();
        int totalAvailable = availableFirst + availableSecond;
        if (totalAvailable == 0) {
            return Preparation.UNAVAILABLE;
        }

        int window = windowSize == -1
                ? Math.max(firstLength, secondLength)
                : windowSize;
        if (Math.abs(firstLength - secondLength) > window) {
            return Preparation.UNAVAILABLE;
        }

        double gamma = (double) (firstLength + secondLength)
                / totalAvailable;
        return new Preparation(true, window, gamma);
    }

    private static double rawCutoff(
            double bestSoFar,
            double gamma
    ) {
        if (bestSoFar == Double.POSITIVE_INFINITY) {
            return INF;
        }
        return bestSoFar * bestSoFar / gamma;
    }

    private static double extendedSquaredDifference(
            SeriesAccessor first,
            int firstIndex,
            SeriesAccessor second,
            int secondIndex
    ) {
        if (first.isMissing(firstIndex)
                || second.isMissing(secondIndex)) {
            return 0.0;
        }
        double difference = first.value(firstIndex)
                - second.value(secondIndex);
        return difference * difference;
    }

    private static boolean invalidHorizontalStep(
            SeriesAccessor first,
            int firstIndex,
            SeriesAccessor second,
            int secondIndex
    ) {
        return first.isMissing(firstIndex)
                || second.isMissing(secondIndex)
                || second.isMissingIfInRange(secondIndex - 1);
    }

    private static boolean invalidVerticalStep(
            SeriesAccessor first,
            int firstIndex,
            SeriesAccessor second,
            int secondIndex
    ) {
        return first.isMissing(firstIndex)
                || first.isMissingIfInRange(firstIndex - 1)
                || second.isMissing(secondIndex);
    }

    private static double safeAdd(double first, double second) {
        if (Double.isInfinite(first) || Double.isInfinite(second)) {
            return INF;
        }
        return first + second;
    }

    private static double minimum(
            double first,
            double second,
            double third
    ) {
        double minimum = first < second ? first : second;
        return minimum < third ? minimum : third;
    }

    private static List<Pair<Integer, Integer>> backtrack(
            byte[][] steps,
            int firstLength,
            int secondLength
    ) {
        List<Pair<Integer, Integer>> reversed = new ArrayList<>(
                firstLength + secondLength - 1
        );
        int firstDp = firstLength;
        int secondDp = secondLength;

        while (firstDp > 0 && secondDp > 0) {
            reversed.add(new Pair<>(firstDp - 1, secondDp - 1));
            byte step = steps[firstDp][secondDp];
            if (step == STEP_DIAGONAL) {
                firstDp--;
                secondDp--;
            } else if (step == STEP_HORIZONTAL) {
                secondDp--;
            } else if (step == STEP_VERTICAL) {
                firstDp--;
            } else {
                return Collections.emptyList();
            }
        }

        Collections.reverse(reversed);
        return reversed;
    }

    private static AccessorPair makeMatchingAccessors(
            Object first,
            Object second
    ) {
        if (first instanceof double[] firstValues
                && second instanceof double[] secondValues) {
            return new AccessorPair(
                    new DoubleSeriesAccessor(firstValues),
                    new DoubleSeriesAccessor(secondValues)
            );
        }
        if (first instanceof float[] firstValues
                && second instanceof float[] secondValues) {
            return new AccessorPair(
                    new FloatSeriesAccessor(firstValues),
                    new FloatSeriesAccessor(secondValues)
            );
        }
        throw unsupportedPair(first, second);
    }

    public int get_random_window(
            ObjectDataset dataset,
            Random random
    ) {
        Objects.requireNonNull(dataset, "Dataset cannot be null.");
        Objects.requireNonNull(random, "Random cannot be null.");
        int bound = Math.max(1, (AppContext.length + 1) / 4);
        return random.nextInt(bound);
    }

    private static void validateBestSoFar(double bestSoFar) {
        if (Double.isNaN(bestSoFar) || bestSoFar < 0.0) {
            throw new IllegalArgumentException(
                    "DTWAROW bestSoFar must be nonnegative and not NaN. "
                            + "Received: " + bestSoFar + "."
            );
        }
    }

    private static void validateWindow(int windowSize) {
        if (windowSize < -1) {
            throw new IllegalArgumentException(
                    "windowSize must be -1 or a nonnegative integer."
            );
        }
    }

    private static IllegalArgumentException unsupportedPair(
            Object first,
            Object second
    ) {
        return new IllegalArgumentException(
                "DTWAROW requires matching double[] or float[] inputs. "
                        + "Received " + typeName(first) + " and "
                        + typeName(second) + "."
        );
    }

    private static String typeName(Object value) {
        return value == null ? "null" : value.getClass().getTypeName();
    }

    private interface SeriesAccessor {
        int length();

        boolean isMissing(int index);

        double value(int index);

        int countAvailable();

        default boolean isMissingIfInRange(int index) {
            return index >= 0 && index < length() && isMissing(index);
        }
    }

    private static final class DoubleSeriesAccessor
            implements SeriesAccessor {
        private final double[] series;

        private DoubleSeriesAccessor(double[] series) {
            this.series = Objects.requireNonNull(
                    series,
                    "Double series cannot be null."
            );
        }

        @Override
        public int length() {
            return series.length;
        }

        @Override
        public boolean isMissing(int index) {
            return Double.isNaN(series[index]);
        }

        @Override
        public double value(int index) {
            return series[index];
        }

        @Override
        public int countAvailable() {
            return MissingDistanceTools.countAvailable(series);
        }
    }

    private static final class FloatSeriesAccessor
            implements SeriesAccessor {
        private final float[] series;

        private FloatSeriesAccessor(float[] series) {
            this.series = Objects.requireNonNull(
                    series,
                    "Float series cannot be null."
            );
        }

        @Override
        public int length() {
            return series.length;
        }

        @Override
        public boolean isMissing(int index) {
            return Float.isNaN(series[index]);
        }

        @Override
        public double value(int index) {
            return series[index];
        }

        @Override
        public int countAvailable() {
            return MissingDistanceTools.countAvailable(series);
        }
    }

    private record AccessorPair(
            SeriesAccessor first,
            SeriesAccessor second
    ) {
    }

    private record Preparation(
            boolean available,
            int window,
            double gamma
    ) {
        private static final Preparation UNAVAILABLE =
                new Preparation(false, 0, Double.NaN);
    }
}
