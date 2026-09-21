package distance.missing;

import core.AppContext;
import core.contracts.ObjectDataset;
import util.Pair;

import java.io.Serial;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Objects;
import java.util.Random;

/**
 * DTW-AROW distance for primitive univariate numeric time series containing
 * missing values.
 *
 * <p>DTW-AROW extends the ordinary DTW recurrence in two ways:</p>
 *
 * <ol>
 *     <li>A local comparison contributes zero when either value is missing;
 *     otherwise it contributes the squared numeric difference.</li>
 *     <li>A horizontal or vertical transition is prohibited when the samples
 *     involved in that transition violate the AROW missingness rule.</li>
 * </ol>
 *
 * <p>Supported input pairs are matching {@code double[]} arrays or matching
 * {@code float[]} arrays. Missing numeric values use primitive NaN. Unequal
 * series lengths are supported. Mixed float/double pairs, boxed numeric arrays,
 * and generic numeric object arrays are not supported.</p>
 *
 * <p>Float values are widened individually when read. Dynamic-programming
 * costs, normalization, and returned distances use double precision. No
 * temporary double representation of a float series is allocated.</p>
 *
 * <p>Methods remain synchronized pending the broader distance ownership and
 * concurrency audit.</p>
 */
public class DTWAROW implements Serializable {

    @Serial
    private static final long serialVersionUID = 1L;

    private static final double INF = Double.POSITIVE_INFINITY;
    private static final byte STEP_NONE = 0;
    private static final byte STEP_DIAGONAL = 1;
    private static final byte STEP_HORIZONTAL = 2;
    private static final byte STEP_VERTICAL = 3;

    public DTWAROW() {
    }

    /** Computes unconstrained DTW-AROW distance. */
    public synchronized double distance(
            Object first,
            Object second
    ) {
        return distance(
                first,
                second,
                Double.POSITIVE_INFINITY,
                -1
        );
    }

    /** Computes unconstrained DTW-AROW distance with a best-so-far bound. */
    public synchronized double distance(
            Object first,
            Object second,
            double bestSoFar
    ) {
        return distance(first, second, bestSoFar, -1);
    }

    /**
     * Computes DTW-AROW distance.
     *
     * @param first first {@code double[]} or {@code float[]} series
     * @param second matching second series
     * @param bestSoFar nonnegative ordinary-distance threshold
     * @param windowSize Sakoe-Chiba radius; {@code -1} means unconstrained
     * @return DTW-AROW distance, or positive infinity when unavailable or
     *         greater than {@code bestSoFar}
     */
    public synchronized double distance(
            Object first,
            Object second,
            double bestSoFar,
            int windowSize
    ) {
        validateBestSoFar(bestSoFar);
        AccessorPair pair = makeMatchingAccessors(first, second);
        return compute(
                pair.first(),
                pair.second(),
                bestSoFar,
                windowSize,
                false
        ).distance();
    }

    /**
     * Computes the optimal DTW-AROW alignment path as zero-based index pairs.
     * An unavailable alignment returns an empty list.
     */
    public synchronized List<Pair<Integer, Integer>> getAlignmentPath(
            Object first,
            Object second,
            int windowSize
    ) {
        AccessorPair pair = makeMatchingAccessors(first, second);
        return compute(
                pair.first(),
                pair.second(),
                Double.POSITIVE_INFINITY,
                windowSize,
                true
        ).path();
    }

    private static DTWResult compute(
            SeriesAccessor first,
            SeriesAccessor second,
            double bestSoFar,
            int windowSize,
            boolean keepPath
    ) {
        int firstLength = first.length();
        int secondLength = second.length();
        if (firstLength == 0 || secondLength == 0) {
            return unavailable();
        }

        int availableFirst = first.countAvailable();
        int availableSecond = second.countAvailable();
        int totalAvailable = availableFirst + availableSecond;
        if (totalAvailable == 0) {
            return unavailable();
        }

        int window = resolveWindow(
                windowSize,
                firstLength,
                secondLength
        );
        if (Math.abs(firstLength - secondLength) > window) {
            return unavailable();
        }

        double gamma =
                (double) (firstLength + secondLength) / totalAvailable;
        double[][] cost =
                new double[firstLength + 1][secondLength + 1];
        byte[][] steps = keepPath
                ? new byte[firstLength + 1][secondLength + 1]
                : null;

        for (double[] row : cost) {
            java.util.Arrays.fill(row, INF);
        }

        // The algorithm's zeroth row and column are DP boundaries.
        for (int firstIndex = 0;
                firstIndex <= firstLength;
                firstIndex++) {
            cost[firstIndex][0] = 0.0;
        }
        for (int secondIndex = 0;
                secondIndex <= secondLength;
                secondIndex++) {
            cost[0][secondIndex] = 0.0;
        }

        for (int firstDp = 1;
                firstDp <= firstLength;
                firstDp++) {
            int secondStart = Math.max(1, firstDp - window);
            int secondStop = Math.min(
                    secondLength,
                    firstDp + window
            );

            for (int secondDp = secondStart;
                    secondDp <= secondStop;
                    secondDp++) {
                int firstIndex = firstDp - 1;
                int secondIndex = secondDp - 1;

                double localCost = extendedSquaredDifference(
                        first,
                        firstIndex,
                        second,
                        secondIndex
                );
                double diagonal = cost[firstDp - 1][secondDp - 1];
                double horizontal = invalidHorizontalStep(
                        first,
                        firstIndex,
                        second,
                        secondIndex
                )
                        ? INF
                        : cost[firstDp][secondDp - 1];
                double vertical = invalidVerticalStep(
                        first,
                        firstIndex,
                        second,
                        secondIndex
                )
                        ? INF
                        : cost[firstDp - 1][secondDp];

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

                cost[firstDp][secondDp] = safeAdd(
                        localCost,
                        minimumPrevious
                );
                if (keepPath) {
                    steps[firstDp][secondDp] = selectedStep;
                }
            }
        }

        double finalCost = cost[firstLength][secondLength];
        if (Double.isInfinite(finalCost)) {
            return unavailable();
        }

        double distance = Math.sqrt(gamma * finalCost);
        if (distance > bestSoFar) {
            return unavailable();
        }

        List<Pair<Integer, Integer>> path = keepPath
                ? backtrack(steps, firstLength, secondLength)
                : Collections.emptyList();
        return new DTWResult(distance, path);
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
        double difference =
                first.value(firstIndex) - second.value(secondIndex);
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

    private static double safeAdd(
            double first,
            double second
    ) {
        if (Double.isInfinite(first) || Double.isInfinite(second)) {
            return INF;
        }
        return first + second;
    }

    private static List<Pair<Integer, Integer>> backtrack(
            byte[][] steps,
            int firstLength,
            int secondLength
    ) {
        List<Pair<Integer, Integer>> reversePath = new ArrayList<>();
        int firstDp = firstLength;
        int secondDp = secondLength;

        while (firstDp > 0 && secondDp > 0) {
            reversePath.add(
                    new Pair<>(firstDp - 1, secondDp - 1)
            );
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

        Collections.reverse(reversePath);
        return reversePath;
    }

    private static int resolveWindow(
            int windowSize,
            int firstLength,
            int secondLength
    ) {
        if (windowSize == -1) {
            return Math.max(firstLength, secondLength);
        }
        if (windowSize < 0) {
            throw new IllegalArgumentException(
                    "windowSize must be -1 or a nonnegative integer."
            );
        }
        return windowSize;
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
                            + "Received: "
                            + bestSoFar
                            + "."
            );
        }
    }

    private static IllegalArgumentException unsupportedPair(
            Object first,
            Object second
    ) {
        return new IllegalArgumentException(
                "DTWAROW requires matching double[] or float[] inputs. "
                        + "Received "
                        + typeName(first)
                        + " and "
                        + typeName(second)
                        + ". Mixed float/double pairs and boxed numeric arrays "
                        + "are not supported."
        );
    }

    private static String typeName(Object value) {
        return value == null
                ? "null"
                : value.getClass().getTypeName();
    }

    private interface SeriesAccessor {
        int length();

        boolean isMissing(int index);

        double value(int index);

        int countAvailable();

        default boolean isMissingIfInRange(int index) {
            return index >= 0
                    && index < length()
                    && isMissing(index);
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

    private record DTWResult(
            double distance,
            List<Pair<Integer, Integer>> path
    ) {
    }

    private static DTWResult unavailable() {
        return new DTWResult(INF, Collections.emptyList());
    }
}
