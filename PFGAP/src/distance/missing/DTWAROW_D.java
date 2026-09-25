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
 * Dependent multivariate extension of DTW-AROW.
 *
 * <p>This is a research extension of the published univariate DTW-AROW
 * algorithm. It preserves the essential AROW principles while using one shared
 * warping path across selected dimensions:</p>
 *
 * <ul>
 *     <li>A time-pair comparison is computable when at least one selected
 *         dimension is jointly observed.</li>
 *     <li>A computable local cost is the scaled squared NaN-Euclidean cost
 *         {@code selectedCount / jointlyObservedCount * squaredSum}.</li>
 *     <li>An incomputable comparison has local cost zero.</li>
 *     <li>Horizontal and vertical transitions are prohibited when either
 *         endpoint comparison is incomputable.</li>
 *     <li>Diagonal transitions remain available through incomputable pairs.</li>
 *     <li>The published zero-boundary and availability-correction principles
 *         are retained.</li>
 * </ul>
 *
 * <p>Matching {@code double[][]}, {@code float[][]}, and numeric
 * {@code Object[][]} inputs are supported. Primitive matrices use NaN for
 * missingness; object matrices additionally accept null. Object matrices are
 * retained as a compatibility path for missing-data imputation and interop,
 * while primitive matrices remain the preferred numerical representation.</p>
 *
 * <p>Distance-only evaluation uses two cost rows and two comparability rows.
 * Path evaluation additionally stores one predecessor byte per matrix cell.
 * Each local cost and comparability result is computed once per cell and then
 * reused for transition checks. A finite {@code bestSoFar} is converted to the
 * exact cumulative-cost cutoff implied by the availability correction.</p>
 *
 * <p>A window of {@code -1} is unconstrained. A nonnegative window is an
 * explicit additional Sakoe-Chiba constraint and is never silently widened.</p>
 */
public final class DTWAROW_D implements Serializable {

    @Serial
    private static final long serialVersionUID = 1L;

    private static final double INF = Double.POSITIVE_INFINITY;
    private static final byte STEP_NONE = 0;
    private static final byte STEP_DIAGONAL = 1;
    private static final byte STEP_HORIZONTAL = 2;
    private static final byte STEP_VERTICAL = 3;

    public DTWAROW_D() {
    }

    public double distance(Object first, Object second) {
        return distance(first, second, Double.POSITIVE_INFINITY, -1, null);
    }

    public double distance(
            Object first,
            Object second,
            double bestSoFar
    ) {
        return distance(first, second, bestSoFar, -1, null);
    }

    public double distance(
            Object first,
            Object second,
            double bestSoFar,
            int windowSize
    ) {
        return distance(first, second, bestSoFar, windowSize, null);
    }

    public double distance(
            Object first,
            Object second,
            double bestSoFar,
            int windowSize,
            int[] selectedDimensions
    ) {
        validateBestSoFar(bestSoFar);
        validateWindow(windowSize);
        AccessorPair pair = makeMatchingAccessors(first, second);
        return computeDistance(
                pair.first(),
                pair.second(),
                bestSoFar,
                windowSize,
                selectedDimensions
        );
    }

    public List<Pair<Integer, Integer>> getAlignmentPath(
            Object first,
            Object second,
            int windowSize
    ) {
        return getAlignmentPath(first, second, windowSize, null);
    }

    public List<Pair<Integer, Integer>> getAlignmentPath(
            Object first,
            Object second,
            int windowSize,
            int[] selectedDimensions
    ) {
        validateWindow(windowSize);
        AccessorPair pair = makeMatchingAccessors(first, second);
        return computePath(
                pair.first(),
                pair.second(),
                windowSize,
                selectedDimensions
        );
    }

    private static double computeDistance(
            MultiSeriesAccessor first,
            MultiSeriesAccessor second,
            double bestSoFar,
            int windowSize,
            int[] selectedDimensions
    ) {
        Preparation preparation = prepare(
                first,
                second,
                windowSize,
                selectedDimensions
        );
        if (!preparation.available()) {
            return INF;
        }

        int firstLength = first.length();
        int secondLength = second.length();
        int window = preparation.window();
        int selectedCount = preparation.selectedCount();
        double gamma = preparation.gamma();
        double rawCutoff = rawCutoff(bestSoFar, gamma);

        double[] previousCost = new double[secondLength + 1];
        double[] currentCost = new double[secondLength + 1];
        boolean[] previousComparable = new boolean[secondLength + 1];
        boolean[] currentComparable = new boolean[secondLength + 1];

        // Published AROW boundary convention: C[i,0] = C[0,j] = 0.
        Arrays.fill(previousCost, 0.0);

        for (int firstDp = 1; firstDp <= firstLength; firstDp++) {
            Arrays.fill(currentCost, INF);
            Arrays.fill(currentComparable, false);
            currentCost[0] = 0.0;

            int secondStart = Math.max(1, firstDp - window);
            int secondStop = Math.min(secondLength, firstDp + window);
            int firstIndex = firstDp - 1;

            for (int secondDp = secondStart;
                 secondDp <= secondStop;
                 secondDp++) {
                int secondIndex = secondDp - 1;
                LocalComparison local = localComparison(
                        first,
                        firstIndex,
                        second,
                        secondIndex,
                        selectedDimensions,
                        selectedCount
                );
                currentComparable[secondDp] = local.comparable();

                double diagonal = previousCost[secondDp - 1];
                double horizontal = local.comparable()
                        && (secondIndex == 0
                        || currentComparable[secondDp - 1])
                        ? currentCost[secondDp - 1]
                        : INF;
                double vertical = local.comparable()
                        && (firstIndex == 0
                        || previousComparable[secondDp])
                        ? previousCost[secondDp]
                        : INF;

                double cumulative = safeAdd(
                        local.cost(),
                        minimum(diagonal, horizontal, vertical)
                );
                currentCost[secondDp] = cumulative > rawCutoff
                        ? INF
                        : cumulative;
            }

            double[] temporaryCost = previousCost;
            previousCost = currentCost;
            currentCost = temporaryCost;

            boolean[] temporaryComparable = previousComparable;
            previousComparable = currentComparable;
            currentComparable = temporaryComparable;
        }

        double finalCost = previousCost[secondLength];
        if (Double.isInfinite(finalCost)) {
            return INF;
        }
        return Math.sqrt(gamma * finalCost);
    }

    private static List<Pair<Integer, Integer>> computePath(
            MultiSeriesAccessor first,
            MultiSeriesAccessor second,
            int windowSize,
            int[] selectedDimensions
    ) {
        Preparation preparation = prepare(
                first,
                second,
                windowSize,
                selectedDimensions
        );
        if (!preparation.available()) {
            return Collections.emptyList();
        }

        int firstLength = first.length();
        int secondLength = second.length();
        int window = preparation.window();
        int selectedCount = preparation.selectedCount();

        double[] previousCost = new double[secondLength + 1];
        double[] currentCost = new double[secondLength + 1];
        boolean[] previousComparable = new boolean[secondLength + 1];
        boolean[] currentComparable = new boolean[secondLength + 1];
        byte[][] steps = new byte[firstLength + 1][secondLength + 1];

        Arrays.fill(previousCost, 0.0);

        for (int firstDp = 1; firstDp <= firstLength; firstDp++) {
            Arrays.fill(currentCost, INF);
            Arrays.fill(currentComparable, false);
            currentCost[0] = 0.0;

            int secondStart = Math.max(1, firstDp - window);
            int secondStop = Math.min(secondLength, firstDp + window);
            int firstIndex = firstDp - 1;

            for (int secondDp = secondStart;
                 secondDp <= secondStop;
                 secondDp++) {
                int secondIndex = secondDp - 1;
                LocalComparison local = localComparison(
                        first,
                        firstIndex,
                        second,
                        secondIndex,
                        selectedDimensions,
                        selectedCount
                );
                currentComparable[secondDp] = local.comparable();

                double diagonal = previousCost[secondDp - 1];
                double horizontal = local.comparable()
                        && (secondIndex == 0
                        || currentComparable[secondDp - 1])
                        ? currentCost[secondDp - 1]
                        : INF;
                double vertical = local.comparable()
                        && (firstIndex == 0
                        || previousComparable[secondDp])
                        ? previousCost[secondDp]
                        : INF;

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

                currentCost[secondDp] = safeAdd(
                        local.cost(),
                        minimumPrevious
                );
                if (!Double.isInfinite(currentCost[secondDp])) {
                    steps[firstDp][secondDp] = selectedStep;
                }
            }

            double[] temporaryCost = previousCost;
            previousCost = currentCost;
            currentCost = temporaryCost;

            boolean[] temporaryComparable = previousComparable;
            previousComparable = currentComparable;
            currentComparable = temporaryComparable;
        }

        if (Double.isInfinite(previousCost[secondLength])) {
            return Collections.emptyList();
        }
        return backtrack(steps, firstLength, secondLength);
    }

    private static Preparation prepare(
            MultiSeriesAccessor first,
            MultiSeriesAccessor second,
            int windowSize,
            int[] selectedDimensions
    ) {
        validateCompatibleSeries(first, second);

        int selectedCount = selectedDimensions == null
                ? first.dimensions()
                : selectedDimensions.length;
        if (selectedCount == 0
                || first.length() == 0
                || second.length() == 0) {
            return Preparation.UNAVAILABLE;
        }

        int availableFirst = countAvailableTimePoints(
                first,
                selectedDimensions,
                selectedCount
        );
        int availableSecond = countAvailableTimePoints(
                second,
                selectedDimensions,
                selectedCount
        );
        int totalAvailable = availableFirst + availableSecond;
        if (totalAvailable == 0) {
            return Preparation.UNAVAILABLE;
        }

        int window = windowSize == -1
                ? Math.max(first.length(), second.length())
                : windowSize;
        if (Math.abs(first.length() - second.length()) > window) {
            return Preparation.UNAVAILABLE;
        }

        double gamma = (double) (first.length() + second.length())
                / totalAvailable;
        return new Preparation(true, window, gamma, selectedCount);
    }

    private static int countAvailableTimePoints(
            MultiSeriesAccessor series,
            int[] selectedDimensions,
            int selectedCount
    ) {
        int count = 0;
        for (int time = 0; time < series.length(); time++) {
            for (int position = 0; position < selectedCount; position++) {
                int dimension = selectedDimensions == null
                        ? position
                        : selectedDimensions[position];
                if (!series.isMissing(dimension, time)) {
                    count++;
                    break;
                }
            }
        }
        return count;
    }

    private static LocalComparison localComparison(
            MultiSeriesAccessor first,
            int firstTime,
            MultiSeriesAccessor second,
            int secondTime,
            int[] selectedDimensions,
            int selectedCount
    ) {
        double squaredSum = 0.0;
        int jointlyObserved = 0;

        for (int position = 0; position < selectedCount; position++) {
            int dimension = selectedDimensions == null
                    ? position
                    : selectedDimensions[position];
            if (first.isMissing(dimension, firstTime)
                    || second.isMissing(dimension, secondTime)) {
                continue;
            }

            double difference = first.value(dimension, firstTime)
                    - second.value(dimension, secondTime);
            squaredSum += difference * difference;
            jointlyObserved++;
        }

        if (jointlyObserved == 0) {
            return LocalComparison.INCOMPUTABLE;
        }

        return new LocalComparison(
                true,
                (double) selectedCount / jointlyObserved * squaredSum
        );
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

    private static double minimum(
            double first,
            double second,
            double third
    ) {
        double minimum = first < second ? first : second;
        return minimum < third ? minimum : third;
    }

    private static double safeAdd(double first, double second) {
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

    private static void validateCompatibleSeries(
            MultiSeriesAccessor first,
            MultiSeriesAccessor second
    ) {
        if (first.dimensions() != second.dimensions()) {
            throw new IllegalArgumentException(
                    "Both multivariate series must have the same number of "
                            + "dimensions."
            );
        }
    }

    private static AccessorPair makeMatchingAccessors(
            Object first,
            Object second
    ) {
        if (first instanceof double[][] firstValues
                && second instanceof double[][] secondValues) {
            return new AccessorPair(
                    new DoubleMatrixAccessor(firstValues),
                    new DoubleMatrixAccessor(secondValues)
            );
        }
        if (first instanceof float[][] firstValues
                && second instanceof float[][] secondValues) {
            return new AccessorPair(
                    new FloatMatrixAccessor(firstValues),
                    new FloatMatrixAccessor(secondValues)
            );
        }
        if (first instanceof Object[][] firstValues
                && second instanceof Object[][] secondValues) {
            return new AccessorPair(
                    new ObjectMatrixAccessor(firstValues),
                    new ObjectMatrixAccessor(secondValues)
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
                    "DTWAROW_D bestSoFar must be nonnegative and not NaN. "
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
                "DTWAROW_D requires matching double[][], float[][], or "
                        + "numeric Object[][] inputs. Received "
                        + typeName(first) + " and " + typeName(second) + "."
        );
    }

    private static String typeName(Object value) {
        return value == null ? "null" : value.getClass().getTypeName();
    }

    private interface MultiSeriesAccessor {
        int dimensions();

        int length();

        boolean isMissing(int dimension, int timeIndex);

        double value(int dimension, int timeIndex);
    }

    private abstract static class AbstractMatrixAccessor
            implements MultiSeriesAccessor {
        private final int dimensions;
        private final int length;

        private AbstractMatrixAccessor(int dimensions, int length) {
            this.dimensions = dimensions;
            this.length = length;
        }

        @Override
        public final int dimensions() {
            return dimensions;
        }

        @Override
        public final int length() {
            return length;
        }
    }

    private static final class DoubleMatrixAccessor
            extends AbstractMatrixAccessor {
        private final double[][] series;

        private DoubleMatrixAccessor(double[][] series) {
            super(series.length, validateRectangular(series));
            this.series = series;
        }

        @Override
        public boolean isMissing(int dimension, int timeIndex) {
            return Double.isNaN(series[dimension][timeIndex]);
        }

        @Override
        public double value(int dimension, int timeIndex) {
            return series[dimension][timeIndex];
        }
    }

    private static final class FloatMatrixAccessor
            extends AbstractMatrixAccessor {
        private final float[][] series;

        private FloatMatrixAccessor(float[][] series) {
            super(series.length, validateRectangular(series));
            this.series = series;
        }

        @Override
        public boolean isMissing(int dimension, int timeIndex) {
            return Float.isNaN(series[dimension][timeIndex]);
        }

        @Override
        public double value(int dimension, int timeIndex) {
            return series[dimension][timeIndex];
        }
    }

    private static final class ObjectMatrixAccessor
            extends AbstractMatrixAccessor {
        private final Object[][] series;

        private ObjectMatrixAccessor(Object[][] series) {
            super(series.length, validateRectangular(series));
            this.series = series;
        }

        @Override
        public boolean isMissing(int dimension, int timeIndex) {
            Object value = series[dimension][timeIndex];
            return value == null
                    || value instanceof Number number
                    && Double.isNaN(number.doubleValue());
        }

        @Override
        public double value(int dimension, int timeIndex) {
            Object value = series[dimension][timeIndex];
            if (!(value instanceof Number number)) {
                throw new IllegalArgumentException(
                        "DTWAROW_D requires numeric observed values. Found "
                                + (value == null
                                ? "null"
                                : value.getClass().getName())
                                + " at dimension " + dimension
                                + ", time index " + timeIndex + "."
                );
            }
            return number.doubleValue();
        }
    }

    private static int validateRectangular(double[][] series) {
        Objects.requireNonNull(series, "Series cannot be null.");
        if (series.length == 0) {
            return 0;
        }
        int length = Objects.requireNonNull(
                series[0],
                "Series row cannot be null."
        ).length;
        for (int dimension = 1; dimension < series.length; dimension++) {
            if (Objects.requireNonNull(
                    series[dimension],
                    "Series row cannot be null."
            ).length != length) {
                throw new IllegalArgumentException(
                        "All dimensions must have consistent time lengths."
                );
            }
        }
        return length;
    }

    private static int validateRectangular(float[][] series) {
        Objects.requireNonNull(series, "Series cannot be null.");
        if (series.length == 0) {
            return 0;
        }
        int length = Objects.requireNonNull(
                series[0],
                "Series row cannot be null."
        ).length;
        for (int dimension = 1; dimension < series.length; dimension++) {
            if (Objects.requireNonNull(
                    series[dimension],
                    "Series row cannot be null."
            ).length != length) {
                throw new IllegalArgumentException(
                        "All dimensions must have consistent time lengths."
                );
            }
        }
        return length;
    }

    private static int validateRectangular(Object[][] series) {
        Objects.requireNonNull(series, "Series cannot be null.");
        if (series.length == 0) {
            return 0;
        }
        int length = Objects.requireNonNull(
                series[0],
                "Series row cannot be null."
        ).length;
        for (int dimension = 1; dimension < series.length; dimension++) {
            if (Objects.requireNonNull(
                    series[dimension],
                    "Series row cannot be null."
            ).length != length) {
                throw new IllegalArgumentException(
                        "All dimensions must have consistent time lengths."
                );
            }
        }
        return length;
    }

    private record AccessorPair(
            MultiSeriesAccessor first,
            MultiSeriesAccessor second
    ) {
    }

    private record LocalComparison(
            boolean comparable,
            double cost
    ) {
        private static final LocalComparison INCOMPUTABLE =
                new LocalComparison(false, 0.0);
    }

    private record Preparation(
            boolean available,
            int window,
            double gamma,
            int selectedCount
    ) {
        private static final Preparation UNAVAILABLE =
                new Preparation(false, 0, Double.NaN, 0);
    }
}