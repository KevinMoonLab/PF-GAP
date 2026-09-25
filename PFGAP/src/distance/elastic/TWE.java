package distance.elastic;

import core.contracts.ObjectDataset;

import java.io.Serial;
import java.io.Serializable;
import java.util.Arrays;
import java.util.Random;

/**
 * Time Warp Edit distance for primitive univariate series.
 *
 * <p>This implementation uses implicit unit-spaced timestamps, a virtual zero
 * value at timestamp zero, squared pointwise value costs (p = 2), stiffness
 * {@code nu}, and edit penalty {@code lambda}. It follows the standard
 * three-operation recurrence: match/substitute, delete from the first series,
 * and delete from the second series.</p>
 *
 * <p>Distance evaluation uses two rows, finite-bound cell pruning, and safe
 * row-level early abandonment. Both {@code double[]} and {@code float[]}
 * inputs are supported without whole-array conversion. TWE returns its ordinary
 * accumulated edit cost; no square root is applied.</p>
 */
public final class TWE implements Serializable {

    @Serial
    private static final long serialVersionUID = 1L;

    private static final double[] NU_PARAMETERS = {
            0.00001, 0.0001, 0.0005, 0.001, 0.005,
            0.01, 0.05, 0.1, 0.5, 1.0
    };

    private static final double[] LAMBDA_PARAMETERS = {
            0.0, 0.011111111, 0.022222222, 0.033333333,
            0.044444444, 0.055555556, 0.066666667,
            0.077777778, 0.088888889, 0.1
    };

    public TWE() {
    }

    public double distance(
            Object first,
            Object second,
            double bestSoFar,
            double nu,
            double lambda
    ) {
        validateParameters(nu, lambda);

        if (first instanceof double[] firstValues
                && second instanceof double[] secondValues) {
            return distance(
                    firstValues,
                    secondValues,
                    bestSoFar,
                    nu,
                    lambda
            );
        }
        if (first instanceof float[] firstValues
                && second instanceof float[] secondValues) {
            return distance(
                    firstValues,
                    secondValues,
                    bestSoFar,
                    nu,
                    lambda
            );
        }
        throw unsupportedPair(first, second);
    }

    private static double distance(
            double[] first,
            double[] second,
            double bestSoFar,
            double nu,
            double lambda
    ) {
        requireNonempty(first.length, second.length);
        double cutoff = normalizedCutoff(bestSoFar);

        if (first.length >= second.length) {
            return distanceKernel(first, second, cutoff, nu, lambda);
        }
        return distanceKernel(second, first, cutoff, nu, lambda);
    }

    private static double distance(
            float[] first,
            float[] second,
            double bestSoFar,
            double nu,
            double lambda
    ) {
        requireNonempty(first.length, second.length);
        double cutoff = normalizedCutoff(bestSoFar);

        if (first.length >= second.length) {
            return distanceKernel(first, second, cutoff, nu, lambda);
        }
        return distanceKernel(second, first, cutoff, nu, lambda);
    }

    private static double distanceKernel(
            double[] rowSeries,
            double[] columnSeries,
            double cutoff,
            double nu,
            double lambda
    ) {
        int rowCount = rowSeries.length;
        int columnCount = columnSeries.length;
        double[] previous = new double[columnCount + 1];
        double[] current = new double[columnCount + 1];
        Arrays.fill(previous, Double.POSITIVE_INFINITY);
        previous[0] = 0.0;

        for (int row = 1; row <= rowCount; row++) {
            current[0] = Double.POSITIVE_INFINITY;
            double rowMinimum = Double.POSITIVE_INFINITY;
            double rowValue = rowSeries[row - 1];
            double previousRowValue = row > 1
                    ? rowSeries[row - 2]
                    : 0.0;
            double rowDeletionCost = square(
                    rowValue - previousRowValue
            ) + lambda + nu;

            for (int column = 1;
                 column <= columnCount;
                 column++) {
                double columnValue = columnSeries[column - 1];
                double previousColumnValue = column > 1
                        ? columnSeries[column - 2]
                        : 0.0;

                double match = previous[column - 1]
                        + square(rowValue - columnValue)
                        + square(
                                previousRowValue
                                        - previousColumnValue
                        )
                        + nu * (
                                Math.abs(row - column)
                                        + Math.abs(
                                                (row - 1)
                                                        - (column - 1)
                                        )
                        );

                double deleteRow = previous[column]
                        + rowDeletionCost;

                double deleteColumn = current[column - 1]
                        + square(
                                columnValue
                                        - previousColumnValue
                        )
                        + lambda
                        + nu;

                double value = minimum(
                        match,
                        deleteRow,
                        deleteColumn
                );
                if (value > cutoff) {
                    value = Double.POSITIVE_INFINITY;
                }
                current[column] = value;
                if (value < rowMinimum) {
                    rowMinimum = value;
                }
            }

            if (rowMinimum == Double.POSITIVE_INFINITY) {
                return Double.POSITIVE_INFINITY;
            }

            double[] temporary = previous;
            previous = current;
            current = temporary;
        }

        return previous[columnCount];
    }

    private static double distanceKernel(
            float[] rowSeries,
            float[] columnSeries,
            double cutoff,
            double nu,
            double lambda
    ) {
        int rowCount = rowSeries.length;
        int columnCount = columnSeries.length;
        double[] previous = new double[columnCount + 1];
        double[] current = new double[columnCount + 1];
        Arrays.fill(previous, Double.POSITIVE_INFINITY);
        previous[0] = 0.0;

        for (int row = 1; row <= rowCount; row++) {
            current[0] = Double.POSITIVE_INFINITY;
            double rowMinimum = Double.POSITIVE_INFINITY;
            double rowValue = rowSeries[row - 1];
            double previousRowValue = row > 1
                    ? rowSeries[row - 2]
                    : 0.0;
            double rowDeletionCost = square(
                    rowValue - previousRowValue
            ) + lambda + nu;

            for (int column = 1;
                 column <= columnCount;
                 column++) {
                double columnValue = columnSeries[column - 1];
                double previousColumnValue = column > 1
                        ? columnSeries[column - 2]
                        : 0.0;

                double match = previous[column - 1]
                        + square(rowValue - columnValue)
                        + square(
                                previousRowValue
                                        - previousColumnValue
                        )
                        + nu * (
                                Math.abs(row - column)
                                        + Math.abs(
                                                (row - 1)
                                                        - (column - 1)
                                        )
                        );

                double deleteRow = previous[column]
                        + rowDeletionCost;

                double deleteColumn = current[column - 1]
                        + square(
                                columnValue
                                        - previousColumnValue
                        )
                        + lambda
                        + nu;

                double value = minimum(
                        match,
                        deleteRow,
                        deleteColumn
                );
                if (value > cutoff) {
                    value = Double.POSITIVE_INFINITY;
                }
                current[column] = value;
                if (value < rowMinimum) {
                    rowMinimum = value;
                }
            }

            if (rowMinimum == Double.POSITIVE_INFINITY) {
                return Double.POSITIVE_INFINITY;
            }

            double[] temporary = previous;
            previous = current;
            current = temporary;
        }

        return previous[columnCount];
    }

    private static double square(double value) {
        return value * value;
    }

    private static double minimum(
            double first,
            double second,
            double third
    ) {
        double minimum = first < second ? first : second;
        return minimum < third ? minimum : third;
    }

    private static double normalizedCutoff(double bestSoFar) {
        if (Double.isNaN(bestSoFar)) {
            throw new IllegalArgumentException(
                    "TWE bestSoFar cannot be NaN."
            );
        }
        return bestSoFar < 0.0 ? 0.0 : bestSoFar;
    }

    private static void validateParameters(
            double nu,
            double lambda
    ) {
        if (!Double.isFinite(nu) || nu < 0.0) {
            throw new IllegalArgumentException(
                    "TWE nu must be finite and nonnegative."
            );
        }
        if (!Double.isFinite(lambda) || lambda < 0.0) {
            throw new IllegalArgumentException(
                    "TWE lambda must be finite and nonnegative."
            );
        }
    }

    private static void requireNonempty(
            int firstLength,
            int secondLength
    ) {
        if (firstLength == 0 || secondLength == 0) {
            throw new IllegalArgumentException(
                    "TWE requires two nonempty time series."
            );
        }
    }

    private static IllegalArgumentException unsupportedPair(
            Object first,
            Object second
    ) {
        return new IllegalArgumentException(
                "TWE requires matching double[] or float[] inputs. Received "
                        + typeName(first) + " and " + typeName(second) + "."
        );
    }

    private static String typeName(Object value) {
        return value == null ? "null" : value.getClass().getTypeName();
    }

    public double get_random_nu(ObjectDataset dataset, Random random) {
        return NU_PARAMETERS[random.nextInt(NU_PARAMETERS.length)];
    }

    public double get_random_lambda(ObjectDataset dataset, Random random) {
        return LAMBDA_PARAMETERS[
                random.nextInt(LAMBDA_PARAMETERS.length)
        ];
    }
}
