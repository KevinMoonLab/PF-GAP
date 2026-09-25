package transformation;

import java.util.Arrays;
import java.util.Objects;

/**
 * Principal component projection for dimension-major numeric matrices.
 *
 * <p>Input orientation is {@code [dimension][observation]}. Dimensions are
 * centered, a symmetric covariance matrix is formed, and its eigenvectors are
 * obtained with Jacobi rotations. Components are returned in descending
 * eigenvalue order with orientation {@code [component][observation]}.</p>
 *
 * <p>This implementation is intended for the modest dimensionalities used by
 * transformation-based distances. It has no external linear-algebra
 * dependency, performs all arithmetic in double precision, and never mutates
 * its input.</p>
 */
public final class PCA {

    private static final double RELATIVE_TOLERANCE = 1.0e-12;

    private PCA() {
    }

    public static double[][] transform(
            double[][] data,
            int numberOfComponents
    ) {
        Shape shape = validate(data, numberOfComponents);
        int dimensionCount = shape.dimensionCount();
        int observationCount = shape.observationCount();

        double[] means = means(data, observationCount);
        double[][] covariance = covariance(
                data,
                means,
                observationCount
        );
        EigenDecomposition decomposition = decompose(covariance);
        int[] order = descendingOrder(decomposition.eigenvalues());

        double[][] projected = new double[numberOfComponents][observationCount];
        double[][] eigenvectors = decomposition.eigenvectors();

        for (int component = 0;
             component < numberOfComponents;
             component++) {
            int eigenvectorColumn = order[component];
            canonicalizeSign(eigenvectors, eigenvectorColumn);
            double[] componentValues = projected[component];

            for (int observation = 0;
                 observation < observationCount;
                 observation++) {
                double value = 0.0;
                for (int dimension = 0;
                     dimension < dimensionCount;
                     dimension++) {
                    value += eigenvectors[dimension][eigenvectorColumn]
                            * (data[dimension][observation] - means[dimension]);
                }
                componentValues[observation] = value;
            }
        }

        return projected;
    }

    private static double[] means(
            double[][] data,
            int observationCount
    ) {
        double[] means = new double[data.length];
        double reciprocalCount = 1.0 / observationCount;

        for (int dimension = 0;
             dimension < data.length;
             dimension++) {
            double sum = 0.0;
            for (double value : data[dimension]) {
                sum += value;
            }
            means[dimension] = sum * reciprocalCount;
        }

        return means;
    }

    private static double[][] covariance(
            double[][] data,
            double[] means,
            int observationCount
    ) {
        int dimensionCount = data.length;
        double[][] covariance = new double[dimensionCount][dimensionCount];
        double denominator = observationCount > 1
                ? observationCount - 1.0
                : 1.0;

        for (int firstDimension = 0;
             firstDimension < dimensionCount;
             firstDimension++) {
            double[] first = data[firstDimension];
            double firstMean = means[firstDimension];

            for (int secondDimension = firstDimension;
                 secondDimension < dimensionCount;
                 secondDimension++) {
                double[] second = data[secondDimension];
                double secondMean = means[secondDimension];
                double sum = 0.0;

                for (int observation = 0;
                     observation < observationCount;
                     observation++) {
                    sum += (first[observation] - firstMean)
                            * (second[observation] - secondMean);
                }

                double value = sum / denominator;
                covariance[firstDimension][secondDimension] = value;
                covariance[secondDimension][firstDimension] = value;
            }
        }

        return covariance;
    }

    private static EigenDecomposition decompose(double[][] source) {
        int size = source.length;
        double[][] matrix = new double[size][];
        for (int row = 0; row < size; row++) {
            matrix[row] = source[row].clone();
        }

        double[][] eigenvectors = identity(size);
        if (size == 1) {
            return new EigenDecomposition(
                    new double[]{matrix[0][0]},
                    eigenvectors
            );
        }

        int maximumIterations = Math.max(32, 50 * size * size);
        for (int iteration = 0;
             iteration < maximumIterations;
             iteration++) {
            Pivot pivot = largestOffDiagonal(matrix);
            double scale = Math.max(
                    1.0,
                    Math.max(
                            Math.abs(matrix[pivot.first()][pivot.first()]),
                            Math.abs(matrix[pivot.second()][pivot.second()])
                    )
            );
            if (pivot.magnitude() <= RELATIVE_TOLERANCE * scale) {
                break;
            }

            applyRotation(
                    matrix,
                    eigenvectors,
                    pivot.first(),
                    pivot.second()
            );
        }

        double[] eigenvalues = new double[size];
        for (int index = 0; index < size; index++) {
            eigenvalues[index] = matrix[index][index];
        }

        return new EigenDecomposition(eigenvalues, eigenvectors);
    }

    private static void applyRotation(
            double[][] matrix,
            double[][] eigenvectors,
            int first,
            int second
    ) {
        double diagonalFirst = matrix[first][first];
        double diagonalSecond = matrix[second][second];
        double offDiagonal = matrix[first][second];
        double angle = 0.5 * Math.atan2(
                2.0 * offDiagonal,
                diagonalSecond - diagonalFirst
        );
        double cosine = Math.cos(angle);
        double sine = Math.sin(angle);

        for (int index = 0; index < matrix.length; index++) {
            if (index == first || index == second) {
                continue;
            }
            double firstValue = matrix[index][first];
            double secondValue = matrix[index][second];
            double rotatedFirst = cosine * firstValue - sine * secondValue;
            double rotatedSecond = sine * firstValue + cosine * secondValue;
            matrix[index][first] = rotatedFirst;
            matrix[first][index] = rotatedFirst;
            matrix[index][second] = rotatedSecond;
            matrix[second][index] = rotatedSecond;
        }

        double cosineSquared = cosine * cosine;
        double sineSquared = sine * sine;
        double twiceSineCosine = 2.0 * sine * cosine;
        matrix[first][first] = cosineSquared * diagonalFirst
                - twiceSineCosine * offDiagonal
                + sineSquared * diagonalSecond;
        matrix[second][second] = sineSquared * diagonalFirst
                + twiceSineCosine * offDiagonal
                + cosineSquared * diagonalSecond;
        matrix[first][second] = 0.0;
        matrix[second][first] = 0.0;

        for (int row = 0; row < eigenvectors.length; row++) {
            double firstValue = eigenvectors[row][first];
            double secondValue = eigenvectors[row][second];
            eigenvectors[row][first] = cosine * firstValue
                    - sine * secondValue;
            eigenvectors[row][second] = sine * firstValue
                    + cosine * secondValue;
        }
    }

    private static Pivot largestOffDiagonal(double[][] matrix) {
        int first = 0;
        int second = 1;
        double magnitude = Math.abs(matrix[first][second]);

        for (int row = 0; row < matrix.length - 1; row++) {
            for (int column = row + 1;
                 column < matrix.length;
                 column++) {
                double candidate = Math.abs(matrix[row][column]);
                if (candidate > magnitude) {
                    magnitude = candidate;
                    first = row;
                    second = column;
                }
            }
        }

        return new Pivot(first, second, magnitude);
    }

    private static double[][] identity(int size) {
        double[][] identity = new double[size][size];
        for (int index = 0; index < size; index++) {
            identity[index][index] = 1.0;
        }
        return identity;
    }

    private static int[] descendingOrder(double[] values) {
        Integer[] boxedOrder = new Integer[values.length];
        for (int index = 0; index < values.length; index++) {
            boxedOrder[index] = index;
        }
        Arrays.sort(
                boxedOrder,
                (first, second) -> Double.compare(
                        values[second],
                        values[first]
                )
        );

        int[] order = new int[values.length];
        for (int index = 0; index < values.length; index++) {
            order[index] = boxedOrder[index];
        }
        return order;
    }

    private static void canonicalizeSign(
            double[][] eigenvectors,
            int column
    ) {
        int largestPosition = 0;
        double largestMagnitude = Math.abs(eigenvectors[0][column]);
        for (int row = 1; row < eigenvectors.length; row++) {
            double magnitude = Math.abs(eigenvectors[row][column]);
            if (magnitude > largestMagnitude) {
                largestMagnitude = magnitude;
                largestPosition = row;
            }
        }

        if (eigenvectors[largestPosition][column] < 0.0) {
            for (double[] eigenvector : eigenvectors) {
                eigenvector[column] = -eigenvector[column];
            }
        }
    }

    private static Shape validate(
            double[][] data,
            int numberOfComponents
    ) {
        Objects.requireNonNull(data, "PCA input cannot be null.");
        if (data.length == 0) {
            throw new IllegalArgumentException(
                    "PCA input cannot have zero dimensions."
            );
        }
        if (numberOfComponents < 1
                || numberOfComponents > data.length) {
            throw new IllegalArgumentException(
                    "PCA component count must be within [1, dimensions]."
            );
        }

        int observationCount = -1;
        for (int dimension = 0;
             dimension < data.length;
             dimension++) {
            double[] row = Objects.requireNonNull(
                    data[dimension],
                    "PCA input row cannot be null at dimension "
                            + dimension
                            + "."
            );
            if (row.length == 0) {
                throw new IllegalArgumentException(
                        "PCA input rows cannot be empty."
                );
            }
            if (observationCount < 0) {
                observationCount = row.length;
            } else if (row.length != observationCount) {
                throw new IllegalArgumentException(
                        "PCA input must be rectangular."
                );
            }
            for (double value : row) {
                if (!Double.isFinite(value)) {
                    throw new IllegalArgumentException(
                            "PCA input values must be finite."
                    );
                }
            }
        }

        return new Shape(data.length, observationCount);
    }

    private record Shape(
            int dimensionCount,
            int observationCount
    ) {
    }

    private record Pivot(
            int first,
            int second,
            double magnitude
    ) {
    }

    private record EigenDecomposition(
            double[] eigenvalues,
            double[][] eigenvectors
    ) {
    }
}
