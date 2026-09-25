package distance.elastic;

import java.io.Serial;
import java.io.Serializable;
import java.util.Arrays;

/**
 * Shape-Based Distance (SBD) using coefficient-normalized cross-correlation.
 *
 * <p>The distance is {@code 1 - max(NCC)}, where the maximum is taken over
 * every linear shift. Inputs are z-normalized, zero-padded, and correlated by
 * a radix-2 FFT, giving O(L log L) time and O(L) auxiliary space.</p>
 *
 * <p>The {@code bestSoFar} parameter is accepted for compatibility with the
 * common distance contract, but is not used for early abandonment. The maximum
 * normalized cross-correlation is not available monotonically during the FFT.</p>
 *
 * <p>Two constant series have distance zero. A constant and a nonconstant
 * series have distance one. These finite conventions prevent undefined
 * zero-norm correlations from entering routing.</p>
 */
public final class SBD implements Serializable {

    @Serial
    private static final long serialVersionUID = 1L;

    private static final double ROUNDING_TOLERANCE = 1.0e-12;

    public SBD() {
    }

    public double distance(Object first, Object second) {
        return distance(first, second, Double.POSITIVE_INFINITY);
    }

    public double distance(
            Object first,
            Object second,
            double bestSoFar
    ) {
        if (first instanceof double[] firstValues
                && second instanceof double[] secondValues) {
            return distance(firstValues, secondValues);
        }
        if (first instanceof float[] firstValues
                && second instanceof float[] secondValues) {
            return distance(firstValues, secondValues);
        }
        throw unsupportedPair(first, second);
    }

    private static double distance(
            double[] first,
            double[] second
    ) {
        requireEqualNonemptyLengths(first.length, second.length);
        NormalizedSeries normalizedFirst = normalize(first);
        NormalizedSeries normalizedSecond = normalize(second);
        return finishDistance(normalizedFirst, normalizedSecond);
    }

    private static double distance(
            float[] first,
            float[] second
    ) {
        requireEqualNonemptyLengths(first.length, second.length);
        NormalizedSeries normalizedFirst = normalize(first);
        NormalizedSeries normalizedSecond = normalize(second);
        return finishDistance(normalizedFirst, normalizedSecond);
    }

    private static double finishDistance(
            NormalizedSeries first,
            NormalizedSeries second
    ) {
        if (first.squaredNorm() == 0.0) {
            return second.squaredNorm() == 0.0 ? 0.0 : 1.0;
        }
        if (second.squaredNorm() == 0.0) {
            return 1.0;
        }

        double denominator = Math.sqrt(
                first.squaredNorm() * second.squaredNorm()
        );
        double maximumCorrelation = maximumCrossCorrelation(
                first.values(),
                second.values()
        );
        double maximumNormalizedCorrelation =
                maximumCorrelation / denominator;

        if (maximumNormalizedCorrelation > 1.0
                && maximumNormalizedCorrelation
                <= 1.0 + ROUNDING_TOLERANCE) {
            maximumNormalizedCorrelation = 1.0;
        } else if (maximumNormalizedCorrelation < -1.0
                && maximumNormalizedCorrelation
                >= -1.0 - ROUNDING_TOLERANCE) {
            maximumNormalizedCorrelation = -1.0;
        }

        maximumNormalizedCorrelation = Math.max(
                -1.0,
                Math.min(1.0, maximumNormalizedCorrelation)
        );
        return 1.0 - maximumNormalizedCorrelation;
    }

    private static NormalizedSeries normalize(double[] input) {
        double mean = 0.0;
        for (double value : input) {
            mean += value;
        }
        mean /= input.length;

        double[] normalized = new double[input.length];
        double squaredNorm = 0.0;
        for (int index = 0; index < input.length; index++) {
            double centered = input[index] - mean;
            normalized[index] = centered;
            squaredNorm += centered * centered;
        }
        return new NormalizedSeries(normalized, squaredNorm);
    }

    private static NormalizedSeries normalize(float[] input) {
        double mean = 0.0;
        for (float value : input) {
            mean += value;
        }
        mean /= input.length;

        double[] normalized = new double[input.length];
        double squaredNorm = 0.0;
        for (int index = 0; index < input.length; index++) {
            double centered = (double) input[index] - mean;
            normalized[index] = centered;
            squaredNorm += centered * centered;
        }
        return new NormalizedSeries(normalized, squaredNorm);
    }

    private static double maximumCrossCorrelation(
            double[] first,
            double[] second
    ) {
        int requiredLength = Math.addExact(first.length, second.length) - 1;
        int fftLength = nextPowerOfTwo(requiredLength);

        double[] firstReal = Arrays.copyOf(first, fftLength);
        double[] firstImaginary = new double[fftLength];
        double[] secondReal = Arrays.copyOf(second, fftLength);
        double[] secondImaginary = new double[fftLength];

        fft(firstReal, firstImaginary, false);
        fft(secondReal, secondImaginary, false);

        for (int index = 0; index < fftLength; index++) {
            double firstRealValue = firstReal[index];
            double firstImaginaryValue = firstImaginary[index];
            double secondRealValue = secondReal[index];
            double secondImaginaryValue = secondImaginary[index];

            // first spectrum multiplied by the conjugate of the second.
            firstReal[index] = firstRealValue * secondRealValue
                    + firstImaginaryValue * secondImaginaryValue;
            firstImaginary[index] = firstImaginaryValue * secondRealValue
                    - firstRealValue * secondImaginaryValue;
        }

        fft(firstReal, firstImaginary, true);

        double maximum = -Double.MAX_VALUE;
        for (int lag = -(second.length - 1);
             lag <= first.length - 1;
             lag++) {
            int circularIndex = lag >= 0
                    ? lag
                    : fftLength + lag;
            if (firstReal[circularIndex] > maximum) {
                maximum = firstReal[circularIndex];
            }
        }
        return maximum;
    }

    private static void fft(
            double[] real,
            double[] imaginary,
            boolean inverse
    ) {
        int length = real.length;

        for (int source = 1, destination = 0;
             source < length;
             source++) {
            int bit = length >>> 1;
            while ((destination & bit) != 0) {
                destination ^= bit;
                bit >>>= 1;
            }
            destination ^= bit;
            if (source < destination) {
                double temporary = real[source];
                real[source] = real[destination];
                real[destination] = temporary;
                temporary = imaginary[source];
                imaginary[source] = imaginary[destination];
                imaginary[destination] = temporary;
            }
        }

        for (int blockLength = 2;
             blockLength <= length;
             blockLength <<= 1) {
            double angle = (inverse ? 2.0 : -2.0)
                    * Math.PI / blockLength;
            double blockCosine = Math.cos(angle);
            double blockSine = Math.sin(angle);
            int halfLength = blockLength >>> 1;

            for (int blockStart = 0;
                 blockStart < length;
                 blockStart += blockLength) {
                double cosine = 1.0;
                double sine = 0.0;

                for (int offset = 0;
                     offset < halfLength;
                     offset++) {
                    int even = blockStart + offset;
                    int odd = even + halfLength;
                    double oddReal = real[odd] * cosine
                            - imaginary[odd] * sine;
                    double oddImaginary = real[odd] * sine
                            + imaginary[odd] * cosine;

                    real[odd] = real[even] - oddReal;
                    imaginary[odd] = imaginary[even] - oddImaginary;
                    real[even] += oddReal;
                    imaginary[even] += oddImaginary;

                    double nextCosine = cosine * blockCosine
                            - sine * blockSine;
                    sine = cosine * blockSine + sine * blockCosine;
                    cosine = nextCosine;
                }
            }
        }

        if (inverse) {
            double reciprocalLength = 1.0 / length;
            for (int index = 0; index < length; index++) {
                real[index] *= reciprocalLength;
                imaginary[index] *= reciprocalLength;
            }
        }
    }

    private static int nextPowerOfTwo(int minimum) {
        if (minimum <= 1) {
            return 1;
        }
        int highest = Integer.highestOneBit(minimum - 1);
        if (highest > (1 << 29)) {
            throw new IllegalArgumentException(
                    "SBD input is too long for radix-2 FFT storage."
            );
        }
        return highest << 1;
    }

    private static void requireEqualNonemptyLengths(
            int firstLength,
            int secondLength
    ) {
        if (firstLength == 0 || secondLength == 0) {
            throw new IllegalArgumentException(
                    "SBD requires nonempty series."
            );
        }
        if (firstLength != secondLength) {
            throw new IllegalArgumentException(
                    "SBD requires equal-length series. Received "
                            + firstLength + " and " + secondLength + "."
            );
        }
    }

    private static IllegalArgumentException unsupportedPair(
            Object first,
            Object second
    ) {
        return new IllegalArgumentException(
                "SBD requires matching double[] or float[] inputs. Received "
                        + typeName(first) + " and " + typeName(second) + "."
        );
    }

    private static String typeName(Object value) {
        return value == null ? "null" : value.getClass().getTypeName();
    }

    private record NormalizedSeries(
            double[] values,
            double squaredNorm
    ) {
    }
}
