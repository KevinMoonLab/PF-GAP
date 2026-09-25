package transformation;

import java.util.Objects;

/**
 * Unified derivative transformation used by derivative-based distances.
 *
 * <p>For interior positions, this class uses the Keogh-Pazzani derivative
 * estimate:</p>
 *
 * <pre>
 * derivative[i] = 0.5 * (
 *         input[i] - input[i - 1]
 *                 + 0.5 * (input[i + 1] - input[i - 1])
 * );
 * </pre>
 *
 * <p>The first and last derivatives copy their nearest interior derivative,
 * preserving the original time length. A one-point series has derivative zero.
 * For a two-point series, both derivative values equal the observed slope.</p>
 *
 * <p>Float inputs are widened during arithmetic and written to
 * {@code double[]} output so derivative-based distances retain double-precision
 * preprocessing and accumulation. The transformation is stateless and safe for
 * concurrent use.</p>
 */
public final class DerivativeTransform {

    private DerivativeTransform() {
    }

    /**
     * Allocates and returns the derivative of a double-valued series.
     *
     * @param input non-null, nonempty source series
     * @return derivative with the same length as {@code input}
     */
    public static double[] transform(double[] input) {
        requireNonempty(input);
        double[] output = new double[input.length];
        transform(input, output);
        return output;
    }

    /**
     * Allocates and returns the derivative of a float-valued series.
     *
     * @param input non-null, nonempty source series
     * @return double-precision derivative with the same length as {@code input}
     */
    public static double[] transform(float[] input) {
        requireNonempty(input);
        double[] output = new double[input.length];
        transform(input, output);
        return output;
    }

    /**
     * Writes the derivative of a double-valued series into caller-owned output.
     *
     * @param input non-null, nonempty source series
     * @param output non-null destination with exactly {@code input.length} slots
     */
    public static void transform(
            double[] input,
            double[] output
    ) {
        requireCompatible(input, output);
        int length = input.length;

        if (length == 1) {
            output[0] = 0.0;
            return;
        }

        if (length == 2) {
            double slope = input[1] - input[0];
            output[0] = slope;
            output[1] = slope;
            return;
        }

        for (int index = 1; index < length - 1; index++) {
            double previous = input[index - 1];
            output[index] = 0.5 * (
                    input[index] - previous
                            + 0.5 * (input[index + 1] - previous)
            );
        }

        output[0] = output[1];
        output[length - 1] = output[length - 2];
    }

    /**
     * Writes the derivative of a float-valued series into caller-owned
     * double-precision output.
     *
     * @param input non-null, nonempty source series
     * @param output non-null destination with exactly {@code input.length} slots
     */
    public static void transform(
            float[] input,
            double[] output
    ) {
        requireCompatible(input, output);
        int length = input.length;

        if (length == 1) {
            output[0] = 0.0;
            return;
        }

        if (length == 2) {
            double slope = (double) input[1] - (double) input[0];
            output[0] = slope;
            output[1] = slope;
            return;
        }

        for (int index = 1; index < length - 1; index++) {
            double previous = input[index - 1];
            output[index] = 0.5 * (
                    (double) input[index] - previous
                            + 0.5 * (
                            (double) input[index + 1] - previous
                    )
            );
        }

        output[0] = output[1];
        output[length - 1] = output[length - 2];
    }

    /**
     * Allocates derivatives for every channel of a double-valued matrix.
     * The input and output orientation is {@code [dimension][time]}.
     *
     * @param input non-null matrix containing non-null, nonempty channel rows
     * @return derivative matrix with the same per-channel lengths
     */
    public static double[][] transform(double[][] input) {
        requireMatrix(input);
        double[][] output = new double[input.length][];

        for (int dimension = 0; dimension < input.length; dimension++) {
            output[dimension] = transform(input[dimension]);
        }

        return output;
    }

    /**
     * Allocates double-precision derivatives for every channel of a
     * float-valued matrix. The input orientation is
     * {@code [dimension][time]}.
     *
     * @param input non-null matrix containing non-null, nonempty channel rows
     * @return double derivative matrix with the same per-channel lengths
     */
    public static double[][] transform(float[][] input) {
        requireMatrix(input);
        double[][] output = new double[input.length][];

        for (int dimension = 0; dimension < input.length; dimension++) {
            output[dimension] = transform(input[dimension]);
        }

        return output;
    }

    /**
     * Writes derivatives for selected double-valued channels into compact
     * caller-owned output. Output row {@code p} corresponds to
     * {@code input[selectedDimensions[p]]}.
     */
    public static void transformSelected(
            double[][] input,
            int[] selectedDimensions,
            double[][] output
    ) {
        requireSelectedOutput(selectedDimensions, output);

        for (int position = 0;
             position < selectedDimensions.length;
             position++) {
            double[] source = input[selectedDimensions[position]];
            double[] destination = output[position];
            transform(source, destination);
        }
    }

    /**
     * Writes derivatives for selected float-valued channels into compact
     * double-precision caller-owned output. Output row {@code p} corresponds
     * to {@code input[selectedDimensions[p]]}.
     */
    public static void transformSelected(
            float[][] input,
            int[] selectedDimensions,
            double[][] output
    ) {
        requireSelectedOutput(selectedDimensions, output);

        for (int position = 0;
             position < selectedDimensions.length;
             position++) {
            float[] source = input[selectedDimensions[position]];
            double[] destination = output[position];
            transform(source, destination);
        }
    }

    /**
     * Allocates a compact derivative matrix containing only selected channels.
     */
    public static double[][] transformSelected(
            double[][] input,
            int[] selectedDimensions
    ) {
        Objects.requireNonNull(input, "Input matrix cannot be null.");
        Objects.requireNonNull(
                selectedDimensions,
                "Selected dimensions cannot be null."
        );

        double[][] output = new double[selectedDimensions.length][];
        for (int position = 0;
             position < selectedDimensions.length;
             position++) {
            output[position] = transform(
                    input[selectedDimensions[position]]
            );
        }
        return output;
    }

    /**
     * Allocates a compact double derivative matrix containing only selected
     * float-valued channels.
     */
    public static double[][] transformSelected(
            float[][] input,
            int[] selectedDimensions
    ) {
        Objects.requireNonNull(input, "Input matrix cannot be null.");
        Objects.requireNonNull(
                selectedDimensions,
                "Selected dimensions cannot be null."
        );

        double[][] output = new double[selectedDimensions.length][];
        for (int position = 0;
             position < selectedDimensions.length;
             position++) {
            output[position] = transform(
                    input[selectedDimensions[position]]
            );
        }
        return output;
    }

    private static void requireCompatible(
            double[] input,
            double[] output
    ) {
        requireNonempty(input);
        Objects.requireNonNull(output, "Derivative output cannot be null.");
        if (output.length != input.length) {
            throw new IllegalArgumentException(
                    "Derivative output length must equal input length. Received "
                            + output.length
                            + " and "
                            + input.length
                            + "."
            );
        }
    }

    private static void requireCompatible(
            float[] input,
            double[] output
    ) {
        requireNonempty(input);
        Objects.requireNonNull(output, "Derivative output cannot be null.");
        if (output.length != input.length) {
            throw new IllegalArgumentException(
                    "Derivative output length must equal input length. Received "
                            + output.length
                            + " and "
                            + input.length
                            + "."
            );
        }
    }

    private static void requireNonempty(double[] input) {
        Objects.requireNonNull(input, "Derivative input cannot be null.");
        if (input.length == 0) {
            throw new IllegalArgumentException(
                    "Derivative input cannot be empty."
            );
        }
    }

    private static void requireNonempty(float[] input) {
        Objects.requireNonNull(input, "Derivative input cannot be null.");
        if (input.length == 0) {
            throw new IllegalArgumentException(
                    "Derivative input cannot be empty."
            );
        }
    }

    private static void requireMatrix(double[][] input) {
        Objects.requireNonNull(input, "Derivative input matrix cannot be null.");
        if (input.length == 0) {
            throw new IllegalArgumentException(
                    "Derivative input matrix cannot have zero dimensions."
            );
        }
        for (double[] row : input) {
            requireNonempty(row);
        }
    }

    private static void requireMatrix(float[][] input) {
        Objects.requireNonNull(input, "Derivative input matrix cannot be null.");
        if (input.length == 0) {
            throw new IllegalArgumentException(
                    "Derivative input matrix cannot have zero dimensions."
            );
        }
        for (float[] row : input) {
            requireNonempty(row);
        }
    }

    private static void requireSelectedOutput(
            int[] selectedDimensions,
            double[][] output
    ) {
        Objects.requireNonNull(
                selectedDimensions,
                "Selected dimensions cannot be null."
        );
        Objects.requireNonNull(output, "Derivative output cannot be null.");
        if (output.length != selectedDimensions.length) {
            throw new IllegalArgumentException(
                    "Selected derivative output must have one row per selected "
                            + "dimension. Received "
                            + output.length
                            + " rows for "
                            + selectedDimensions.length
                            + " selections."
            );
        }
    }
}