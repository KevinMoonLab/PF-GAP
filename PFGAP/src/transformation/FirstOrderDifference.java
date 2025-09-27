package transformation;

public class FirstOrderDifference {

    // For 1D input
    public static double[] computeFirstOrderDifference(double[] input) {
        if (input == null || input.length < 2) {
            return new double[0];
        }

        double[] gradient = new double[input.length - 1];
        for (int i = 0; i < input.length - 1; i++) {
            gradient[i] = input[i + 1] - input[i];
        }

        return gradient;
    }

    // For 2D input
    public static double[][] computeFirstOrderDifference(double[][] input) {
        if (input == null || input.length == 0) {
            return new double[0][];
        }

        double[][] gradients = new double[input.length][];
        for (int i = 0; i < input.length; i++) {
            double[] row = input[i];
            gradients[i] = computeFirstOrderDifference(row); // reuse 1D method
        }

        return gradients;
    }
}
