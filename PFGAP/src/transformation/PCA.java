package transformation;

import java.util.Arrays;

public class PCA {

    // Simple PCA: reduce to `numComponents` using covariance matrix and eigen decomposition
    public static double[][] transform(double[][] data, int numComponents) {
        // Placeholder: implement PCA properly or integrate with a library
        // For now, return first dimension as a mock projection
        double[][] projected = new double[numComponents][data[0].length];
        for (int i = 0; i < data[0].length; i++) {
            projected[0][i] = data[0][i]; // mock: just use first dimension
        }
        return projected;
    }
}