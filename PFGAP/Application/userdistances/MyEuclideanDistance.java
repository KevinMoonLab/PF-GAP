package userdistances;

import distance.api.DistanceFunction;

public class MyEuclideanDistance implements DistanceFunction {
    @Override
    public double compute(Object t1, Object t2) {
        double[] a = (double[]) t1;
        double[] b = (double[]) t2;
        double sum = 0;
        for (int i = 0; i < a.length; i++) {
            sum += Math.pow(a[i] - b[i], 2);
        }
        return Math.sqrt(sum);
    }
}
