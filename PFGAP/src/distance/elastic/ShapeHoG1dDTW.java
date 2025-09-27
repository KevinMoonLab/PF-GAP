package distance.elastic;

import transformation.FirstOrderDifference;
import transformation.HistogramOfGradients;

import java.io.Serializable;

public class ShapeHoG1dDTW extends DTW{
    //double[] deriv1, deriv2;

    public ShapeHoG1dDTW() {

    }

    //public synchronized double distance(double[] series1, double[] series2, double bsf, int w) {
    public synchronized double distance(Object Series1, Object Series2, double bsf, int w) {

        double[] series1 = (double[]) Series1;
        double[] series2 = (double[]) Series2;

//		System.out.println("calling ddtw with w="+w);

        series1 = FirstOrderDifference.computeFirstOrderDifference(series1);
        series2 = FirstOrderDifference.computeFirstOrderDifference(series2);

        series1 = HistogramOfGradients.computeHistogram(series1);
        series2 = HistogramOfGradients.computeHistogram(series2);

        return super.distance(series1, series2, bsf,w);
    }


}
