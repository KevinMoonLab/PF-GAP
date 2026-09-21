package imputation.initial;

import datasets.ListObjectDataset;
import imputation.util.MissingIndices;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;

/** Global per-position median imputation for primitive numeric data. */
public class GlobalMedianImpute extends Imputer {
    @Override
    public void Impute(ListObjectDataset dataset) {
        Objects.requireNonNull(dataset, "Dataset cannot be null.");
        List<Object> data = Objects.requireNonNull(dataset.getData(), "Dataset data cannot be null.");
        MissingIndices missing = Objects.requireNonNull(dataset.getMissingIndices(), "MissingIndices cannot be null.");
        if (data.isEmpty()) return;
        GlobalMeanImpute.checkInstances(data, missing);
        if (missing.is2D()) impute2D(data, missing); else impute1D(data, missing);
    }

    private static void impute1D(List<Object> data, MissingIndices missing) {
        boolean useFloat = data.get(0) instanceof float[];
        int features = GlobalMeanImpute.length1D(data.get(0));
        double[][] columns = new double[features][data.size()];
        int[] counts = new int[features];
        for (Object series : data) {
            GlobalMeanImpute.requireKind(series, useFloat, false);
            if (GlobalMeanImpute.length1D(series) != features) throw new IllegalArgumentException("Length mismatch.");
            for (int p=0;p<features;p++) { double v=GlobalMeanImpute.get1D(series,p); if(!Double.isNaN(v)) columns[p][counts[p]++]=v; }
        }
        double[] medians = new double[features];
        for(int p=0;p<features;p++) medians[p]=median(columns[p],counts[p]);
        for(int i=0;i<data.size();i++) for(int o=missing.start1D(i);o<missing.end1D(i);o++) { int p=missing.positionAt(o); GlobalMeanImpute.requireMissing1D(data.get(i),p); GlobalMeanImpute.set1D(data.get(i),p,medians[p],useFloat); }
    }

    private static void impute2D(List<Object> data, MissingIndices missing) {
        boolean useFloat = data.get(0) instanceof float[][];
        int dimensions=GlobalMeanImpute.dimensions(data.get(0));
        int[] lengths=GlobalMeanImpute.lengths(data.get(0),dimensions);
        double[][][] columns=new double[dimensions][][]; int[][] counts=new int[dimensions][];
        for(int d=0;d<dimensions;d++){ columns[d]=new double[lengths[d]][data.size()]; counts[d]=new int[lengths[d]]; }
        for(Object series:data){ GlobalMeanImpute.requireKind(series,useFloat,true); GlobalMeanImpute.requireShape(series,dimensions,lengths); for(int d=0;d<dimensions;d++) for(int p=0;p<lengths[d];p++){ double v=GlobalMeanImpute.get2D(series,d,p); if(!Double.isNaN(v)) columns[d][p][counts[d][p]++]=v; } }
        double[][] medians=new double[dimensions][]; for(int d=0;d<dimensions;d++){ medians[d]=new double[lengths[d]]; for(int p=0;p<lengths[d];p++) medians[d][p]=median(columns[d][p],counts[d][p]); }
        for(int i=0;i<data.size();i++){ if(missing.dimensionCount(i)!=dimensions) throw new IllegalArgumentException("Metadata dimension mismatch."); for(int d=0;d<dimensions;d++) for(int o=missing.start2D(i,d);o<missing.end2D(i,d);o++){ int p=missing.positionAt(o); GlobalMeanImpute.requireMissing2D(data.get(i),d,p); GlobalMeanImpute.set2D(data.get(i),d,p,medians[d][p],useFloat); } }
    }

    private static double median(double[] values,int count){ if(count==0)return 0.0; Arrays.sort(values,0,count); int m=count/2; return (count&1)==1?values[m]:(values[m-1]+values[m])/2.0; }
}
