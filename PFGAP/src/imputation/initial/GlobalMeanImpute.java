package imputation.initial;

import datasets.ListObjectDataset;
import imputation.util.MissingIndices;
import java.util.List;
import java.util.Objects;

/** Global per-position mean imputation for primitive numeric data. */
public class GlobalMeanImpute extends Imputer {
    @Override
    public void Impute(ListObjectDataset dataset) {
        Objects.requireNonNull(dataset, "Dataset cannot be null.");
        List<Object> data = Objects.requireNonNull(dataset.getData(), "Dataset data cannot be null.");
        MissingIndices missing = Objects.requireNonNull(dataset.getMissingIndices(), "MissingIndices cannot be null.");
        if (data.isEmpty()) return;
        checkInstances(data, missing);
        if (missing.is2D()) impute2D(data, missing); else impute1D(data, missing);
    }

    private static void impute1D(List<Object> data, MissingIndices missing) {
        boolean useFloat = data.get(0) instanceof float[];
        int features = length1D(data.get(0));
        double[] sums = new double[features];
        int[] counts = new int[features];
        for (Object series : data) {
            requireKind(series, useFloat, false);
            if (length1D(series) != features) throw new IllegalArgumentException("All observations must have equal length.");
            for (int feature = 0; feature < features; feature++) {
                double value = get1D(series, feature);
                if (!Double.isNaN(value)) { sums[feature] += value; counts[feature]++; }
            }
        }
        for (int instance = 0; instance < data.size(); instance++) {
            Object series = data.get(instance);
            for (int offset = missing.start1D(instance); offset < missing.end1D(instance); offset++) {
                int feature = missing.positionAt(offset);
                requireMissing1D(series, feature);
                double value = counts[feature] == 0 ? 0.0 : sums[feature] / counts[feature];
                set1D(series, feature, value, useFloat);
            }
        }
    }

    private static void impute2D(List<Object> data, MissingIndices missing) {
        boolean useFloat = data.get(0) instanceof float[][];
        int dimensions = dimensions(data.get(0));
        int[] lengths = lengths(data.get(0), dimensions);
        double[][] sums = new double[dimensions][];
        int[][] counts = new int[dimensions][];
        for (int d = 0; d < dimensions; d++) { sums[d] = new double[lengths[d]]; counts[d] = new int[lengths[d]]; }
        for (Object series : data) {
            requireKind(series, useFloat, true);
            requireShape(series, dimensions, lengths);
            for (int d = 0; d < dimensions; d++) for (int p = 0; p < lengths[d]; p++) {
                double value = get2D(series, d, p);
                if (!Double.isNaN(value)) { sums[d][p] += value; counts[d][p]++; }
            }
        }
        for (int instance = 0; instance < data.size(); instance++) {
            if (missing.dimensionCount(instance) != dimensions) throw new IllegalArgumentException("Metadata dimension count mismatch.");
            Object series = data.get(instance);
            for (int d = 0; d < dimensions; d++) for (int offset = missing.start2D(instance, d); offset < missing.end2D(instance, d); offset++) {
                int p = missing.positionAt(offset);
                requireMissing2D(series, d, p);
                double value = counts[d][p] == 0 ? 0.0 : sums[d][p] / counts[d][p];
                set2D(series, d, p, value, useFloat);
            }
        }
    }

    static void checkInstances(List<Object> data, MissingIndices missing) { if (missing.instanceCount() != data.size()) throw new IllegalArgumentException("Metadata instance count mismatch."); }
    static int length1D(Object s) { if (s instanceof double[] x) return x.length; if (s instanceof float[] x) return x.length; throw unsupported(s, false); }
    static int dimensions(Object s) { if (s instanceof double[][] x) return x.length; if (s instanceof float[][] x) return x.length; throw unsupported(s, true); }
    static int[] lengths(Object s, int dimensions) { int[] result = new int[dimensions]; for (int d=0; d<dimensions; d++) result[d] = s instanceof double[][] x ? Objects.requireNonNull(x[d]).length : Objects.requireNonNull(((float[][])s)[d]).length; return result; }
    static double get1D(Object s, int p) { return s instanceof double[] x ? x[p] : ((float[])s)[p]; }
    static double get2D(Object s, int d, int p) { return s instanceof double[][] x ? x[d][p] : ((float[][])s)[d][p]; }
    static void set1D(Object s, int p, double v, boolean f) { if (f) ((float[])s)[p]=(float)v; else ((double[])s)[p]=v; }
    static void set2D(Object s, int d, int p, double v, boolean f) { if (f) ((float[][])s)[d][p]=(float)v; else ((double[][])s)[d][p]=v; }
    static void requireKind(Object s, boolean f, boolean matrix) { boolean ok = matrix ? (f ? s instanceof float[][] : s instanceof double[][]) : (f ? s instanceof float[] : s instanceof double[]); if (!ok) throw new IllegalArgumentException("Mixed or unsupported numeric storage."); }
    static void requireShape(Object s, int dimensions, int[] expected) { if (dimensions(s)!=dimensions) throw new IllegalArgumentException("Dimension mismatch."); int[] actual=lengths(s,dimensions); for(int d=0;d<dimensions;d++) if(actual[d]!=expected[d]) throw new IllegalArgumentException("Shape mismatch."); }
    static void requireMissing1D(Object s,int p) { if(p<0||p>=length1D(s)||!Double.isNaN(get1D(s,p))) throw new IllegalStateException("Invalid or stale missing target."); }
    static void requireMissing2D(Object s,int d,int p) { int[] lens=lengths(s,dimensions(s)); if(d<0||d>=lens.length||p<0||p>=lens[d]||!Double.isNaN(get2D(s,d,p))) throw new IllegalStateException("Invalid or stale missing target."); }
    static IllegalArgumentException unsupported(Object s,boolean matrix) { return new IllegalArgumentException("Unsupported numeric type: "+(s==null?"null":s.getClass().getTypeName())+". Expected "+(matrix?"double[][] or float[][].":"double[] or float[].")); }
}
