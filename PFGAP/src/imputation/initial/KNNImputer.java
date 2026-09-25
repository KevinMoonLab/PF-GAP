package imputation.initial;

import datasets.ListObjectDataset;
import distance.DistanceRegistry;
import distance.MEASURE;
import imputation.util.MissingIndices;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * K-nearest-neighbor initial imputation using an ensemble of masked distances.
 * Numeric inputs preserve primitive float/double storage. Categorical inputs
 * remain Object arrays. Updates use Jacobi semantics: every neighbor value is
 * read from the unchanged source dataset.
 */
public class KNNImputer extends Imputer {
    private final MEASURE[] measures;
    private final int k;

    public KNNImputer(MEASURE[] measures, int k) {
        this.measures = Objects.requireNonNull(measures, "Measures cannot be null.").clone();
        if (measures.length == 0) throw new IllegalArgumentException("At least one distance measure is required.");
        for (MEASURE measure : measures) Objects.requireNonNull(measure, "Distance measure cannot be null.");
        if (k < 1) throw new IllegalArgumentException("k must be positive.");
        this.k = k;
    }

    @Override
    public void Impute(ListObjectDataset dataset) {
        Objects.requireNonNull(dataset, "Dataset cannot be null.");
        List<Object> source = Objects.requireNonNull(dataset.getData(), "Dataset data cannot be null.");
        MissingIndices missing = Objects.requireNonNull(dataset.getMissingIndices(), "MissingIndices cannot be null.");
        if (source.isEmpty()) return;
        if (missing.instanceCount() != source.size()) throw new IllegalArgumentException("Missing metadata instance count mismatch.");

        List<MaskedDistance> distances = new ArrayList<>(measures.length);
        for (MEASURE measure : measures) distances.add(DistanceRegistry.getMaskedDistance(measure));

        Object[] updated = new Object[source.size()];
        boolean twoDimensional = missing.is2D();
        for (int instance = 0; instance < source.size(); instance++) {
            Object target = Objects.requireNonNull(source.get(instance), "Series cannot be null.");
            List<Neighbor> neighbors = hasMissing(missing, instance)
                    ? findKNearest(target, source, instance, distances)
                    : List.of();
            updated[instance] = twoDimensional
                    ? impute2D(target, source, missing, instance, neighbors)
                    : impute1D(target, source, missing, instance, neighbors);
        }

        dataset.setData(new ArrayList<>(Arrays.asList(updated)));
        dataset.setMissingIndices(missing);
    }

    private static Object impute1D(
            Object target,
            List<Object> source,
            MissingIndices missing,
            int instance,
            List<Neighbor> neighbors
    ) {
        int start = missing.start1D(instance);
        int end = missing.end1D(instance);
        if (target instanceof double[] row) {
            double[] result = row.clone();
            for (int o=start;o<end;o++){int p=missing.positionAt(o);requireNumericTarget(result,p);result[p]=numericMean1D(source,neighbors,p);}
            return result;
        }
        if (target instanceof float[] row) {
            float[] result = row.clone();
            for (int o=start;o<end;o++){int p=missing.positionAt(o);requireNumericTarget(result,p);result[p]=(float)numericMean1D(source,neighbors,p);}
            return result;
        }
        if (target instanceof Object[] row) {
            Object[] result = row.clone();
            for (int o=start;o<end;o++){int p=missing.positionAt(o);requireCategoricalTarget(result,p);result[p]=categoricalMode1D(source,neighbors,p);}
            return result;
        }
        throw unsupported(target, false);
    }

    private static Object impute2D(
            Object target,
            List<Object> source,
            MissingIndices missing,
            int instance,
            List<Neighbor> neighbors
    ) {
        int dimensions = missing.dimensionCount(instance);
        if (target instanceof double[][] matrix) {
            double[][] result = copy(matrix); requireDimensions(result.length,dimensions);
            for(int d=0;d<dimensions;d++)for(int o=missing.start2D(instance,d);o<missing.end2D(instance,d);o++){int p=missing.positionAt(o);requireNumericTarget(result[d],p);result[d][p]=numericMean2D(source,neighbors,d,p);}
            return result;
        }
        if (target instanceof float[][] matrix) {
            float[][] result = copy(matrix); requireDimensions(result.length,dimensions);
            for(int d=0;d<dimensions;d++)for(int o=missing.start2D(instance,d);o<missing.end2D(instance,d);o++){int p=missing.positionAt(o);requireNumericTarget(result[d],p);result[d][p]=(float)numericMean2D(source,neighbors,d,p);}
            return result;
        }
        if (target instanceof Object[][] matrix) {
            Object[][] result = copy(matrix); requireDimensions(result.length,dimensions);
            for(int d=0;d<dimensions;d++)for(int o=missing.start2D(instance,d);o<missing.end2D(instance,d);o++){int p=missing.positionAt(o);requireCategoricalTarget(result[d],p);result[d][p]=categoricalMode2D(source,neighbors,d,p);}
            return result;
        }
        throw unsupported(target, true);
    }

    private List<Neighbor> findKNearest(Object target,List<Object> data,int targetIndex,List<MaskedDistance> distances){
        List<Neighbor> candidates=new ArrayList<>();
        for(int index=0;index<data.size();index++){
            if(index==targetIndex)continue;
            double total=0.0;
            for(MaskedDistance distance:distances){double value=distance.compute(target,data.get(index));if(!Double.isFinite(value)){total=Double.POSITIVE_INFINITY;break;}total+=value;}
            if(Double.isFinite(total))candidates.add(new Neighbor(index,total));
        }
        candidates.sort(Comparator.comparingDouble(Neighbor::distance).thenComparingInt(Neighbor::index));
        if(candidates.size()>k)return new ArrayList<>(candidates.subList(0,k));
        return candidates;
    }

    private static double numericMean1D(List<Object> data,List<Neighbor> neighbors,int position){double sum=0;int count=0;for(Neighbor n:neighbors){Object s=data.get(n.index());double v;if(s instanceof double[] x&&position<x.length)v=x[position];else if(s instanceof float[] x&&position<x.length)v=x[position];else continue;if(!Double.isNaN(v)){sum+=v;count++;}}return count==0?0.0:sum/count;}
    private static double numericMean2D(List<Object> data,List<Neighbor> neighbors,int dimension,int position){double sum=0;int count=0;for(Neighbor n:neighbors){Object s=data.get(n.index());double v;if(s instanceof double[][] x&&dimension<x.length&&position<x[dimension].length)v=x[dimension][position];else if(s instanceof float[][] x&&dimension<x.length&&position<x[dimension].length)v=x[dimension][position];else continue;if(!Double.isNaN(v)){sum+=v;count++;}}return count==0?0.0:sum/count;}
    private static Object categoricalMode1D(List<Object> data,List<Neighbor> neighbors,int position){Map<Object,Integer> counts=new HashMap<>();for(Neighbor n:neighbors){Object s=data.get(n.index());if(s instanceof Object[] x&&position<x.length&&x[position]!=null)counts.merge(x[position],1,Integer::sum);}return firstMaximum(counts);}
    private static Object categoricalMode2D(List<Object> data,List<Neighbor> neighbors,int dimension,int position){Map<Object,Integer> counts=new HashMap<>();for(Neighbor n:neighbors){Object s=data.get(n.index());if(s instanceof Object[][] x&&dimension<x.length&&position<x[dimension].length&&x[dimension][position]!=null)counts.merge(x[dimension][position],1,Integer::sum);}return firstMaximum(counts);}
    private static Object firstMaximum(Map<Object,Integer> counts){Object result=null;int best=-1;for(Map.Entry<Object,Integer> e:counts.entrySet())if(e.getValue()>best){result=e.getKey();best=e.getValue();}return result;}

    private static boolean hasMissing(MissingIndices missing,int instance){if(missing.is1D())return missing.start1D(instance)<missing.end1D(instance);for(int d=0;d<missing.dimensionCount(instance);d++)if(missing.start2D(instance,d)<missing.end2D(instance,d))return true;return false;}
    private static void requireNumericTarget(double[] row,int p){if(p<0||p>=row.length)throw new IndexOutOfBoundsException();if(!Double.isNaN(row[p]))throw new IllegalStateException("Recorded numeric target is not NaN.");}
    private static void requireNumericTarget(float[] row,int p){if(p<0||p>=row.length)throw new IndexOutOfBoundsException();if(!Float.isNaN(row[p]))throw new IllegalStateException("Recorded numeric target is not NaN.");}
    private static void requireCategoricalTarget(Object[] row,int p){if(p<0||p>=row.length)throw new IndexOutOfBoundsException();if(row[p]!=null)throw new IllegalStateException("Recorded categorical target is not null.");}
    private static void requireDimensions(int actual,int expected){if(actual!=expected)throw new IllegalArgumentException("Dimension count mismatch.");}
    private static double[][] copy(double[][] x){double[][] r=new double[x.length][];for(int i=0;i<x.length;i++)r[i]=Objects.requireNonNull(x[i]).clone();return r;}
    private static float[][] copy(float[][] x){float[][] r=new float[x.length][];for(int i=0;i<x.length;i++)r[i]=Objects.requireNonNull(x[i]).clone();return r;}
    private static Object[][] copy(Object[][] x){Object[][] r=new Object[x.length][];for(int i=0;i<x.length;i++)r[i]=Objects.requireNonNull(x[i]).clone();return r;}
    private static IllegalArgumentException unsupported(Object value,boolean matrix){return new IllegalArgumentException("Unsupported series type: "+value.getClass().getTypeName()+" for "+(matrix?"2D":"1D")+" KNN imputation.");}
    private record Neighbor(int index,double distance){}
}
