package imputation.initial;

import datasets.ListObjectDataset;
import imputation.util.MissingIndices;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;

/** Median initial imputation for primitive float and double observations. */
public class MedianImpute extends Imputer {
    @Override
    public void Impute(ListObjectDataset dataset) {
        Objects.requireNonNull(dataset, "Dataset cannot be null.");
        List<Object> data = Objects.requireNonNull(dataset.getData(), "Dataset data cannot be null.");
        MissingIndices missing = Objects.requireNonNull(dataset.getMissingIndices(), "MissingIndices cannot be null.");
        if (missing.instanceCount() != data.size()) throw new IllegalArgumentException("Missing metadata instance count mismatch.");
        for (int instance = 0; instance < data.size(); instance++) {
            Object series = Objects.requireNonNull(data.get(instance), "Numeric series cannot be null.");
            if (missing.is2D()) impute2D(series, missing, instance);
            else impute1D(series, missing, missing.start1D(instance), missing.end1D(instance), instance, -1);
        }
    }
    private static void impute2D(Object series, MissingIndices missing, int instance) {
        int dimensions = missing.dimensionCount(instance);
        if (series instanceof double[][] matrix) {
            requireDimensions(matrix.length, dimensions);
            for (int d=0; d<dimensions; d++) imputeDouble(Objects.requireNonNull(matrix[d]), missing, missing.start2D(instance,d), missing.end2D(instance,d), instance,d);
        } else if (series instanceof float[][] matrix) {
            requireDimensions(matrix.length, dimensions);
            for (int d=0; d<dimensions; d++) imputeFloat(Objects.requireNonNull(matrix[d]), missing, missing.start2D(instance,d), missing.end2D(instance,d), instance,d);
        } else throw unsupported(series, true);
    }
    private static void impute1D(Object series, MissingIndices missing, int start, int end, int instance, int dimension) {
        if (series instanceof double[] row) imputeDouble(row,missing,start,end,instance,dimension);
        else if (series instanceof float[] row) imputeFloat(row,missing,start,end,instance,dimension);
        else throw unsupported(series,false);
    }
    private static void imputeDouble(double[] row, MissingIndices missing, int start, int end, int instance, int dimension) {
        double[] observed=new double[row.length]; int count=0;
        for(double value:row) if(!Double.isNaN(value)) observed[count++]=value;
        Arrays.sort(observed,0,count); double value=median(observed,count);
        for(int offset=start;offset<end;offset++){int index=missing.positionAt(offset); requireMissing(index,row.length,Double.isNaN(row[index]),instance,dimension); row[index]=value;}
    }
    private static void imputeFloat(float[] row, MissingIndices missing, int start, int end, int instance, int dimension) {
        float[] observed=new float[row.length]; int count=0;
        for(float value:row) if(!Float.isNaN(value)) observed[count++]=value;
        Arrays.sort(observed,0,count); float value=median(observed,count);
        for(int offset=start;offset<end;offset++){int index=missing.positionAt(offset); requireMissing(index,row.length,Float.isNaN(row[index]),instance,dimension); row[index]=value;}
    }
    private static double median(double[] values,int count){if(count==0)return 0.0;int m=count/2;return (count&1)==1?values[m]:(values[m-1]+values[m])/2.0;}
    private static float median(float[] values,int count){if(count==0)return 0.0f;int m=count/2;return (count&1)==1?values[m]:(float)(((double)values[m-1]+values[m])/2.0);}
    private static void requireDimensions(int actual,int expected){if(actual!=expected)throw new IllegalArgumentException("Dimension count mismatch.");}
    private static void requireMissing(int index,int length,boolean missing,int instance,int dimension){if(index<0||index>=length)throw new IndexOutOfBoundsException("Missing position outside row.");if(!missing)throw new IllegalStateException("Recorded position is not NaN at instance "+instance+(dimension<0?"":", dimension "+dimension)+".");}
    private static IllegalArgumentException unsupported(Object series,boolean matrix){return new IllegalArgumentException("Unsupported numeric type: "+series.getClass().getTypeName()+". Expected "+(matrix?"double[][] or float[][].":"double[] or float[]."));}
}
