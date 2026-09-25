package imputation.initial;

import datasets.ListObjectDataset;
import imputation.util.MissingIndices;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/** Global per-position mode imputation for categorical Object arrays. */
public class GlobalModeImpute extends Imputer {
    @Override
    public void Impute(ListObjectDataset dataset){
        Objects.requireNonNull(dataset,"Dataset cannot be null.");
        List<Object> data=Objects.requireNonNull(dataset.getData(),"Dataset data cannot be null.");
        MissingIndices missing=Objects.requireNonNull(dataset.getMissingIndices(),"MissingIndices cannot be null.");
        if(data.isEmpty())return;
        if(missing.instanceCount()!=data.size())throw new IllegalArgumentException("Metadata instance count mismatch.");
        if(missing.is2D())impute2D(data,missing);else impute1D(data,missing);
    }

    private static void impute1D(List<Object> data,MissingIndices missing){
        if(!(data.get(0) instanceof Object[] first))throw unsupported(data.get(0),false);
        int features=first.length;
        @SuppressWarnings("unchecked") Map<Object,Integer>[] counts=new Map[features];
        for(int p=0;p<features;p++)counts[p]=new LinkedHashMap<>();
        for(Object series:data){if(!(series instanceof Object[] row)||row.length!=features)throw unsupported(series,false);for(int p=0;p<features;p++)if(row[p]!=null)counts[p].merge(row[p],1,Integer::sum);}
        Object[] modes=new Object[features];for(int p=0;p<features;p++)modes[p]=firstMaximum(counts[p]);
        for(int i=0;i<data.size();i++){Object[] row=(Object[])data.get(i);for(int o=missing.start1D(i);o<missing.end1D(i);o++){int p=missing.positionAt(o);requireTarget(row,p);row[p]=modes[p];}}
    }

    private static void impute2D(List<Object> data,MissingIndices missing){
        if(!(data.get(0) instanceof Object[][] first))throw unsupported(data.get(0),true);
        int dimensions=first.length;int[] lengths=new int[dimensions];for(int d=0;d<dimensions;d++)lengths[d]=Objects.requireNonNull(first[d]).length;
        @SuppressWarnings("unchecked") Map<Object,Integer>[][] counts=new Map[dimensions][];
        for(int d=0;d<dimensions;d++){counts[d]=new Map[lengths[d]];for(int p=0;p<lengths[d];p++)counts[d][p]=new LinkedHashMap<>();}
        for(Object series:data){if(!(series instanceof Object[][] matrix)||matrix.length!=dimensions)throw unsupported(series,true);for(int d=0;d<dimensions;d++){if(Objects.requireNonNull(matrix[d]).length!=lengths[d])throw new IllegalArgumentException("Shape mismatch.");for(int p=0;p<lengths[d];p++)if(matrix[d][p]!=null)counts[d][p].merge(matrix[d][p],1,Integer::sum);}}
        Object[][] modes=new Object[dimensions][];for(int d=0;d<dimensions;d++){modes[d]=new Object[lengths[d]];for(int p=0;p<lengths[d];p++)modes[d][p]=firstMaximum(counts[d][p]);}
        for(int i=0;i<data.size();i++){if(missing.dimensionCount(i)!=dimensions)throw new IllegalArgumentException("Metadata dimension mismatch.");Object[][] matrix=(Object[][])data.get(i);for(int d=0;d<dimensions;d++)for(int o=missing.start2D(i,d);o<missing.end2D(i,d);o++){int p=missing.positionAt(o);requireTarget(matrix[d],p);matrix[d][p]=modes[d][p];}}
    }

    private static Object firstMaximum(Map<Object,Integer> counts){Object result=null;int maximum=-1;for(Map.Entry<Object,Integer> entry:counts.entrySet())if(entry.getValue()>maximum){result=entry.getKey();maximum=entry.getValue();}return result;}
    private static void requireTarget(Object[] row,int p){if(p<0||p>=row.length)throw new IndexOutOfBoundsException("Missing position outside row.");if(row[p]!=null)throw new IllegalStateException("Recorded categorical target is not null.");}
    private static IllegalArgumentException unsupported(Object value,boolean matrix){return new IllegalArgumentException("Unsupported categorical type: "+(value==null?"null":value.getClass().getTypeName())+". Expected "+(matrix?"Object[][].":"Object[]."));}
}
