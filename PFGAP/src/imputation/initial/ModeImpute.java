package imputation.initial;

import datasets.ListObjectDataset;
import imputation.util.MissingIndices;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/** Mode initial imputation for categorical Object arrays using CSR metadata. */
public class ModeImpute extends Imputer {
    @Override
    public void Impute(ListObjectDataset dataset) {
        Objects.requireNonNull(dataset,"Dataset cannot be null.");
        List<Object> data=Objects.requireNonNull(dataset.getData(),"Dataset data cannot be null.");
        MissingIndices missing=Objects.requireNonNull(dataset.getMissingIndices(),"MissingIndices cannot be null.");
        if(missing.instanceCount()!=data.size())throw new IllegalArgumentException("Missing metadata instance count mismatch.");
        for(int instance=0;instance<data.size();instance++){
            Object series=Objects.requireNonNull(data.get(instance),"Categorical series cannot be null.");
            if(missing.is2D()) impute2D(series,missing,instance);
            else if(series instanceof Object[] row) imputeRow(row,missing,missing.start1D(instance),missing.end1D(instance));
            else throw unsupported(series,false);
        }
    }
    private static void impute2D(Object series,MissingIndices missing,int instance){
        if(!(series instanceof Object[][] matrix))throw unsupported(series,true);
        int dimensions=missing.dimensionCount(instance);
        if(matrix.length!=dimensions)throw new IllegalArgumentException("Dimension count mismatch.");
        for(int d=0;d<dimensions;d++)imputeRow(Objects.requireNonNull(matrix[d],"Categorical dimension cannot be null."),missing,missing.start2D(instance,d),missing.end2D(instance,d));
    }
    private static void imputeRow(Object[] row,MissingIndices missing,int start,int end){
        Map<Object,Integer> counts=new HashMap<>();
        for(Object value:row)if(value!=null)counts.merge(value,1,Integer::sum);
        Object mode=null;int best=-1;
        for(Map.Entry<Object,Integer> entry:counts.entrySet())if(entry.getValue()>best){best=entry.getValue();mode=entry.getKey();}
        for(int offset=start;offset<end;offset++){int index=missing.positionAt(offset);if(index<0||index>=row.length)throw new IndexOutOfBoundsException("Missing position outside row.");if(row[index]!=null)throw new IllegalStateException("Recorded categorical missing position is not null.");row[index]=mode;}
    }
    private static IllegalArgumentException unsupported(Object series,boolean matrix){return new IllegalArgumentException("Unsupported categorical type: "+series.getClass().getTypeName()+". Expected "+(matrix?"Object[][].":"Object[]."));}
}
