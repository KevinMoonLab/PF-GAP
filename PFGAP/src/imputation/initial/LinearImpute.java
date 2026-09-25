package imputation.initial;

import datasets.ListObjectDataset;
import imputation.util.MissingIndices;

import java.util.List;
import java.util.Objects;

/** Linear interpolation for primitive float and double observations. */
public class LinearImpute extends Imputer {
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
            for (int d = 0; d < dimensions; d++) imputeDouble(Objects.requireNonNull(matrix[d]), missing, missing.start2D(instance,d), missing.end2D(instance,d), instance,d);
        } else if (series instanceof float[][] matrix) {
            requireDimensions(matrix.length, dimensions);
            for (int d = 0; d < dimensions; d++) imputeFloat(Objects.requireNonNull(matrix[d]), missing, missing.start2D(instance,d), missing.end2D(instance,d), instance,d);
        } else throw unsupported(series, true);
    }

    private static void impute1D(Object series, MissingIndices missing, int start, int end, int instance, int dimension) {
        if (series instanceof double[] row) imputeDouble(row, missing, start, end, instance, dimension);
        else if (series instanceof float[] row) imputeFloat(row, missing, start, end, instance, dimension);
        else throw unsupported(series, false);
    }

    private static void imputeDouble(double[] row, MissingIndices missing, int start, int end, int instance, int dimension) {
        validateTargets(row, missing, start, end, instance, dimension);
        int i = 0;
        while (i < row.length) {
            if (!Double.isNaN(row[i])) { i++; continue; }
            int first = i;
            while (i < row.length && Double.isNaN(row[i])) i++;
            int right = i;
            int left = first - 1;
            if (left >= 0 && right < row.length) {
                double leftValue = row[left];
                double increment = (row[right] - leftValue) / (right - left);
                for (int p = first; p < right; p++) row[p] = leftValue + increment * (p - left);
            } else if (left >= 0) {
                for (int p = first; p < right; p++) row[p] = row[left];
            } else if (right < row.length) {
                for (int p = first; p < right; p++) row[p] = row[right];
            } else {
                for (int p = first; p < right; p++) row[p] = 0.0;
            }
        }
    }

    private static void imputeFloat(float[] row, MissingIndices missing, int start, int end, int instance, int dimension) {
        validateTargets(row, missing, start, end, instance, dimension);
        int i = 0;
        while (i < row.length) {
            if (!Float.isNaN(row[i])) { i++; continue; }
            int first = i;
            while (i < row.length && Float.isNaN(row[i])) i++;
            int right = i;
            int left = first - 1;
            if (left >= 0 && right < row.length) {
                double leftValue = row[left];
                double increment = ((double) row[right] - leftValue) / (right - left);
                for (int p = first; p < right; p++) row[p] = (float) (leftValue + increment * (p - left));
            } else if (left >= 0) {
                for (int p = first; p < right; p++) row[p] = row[left];
            } else if (right < row.length) {
                for (int p = first; p < right; p++) row[p] = row[right];
            } else {
                for (int p = first; p < right; p++) row[p] = 0.0f;
            }
        }
    }

    private static void validateTargets(double[] row, MissingIndices missing, int start, int end, int instance, int dimension) {
        for (int o=start;o<end;o++){int p=missing.positionAt(o);requirePosition(p,row.length);if(!Double.isNaN(row[p]))throw stale(instance,dimension);}
    }
    private static void validateTargets(float[] row, MissingIndices missing, int start, int end, int instance, int dimension) {
        for (int o=start;o<end;o++){int p=missing.positionAt(o);requirePosition(p,row.length);if(!Float.isNaN(row[p]))throw stale(instance,dimension);}
    }
    private static void requirePosition(int p,int length){if(p<0||p>=length)throw new IndexOutOfBoundsException("Missing position outside row.");}
    private static void requireDimensions(int actual,int expected){if(actual!=expected)throw new IllegalArgumentException("Dimension count mismatch.");}
    private static IllegalStateException stale(int i,int d){return new IllegalStateException("Recorded missing position is not NaN at instance "+i+(d<0?"":", dimension "+d)+".");}
    private static IllegalArgumentException unsupported(Object s,boolean matrix){return new IllegalArgumentException("Unsupported numeric type: "+s.getClass().getTypeName()+". Expected "+(matrix?"double[][] or float[][].":"double[] or float[]."));}
}
