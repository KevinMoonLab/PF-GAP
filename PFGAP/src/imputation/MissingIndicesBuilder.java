package imputation;

import core.AppContext;

import java.util.ArrayList;
import java.util.List;

public class MissingIndicesBuilder {

    public static MissingIndices buildFromDataset(List<Object> dataset) {
        if (dataset == null || dataset.isEmpty()) {
            throw new IllegalArgumentException("Dataset cannot be null or empty.");
        }

        Object first = dataset.get(0);
        if (first instanceof Object[][]) {
            return buildFrom2D(dataset);
        } else if (first instanceof Object[]) {
            return buildFrom1D(dataset);
        } else {
            throw new IllegalArgumentException("Unsupported data type: " + first.getClass());
        }
    }

    private static MissingIndices buildFrom1D(List<Object> dataset) {
        List<List<Integer>> allMissing = new ArrayList<>();

        for (Object instance : dataset) {
            Object[] row = (Object[]) instance;
            List<Integer> missing = new ArrayList<>();
            for (int i = 0; i < row.length; i++) {
                if (row[i] == null) {
                    missing.add(i);
                }
            }
            allMissing.add(missing);
        }

        return MissingIndices.from1D(allMissing);
    }

    private static MissingIndices buildFrom2D(List<Object> dataset) {
        List<List<List<Integer>>> allMissing = new ArrayList<>();

        for (Object instance : dataset) {
            Object[][] matrix = (Object[][]) instance;
            List<List<Integer>> instanceMissing = new ArrayList<>();

            for (Object[] row : matrix) {
                List<Integer> missing = new ArrayList<>();
                for (int i = 0; i < row.length; i++) {
                    if (row[i] == null) {
                        missing.add(i);
                    }
                }
                instanceMissing.add(missing);
            }

            allMissing.add(instanceMissing);
        }

        return MissingIndices.from2D(allMissing);
    }
}

/*public class MissingIndicesBuilder {

    // Default behavior: assume training data
    public static MissingIndices buildFrom(Object instance) {
        return buildFrom(instance, true);
    }

    public static MissingIndices buildFrom(Object instance, boolean isTrain) {
        if (instance == null) {
            throw new IllegalArgumentException("Instance cannot be null.");
        }

        if (instance.getClass().isArray()) {
            Class<?> componentType = instance.getClass().getComponentType();
            if (componentType != null && componentType.isArray()) {
                return buildFrom2D((Object[][]) instance, isTrain);
            } else {
                return buildFrom1D((Object[]) instance, isTrain);
            }
        } else {
            throw new IllegalArgumentException("Unsupported data type: " + instance.getClass());
        }
    }

    private static MissingIndices buildFrom1D(Object[] row, boolean isTrain) {
        List<Integer> nullIndices = new ArrayList<>();
        for (int i = 0; i < row.length; i++) {
            if (row[i] == null) {
                nullIndices.add(i);
            }
        }
        MissingIndices mi = MissingIndices.from1D(nullIndices);
        if (isTrain) {
            AppContext.missing_train_indices.add(mi);
        } else {
            AppContext.missing_test_indices.add(mi);
        }
        return mi;
    }

    private static MissingIndices buildFrom2D(Object[][] matrix, boolean isTrain) {
        List<List<Integer>> nullIndices2D = new ArrayList<>();
        for (Object[] row : matrix) {
            List<Integer> rowNulls = new ArrayList<>();
            for (int i = 0; i < row.length; i++) {
                if (row[i] == null) {
                    rowNulls.add(i);
                }
            }
            nullIndices2D.add(rowNulls);
        }
        MissingIndices mi = MissingIndices.from2D(nullIndices2D);
        if (isTrain) {
            AppContext.missing_train_indices.add(mi);
        } else {
            AppContext.missing_test_indices.add(mi);
        }
        return mi;
    }
}*/
