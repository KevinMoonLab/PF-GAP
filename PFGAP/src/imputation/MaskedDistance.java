package imputation;

import java.util.*;

public class MaskedDistance {

    private final Object distanceInstance;

    public MaskedDistance(Object distanceInstance) {
        this.distanceInstance = distanceInstance;
    }

    public double compute(Object a, Object b) {
        Object aFiltered, bFiltered;

        if (a instanceof Double[] && b instanceof Double[]) {
            aFiltered = filter1D((Double[]) a, (Double[]) b)[0];
            bFiltered = filter1D((Double[]) a, (Double[]) b)[1];
        } else if (a instanceof Double[][] && b instanceof Double[][]) {
            aFiltered = filter2D((Double[][]) a, (Double[][]) b)[0];
            bFiltered = filter2D((Double[][]) a, (Double[][]) b)[1];
        } else if (a instanceof Object[] && b instanceof Object[]) {
            aFiltered = filter1D((Object[]) a, (Object[]) b)[0];
            bFiltered = filter1D((Object[]) a, (Object[]) b)[1];
        } else if (a instanceof Object[][] && b instanceof Object[][]) {
            aFiltered = filter2D((Object[][]) a, (Object[][]) b)[0];
            bFiltered = filter2D((Object[][]) a, (Object[][]) b)[1];
        } else {
            throw new IllegalArgumentException("Unsupported input types.");
        }

        if (isEmpty(aFiltered)) return Double.POSITIVE_INFINITY;

        try {
            return (double) distanceInstance.getClass()
                    .getMethod("distance", Object.class, Object.class)
                    .invoke(distanceInstance, aFiltered, bFiltered);
        } catch (Exception e) {
            throw new RuntimeException("Error invoking distance(Object, Object)", e);
        }
    }

    private Object[] filter1D(Double[] a, Double[] b) {
        List<Double> af = new ArrayList<>();
        List<Double> bf = new ArrayList<>();
        for (int i = 0; i < a.length; i++) {
            if (a[i] != null && b[i] != null) {
                af.add(a[i]);
                bf.add(b[i]);
            }
        }
        return new Object[]{af.toArray(new Double[0]), bf.toArray(new Double[0])};
    }

    private Object[] filter1D(Object[] a, Object[] b) {
        List<Object> af = new ArrayList<>();
        List<Object> bf = new ArrayList<>();
        for (int i = 0; i < a.length; i++) {
            if (a[i] != null && b[i] != null) {
                af.add(a[i]);
                bf.add(b[i]);
            }
        }
        return new Object[]{af.toArray(new Object[0]), bf.toArray(new Object[0])};
    }

    private Object[] filter2D(Double[][] a, Double[][] b) {
        List<Double[]> af = new ArrayList<>();
        List<Double[]> bf = new ArrayList<>();
        for (int i = 0; i < Math.min(a.length, b.length); i++) {
            List<Double> rowA = new ArrayList<>();
            List<Double> rowB = new ArrayList<>();
            for (int j = 0; j < Math.min(a[i].length, b[i].length); j++) {
                if (a[i][j] != null && b[i][j] != null) {
                    rowA.add(a[i][j]);
                    rowB.add(b[i][j]);
                }
            }
            if (!rowA.isEmpty()) {
                af.add(rowA.toArray(new Double[0]));
                bf.add(rowB.toArray(new Double[0]));
            }
        }
        return new Object[]{af.toArray(new Double[0][0]), bf.toArray(new Double[0][0])};
    }

    private Object[] filter2D(Object[][] a, Object[][] b) {
        List<Object[]> af = new ArrayList<>();
        List<Object[]> bf = new ArrayList<>();
        for (int i = 0; i < Math.min(a.length, b.length); i++) {
            List<Object> rowA = new ArrayList<>();
            List<Object> rowB = new ArrayList<>();
            for (int j = 0; j < Math.min(a[i].length, b[i].length); j++) {
                if (a[i][j] != null && b[i][j] != null) {
                    rowA.add(a[i][j]);
                    rowB.add(b[i][j]);
                }
            }
            if (!rowA.isEmpty()) {
                af.add(rowA.toArray(new Object[0]));
                bf.add(rowB.toArray(new Object[0]));
            }
        }
        return new Object[]{af.toArray(new Object[0][0]), bf.toArray(new Object[0][0])};
    }

    private boolean isEmpty(Object filtered) {
        if (filtered instanceof Object[]) {
            return ((Object[]) filtered).length == 0;
        } else if (filtered instanceof Object[][]) {
            return ((Object[][]) filtered).length == 0;
        }
        return true;
    }
}