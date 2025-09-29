package imputation;

import java.io.Serializable;
import java.util.List;

public class MissingIndices implements Serializable {
    public List<List<Integer>> indices1D;
    public List<List<List<Integer>>> indices2D;

    private MissingIndices() {}

    public static MissingIndices from1D(List<List<Integer>> indices) {
        MissingIndices mi = new MissingIndices();
        mi.indices1D = indices;
        return mi;
    }

    public static MissingIndices from2D(List<List<List<Integer>>> indices) {
        MissingIndices mi = new MissingIndices();
        mi.indices2D = indices;
        return mi;
    }

    public boolean is2D() {
        return indices2D != null;
    }
}

/*public class MissingIndices {
    public List<Integer> indices1D;
    public List<List<Integer>> indices2D;

    private MissingIndices() {}

    public List<?> getMissingIndices() {
        if (is2D()) {
            return this.indices2D;
        } else {
            return this.indices1D;
        }
    }

    public static MissingIndices from1D(List<Integer> indices) {
        MissingIndices mi = new MissingIndices();
        mi.indices1D = indices;
        return mi;
    }

    public static MissingIndices from2D(List<List<Integer>> indices) {
        MissingIndices mi = new MissingIndices();
        mi.indices2D = indices;
        return mi;
    }

    public boolean is2D() {
        return indices2D != null;
    }
}*/

