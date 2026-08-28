package purity;

import datasets.ListObjectDataset;
import purity.isolation.IsolationPathLength;
import purity.isolation.IsolationMinChild;
import purity.isolation.IsolationSizeEntropy;
import purity.isolation.IsolationCount;

public class SplitScorer {

    private SplitScorer() {
        // Utility class
    }

    public static double compute(
            String purityMeasure,
            int parentSize,
            ListObjectDataset[] splits) {

        if (purityMeasure == null) {
            throw new IllegalArgumentException("purityMeasure cannot be null.");
        }

        if (splits == null || splits.length == 0) {
            return Double.POSITIVE_INFINITY;
        }

        String measure = purityMeasure.trim().toLowerCase();

        switch (measure) {
            case "isolation_min_child":
            case "isolationminchild":
                return isolationMinChild(splits);

            case "isolation_size_entropy":
            case "isolationsizeentropy":
                return isolationSizeEntropy(splits);

            case "isolation_path_length":
            case "isolationpathlength":
                return isolationPathLength(splits);

            case "isolation_count":
            case "isolationcount":
                //return weightedNodePurity(parentSize, splits, "isolation_count");
                return isolationCount(splits);

            default:
                return weightedNodePurity(parentSize, splits, purityMeasure);
        }
    }

    private static double weightedNodePurity(
            int parentSize,
            ListObjectDataset[] splits,
            String purityMeasure) {

        if (parentSize <= 0) {
            return Double.POSITIVE_INFINITY;
        }

        double weightedPurity = 0.0;

        for (ListObjectDataset split : splits) {

            if (split == null || split.size() == 0) {
                return Double.POSITIVE_INFINITY;
            }

            weightedPurity += ((double) split.size() / parentSize)
                    * split.purity(purityMeasure);
        }

        return weightedPurity;
    }

    private static int[] childSizes(ListObjectDataset[] splits) {
        int[] sizes = new int[splits.length];

        for (int i = 0; i < splits.length; i++) {
            sizes[i] = splits[i] == null ? 0 : splits[i].size();
        }

        return sizes;
    }

    private static double isolationMinChild(ListObjectDataset[] splits) {
        return IsolationMinChild.computeFromChildSizes(childSizes(splits));
    }

    private static double isolationSizeEntropy(ListObjectDataset[] splits) {
        return IsolationSizeEntropy.computeFromChildSizes(childSizes(splits));
    }

    private static double isolationPathLength(ListObjectDataset[] splits) {
        return IsolationPathLength.computeFromChildSizes(childSizes(splits));
    }

    private static double isolationCount(ListObjectDataset[] splits) {
        return IsolationCount.computeFromChildSizes(childSizes(splits));
    }
}