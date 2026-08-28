package purity.isolation;

import java.util.List;

/**
 * Isolation-style node impurity based only on node size.
 *
 * This purity measure is intended for distance-based isolation forests
 * implemented through the existing Proximity Forest machinery.
 *
 * Unlike supervised purity measures such as Gini or Variance, this measure
 * ignores labels entirely. The List<Object> argument is used only as a
 * convenient carrier for the number of instances reaching the node.
 *
 * Lower values are better:
 *
 *     size <= minLeafSize  -> 0.0
 *     otherwise            -> size
 *
 * Thus singleton nodes are maximally pure, encouraging the tree to isolate
 * individual observations.
 *
 * Notes:
 * - This class is compatible with the current static compute(List<Object>)
 *   style used by Gini and Variance.
 * - If the existing splitter scores candidate splits by weighted child
 *   impurity, this measure primarily provides a stopping rule. More refined
 *   split-selection behavior can be added through child-size specific purity
 *   classes such as IsolationMinChild, IsolationSizeEntropy, or
 *   IsolationPathLength.
 */
public class IsolationCount {

    private static int minLeafSize = 1;

    private IsolationCount() {
        // Utility class
    }

    /**
     * Set the node size at or below which a node is considered isolated.
     *
     * @param size minimum isolated leaf size; values less than 1 are coerced to 1
     */
    public static void setMinLeafSize(int size) {
        minLeafSize = Math.max(1, size);
    }

    /**
     * Get the current minimum isolated leaf size.
     *
     * @return minimum isolated leaf size
     */
    public static int getMinLeafSize() {
        return minLeafSize;
    }

    /**
     * Computes isolation impurity from a node's instance list.
     *
     * Labels may be null. For isolation mode, labels are ignored.
     *
     * @param labels labels or placeholder objects for instances reaching a node
     * @return 0.0 if node size is at most minLeafSize, otherwise node size
     */
    public static double compute(List<Object> labels) {

        if (labels == null || labels.isEmpty()) {
            return 0.0;
        }

        int size = labels.size();

        if (size <= minLeafSize) {
            return 0.0;
        }

        return size;
    }

    /**
     * Convenience overload when only the node size is available.
     *
     * @param nodeSize number of instances reaching the node
     * @return 0.0 if node size is at most minLeafSize, otherwise node size
     */
    public static double compute(int nodeSize) {

        if (nodeSize <= minLeafSize) {
            return 0.0;
        }

        return nodeSize;
    }

    public static double computeFromChildSizes(int[] childSizes) {

        if (childSizes == null || childSizes.length == 0) {
            return Double.POSITIVE_INFINITY;
        }

        int total = 0;

        for (int size : childSizes) {
            if (size <= 0) {
                return Double.POSITIVE_INFINITY;
            }

            total += size;
        }

        if (total <= 0) {
            return Double.POSITIVE_INFINITY;
        }

        double weightedCount = 0.0;

        for (int size : childSizes) {
            weightedCount += ((double) size / total) * compute(size);
        }

        return weightedCount;
    }
}