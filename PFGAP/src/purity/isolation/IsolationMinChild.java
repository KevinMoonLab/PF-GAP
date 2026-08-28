package purity.isolation;

import java.util.List;

/**
 * Isolation-style split impurity based on the smallest nonempty child size.
 *
 * This measure is designed for distance-based isolation forests implemented
 * through the existing Proximity Forest split-selection machinery.
 *
 * Lower values are better:
 *
 *     a split producing child sizes [1, n - 1] is very good;
 *     a balanced split such as [n / 2, n / 2] is less isolation-like.
 *
 * The class includes:
 *
 *     compute(List<Object> labels)
 *
 * for compatibility with the existing static purity API, and additional
 * overloads for directly scoring candidate child sizes.
 *
 * In isolation mode, labels are ignored. Only counts matter.
 */
public class IsolationMinChild {

    private IsolationMinChild() {
        // Utility class
    }

    /**
     * Compatibility method for the existing static purity API.
     *
     * For a single node, this behaves like a count-based impurity:
     * smaller nodes are considered purer.
     *
     * @param labels labels or placeholder objects for instances reaching a node
     * @return node size, or 0.0 for null/empty nodes
     */
    public static double compute(List<Object> labels) {

        if (labels == null || labels.isEmpty()) {
            return 0.0;
        }

        return labels.size();
    }

    /**
     * Scores a candidate split using child label lists.
     *
     * Empty child lists are ignored because an empty branch is typically not
     * a useful isolating split. If all children are empty, the score is
     * positive infinity.
     *
     * @param childLabelLists labels or placeholder objects in each child node
     * @return size of the smallest nonempty child
     */
    public static double computeFromChildren(
            List<List<Object>> childLabelLists) {

        if (childLabelLists == null || childLabelLists.isEmpty()) {
            return Double.POSITIVE_INFINITY;
        }

        int minNonempty = Integer.MAX_VALUE;

        for (List<Object> childLabels : childLabelLists) {

            if (childLabels == null || childLabels.isEmpty()) {
                continue;
            }

            minNonempty = Math.min(minNonempty, childLabels.size());
        }

        if (minNonempty == Integer.MAX_VALUE) {
            return Double.POSITIVE_INFINITY;
        }

        return minNonempty;
    }

    /**
     * Scores a candidate split using child sizes.
     *
     * Empty child nodes are ignored. A split producing a singleton child
     * receives score 1.0, which is highly isolation-like.
     *
     * @param childSizes sizes of child nodes
     * @return size of the smallest nonempty child
     */
    public static double computeFromChildSizes(int[] childSizes) {

        if (childSizes == null || childSizes.length == 0) {
            return Double.POSITIVE_INFINITY;
        }

        int minNonempty = Integer.MAX_VALUE;

        for (int size : childSizes) {

            if (size <= 0) {
                continue;
            }

            minNonempty = Math.min(minNonempty, size);
        }

        if (minNonempty == Integer.MAX_VALUE) {
            return Double.POSITIVE_INFINITY;
        }

        return minNonempty;
    }

    /**
     * Scores a binary candidate split.
     *
     * A split with one empty child is treated as invalid and receives
     * positive infinity. This discourages degenerate splits.
     *
     * @param leftSize size of left child
     * @param rightSize size of right child
     * @return smaller child size if both children are nonempty; otherwise infinity
     */
    public static double computeBinary(int leftSize, int rightSize) {

        if (leftSize <= 0 || rightSize <= 0) {
            return Double.POSITIVE_INFINITY;
        }

        return Math.min(leftSize, rightSize);
    }

    /**
     * Normalized version of the min-child criterion.
     *
     * This returns:
     *
     *     minNonemptyChildSize / parentSize
     *
     * so the score lies in approximately (0, 1] for valid splits.
     *
     * Lower values are better.
     *
     * @param childSizes sizes of child nodes
     * @param parentSize size of parent node
     * @return normalized smallest nonempty child size
     */
    public static double computeNormalizedFromChildSizes(
            int[] childSizes,
            int parentSize) {

        if (parentSize <= 0) {
            return Double.POSITIVE_INFINITY;
        }

        double minChild = computeFromChildSizes(childSizes);

        if (Double.isInfinite(minChild)) {
            return Double.POSITIVE_INFINITY;
        }

        return minChild / parentSize;
    }
}