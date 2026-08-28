package purity.isolation;

import java.util.List;

/**
 * Isolation-style split impurity based on child-size entropy.
 *
 * This measure ignores labels and scores candidate splits according to the
 * distribution of instances among child nodes.
 *
 * Lower values are better:
 *
 *     highly unbalanced split -> low entropy -> more isolation-like
 *     balanced split          -> high entropy -> less isolation-like
 *
 * For example, for a binary split:
 *
 *     [1, n - 1]      has low entropy
 *     [n / 2, n / 2]  has high entropy
 *
 * This class includes:
 *
 *     compute(List<Object> labels)
 *
 * for compatibility with the existing static purity API, plus additional
 * methods for scoring candidate child sizes directly.
 *
 * In isolation mode, labels are ignored. Only node counts matter.
 */
public class IsolationSizeEntropy {

    private IsolationSizeEntropy() {
        // Utility class
    }

    /**
     * Compatibility method for the existing static purity API.
     *
     * For a single node, this behaves as a count-based impurity:
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
     * Computes child-size entropy from child label lists.
     *
     * Empty child lists are ignored. If all children are empty, the score is
     * positive infinity.
     *
     * @param childLabelLists labels or placeholder objects in each child node
     * @return entropy of nonempty child-size proportions
     */
    public static double computeFromChildren(
            List<List<Object>> childLabelLists) {

        if (childLabelLists == null || childLabelLists.isEmpty()) {
            return Double.POSITIVE_INFINITY;
        }

        int[] childSizes = new int[childLabelLists.size()];

        for (int i = 0; i < childLabelLists.size(); i++) {
            List<Object> childLabels = childLabelLists.get(i);
            childSizes[i] = childLabels == null ? 0 : childLabels.size();
        }

        return computeFromChildSizes(childSizes);
    }

    /**
     * Computes child-size entropy from child sizes.
     *
     * Empty child nodes are ignored. This means the entropy is computed over
     * the distribution of instances among nonempty children.
     *
     * @param childSizes sizes of child nodes
     * @return entropy of nonempty child-size proportions
     */
    public static double computeFromChildSizes(int[] childSizes) {

        if (childSizes == null || childSizes.length == 0) {
            return Double.POSITIVE_INFINITY;
        }

        int total = 0;

        for (int size : childSizes) {
            if (size > 0) {
                total += size;
            }
        }

        if (total == 0) {
            return Double.POSITIVE_INFINITY;
        }

        double entropy = 0.0;

        for (int size : childSizes) {

            if (size <= 0) {
                continue;
            }

            double p = (double) size / total;
            entropy -= p * Math.log(p);
        }

        return entropy;
    }

    /**
     * Computes normalized child-size entropy.
     *
     * The unnormalized entropy ranges from 0 to log(k), where k is the number
     * of nonempty children. This method divides by log(k), yielding a value
     * in [0, 1] when there are at least two nonempty children.
     *
     * Lower values are more isolation-like.
     *
     * @param childSizes sizes of child nodes
     * @return normalized entropy, or 0.0 for a single nonempty child
     */
    public static double computeNormalizedFromChildSizes(int[] childSizes) {

        if (childSizes == null || childSizes.length == 0) {
            return Double.POSITIVE_INFINITY;
        }

        int nonemptyChildren = 0;

        for (int size : childSizes) {
            if (size > 0) {
                nonemptyChildren++;
            }
        }

        if (nonemptyChildren == 0) {
            return Double.POSITIVE_INFINITY;
        }

        if (nonemptyChildren == 1) {
            return 0.0;
        }

        double entropy = computeFromChildSizes(childSizes);

        if (Double.isInfinite(entropy)) {
            return Double.POSITIVE_INFINITY;
        }

        return entropy / Math.log(nonemptyChildren);
    }

    /**
     * Scores a binary candidate split using entropy.
     *
     * A split with one empty child is treated as invalid and receives
     * positive infinity. This discourages degenerate splits.
     *
     * @param leftSize size of left child
     * @param rightSize size of right child
     * @return entropy of the binary child-size distribution
     */
    public static double computeBinary(int leftSize, int rightSize) {

        if (leftSize <= 0 || rightSize <= 0) {
            return Double.POSITIVE_INFINITY;
        }

        int total = leftSize + rightSize;

        double pLeft = (double) leftSize / total;
        double pRight = (double) rightSize / total;

        return -pLeft * Math.log(pLeft) - pRight * Math.log(pRight);
    }

    /**
     * Scores a binary candidate split using normalized entropy.
     *
     * For a valid binary split, the maximum entropy is log(2), so this returns
     * a value in [0, 1].
     *
     * Lower values are more isolation-like.
     *
     * @param leftSize size of left child
     * @param rightSize size of right child
     * @return normalized binary entropy
     */
    public static double computeNormalizedBinary(int leftSize, int rightSize) {

        double entropy = computeBinary(leftSize, rightSize);

        if (Double.isInfinite(entropy)) {
            return Double.POSITIVE_INFINITY;
        }

        return entropy / Math.log(2.0);
    }
}