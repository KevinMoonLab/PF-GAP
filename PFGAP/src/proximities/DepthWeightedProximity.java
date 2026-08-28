package proximities;

import trees.ProximityForest;
import trees.ProximityTree;

import java.util.ArrayList;
import java.util.List;

public class DepthWeightedProximity {

    public static double compute(
            Integer i,
            Integer j,
            ProximityForest pf) {

        ProximityTree[] trees =
                pf.getTrees();

        if (trees == null ||
                trees.length == 0) {

            return 0.0;
        }

        double sum = 0.0;

        for (ProximityTree tree : trees) {

            ProximityTree.Node leafI =
                    ProximityUtils.getLeafContainingTrainIndex(
                            i,
                            tree);

            ProximityTree.Node leafJ =
                    ProximityUtils.getLeafContainingTrainIndex(
                            j,
                            tree);

            sum += depthWeightedSimilarity(
                    tree,
                    leafI,
                    leafJ);
        }

        return sum / trees.length;
    }

    public static double computeTestTrain(
            Integer testIndex,
            Integer trainIndex,
            ProximityForest pf) {

        ProximityTree[] trees =
                pf.getTrees();

        if (trees == null ||
                trees.length == 0) {

            return 0.0;
        }

        double sum = 0.0;

        for (ProximityTree tree : trees) {

            ProximityTree.Node testLeaf =
                    ProximityUtils.getLeafContainingTestIndex(
                            testIndex,
                            tree);

            ProximityTree.Node trainLeaf =
                    ProximityUtils.getLeafContainingTrainIndex(
                            trainIndex,
                            tree);

            sum += depthWeightedSimilarity(
                    tree,
                    testLeaf,
                    trainLeaf);
        }

        return sum / trees.length;
    }

    private static double depthWeightedSimilarity(
            ProximityTree tree,
            ProximityTree.Node leafA,
            ProximityTree.Node leafB) {

        if (tree == null ||
                tree.getRootNode() == null ||
                leafA == null ||
                leafB == null) {

            return 0.0;
        }

        List<ProximityTree.Node> pathA =
                getPathFromRootToNode(
                        tree.getRootNode(),
                        leafA);

        List<ProximityTree.Node> pathB =
                getPathFromRootToNode(
                        tree.getRootNode(),
                        leafB);

        if (pathA == null ||
                pathB == null ||
                pathA.isEmpty() ||
                pathB.isEmpty()) {

            return 0.0;
        }

        int commonPrefixLength =
                commonPrefixLength(
                        pathA,
                        pathB);

        if (commonPrefixLength <= 0) {
            return 0.0;
        }

        /*
         * Depth convention:
         *
         * root depth = 0
         * child depth = 1
         * leaf depth = path.size() - 1
         *
         * commonPrefixLength counts the number of shared nodes
         * from the root downward.
         *
         * Therefore:
         *
         *     commonPrefixLength / max(pathA.size(), pathB.size())
         *
         * gives:
         *
         *     same leaf -> 1.0
         *     deeper common ancestry -> larger similarity
         *     only root shared -> small positive similarity
         *     always in [0, 1]
         */
        int maxPathLength =
                Math.max(
                        pathA.size(),
                        pathB.size());

        return ((double) commonPrefixLength)
                / maxPathLength;
    }

    private static List<ProximityTree.Node> getPathFromRootToNode(
            ProximityTree.Node root,
            ProximityTree.Node target) {

        if (root == null ||
                target == null) {

            return null;
        }

        ArrayList<ProximityTree.Node> path =
                new ArrayList<>();

        boolean found =
                findPath(
                        root,
                        target,
                        path);

        if (!found) {
            return null;
        }

        return path;
    }

    private static boolean findPath(
            ProximityTree.Node current,
            ProximityTree.Node target,
            List<ProximityTree.Node> path) {

        if (current == null) {
            return false;
        }

        path.add(current);

        if (current == target) {
            return true;
        }

        ProximityTree.Node[] children =
                current.get_children();

        if (children != null) {

            for (ProximityTree.Node child : children) {

                if (findPath(
                        child,
                        target,
                        path)) {

                    return true;
                }
            }
        }

        path.remove(path.size() - 1);

        return false;
    }

    private static int commonPrefixLength(
            List<ProximityTree.Node> pathA,
            List<ProximityTree.Node> pathB) {

        int max =
                Math.min(
                        pathA.size(),
                        pathB.size());

        int count = 0;

        for (int i = 0; i < max; i++) {

            if (pathA.get(i) == pathB.get(i)) {
                count++;
            } else {
                break;
            }
        }

        return count;
    }
}