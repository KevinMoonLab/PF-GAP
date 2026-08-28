package proximities;

import trees.ProximityTree;

import java.util.ArrayList;

public class ProximityUtils {

    public static ProximityTree.Node getLeafContainingTrainIndex(
            Integer index,
            ProximityTree tree) {

        if (tree == null || tree.getLeaves() == null) {
            return null;
        }

        for (ProximityTree.Node leaf : tree.getLeaves()) {

            if (leaf == null ||
                    leaf.getInBagIndices() == null) {
                continue;
            }

            if (leaf.getInBagIndices().contains(index)) {
                return leaf;
            }
        }

        return null;
    }

    public static ProximityTree.Node getLeafContainingTestIndex(
            Integer index,
            ProximityTree tree) {

        if (tree == null || tree.getLeaves() == null) {
            return null;
        }

        for (ProximityTree.Node leaf : tree.getLeaves()) {

            if (leaf == null ||
                    leaf.TestIndices == null) {
                continue;
            }

            if (leaf.TestIndices.contains(index)) {
                return leaf;
            }
        }

        return null;
    }

    public static boolean sameLeaf(
            Integer i,
            Integer j,
            ProximityTree tree) {

        ProximityTree.Node leafI =
                getLeafContainingTrainIndex(
                        i,
                        tree);

        ProximityTree.Node leafJ =
                getLeafContainingTrainIndex(
                        j,
                        tree);

        return leafI != null &&
                leafI == leafJ;
    }

    public static boolean sameLeafTestTrain(
            Integer testIndex,
            Integer trainIndex,
            ProximityTree tree) {

        ProximityTree.Node testLeaf =
                getLeafContainingTestIndex(
                        testIndex,
                        tree);

        ProximityTree.Node trainLeaf =
                getLeafContainingTrainIndex(
                        trainIndex,
                        tree);

        return testLeaf != null &&
                testLeaf == trainLeaf;
    }

    public static int sharedLeafSize(
            Integer i,
            Integer j,
            ProximityTree tree) {

        ProximityTree.Node leafI =
                getLeafContainingTrainIndex(
                        i,
                        tree);

        ProximityTree.Node leafJ =
                getLeafContainingTrainIndex(
                        j,
                        tree);

        if (leafI != null &&
                leafI == leafJ &&
                leafI.getInBagIndices() != null) {

            return leafI.getInBagIndices()
                    .size();
        }

        return 0;
    }

    public static int sharedLeafSizeTestTrain(
            Integer testIndex,
            Integer trainIndex,
            ProximityTree tree) {

        ProximityTree.Node testLeaf =
                getLeafContainingTestIndex(
                        testIndex,
                        tree);

        ProximityTree.Node trainLeaf =
                getLeafContainingTrainIndex(
                        trainIndex,
                        tree);

        if (testLeaf != null &&
                testLeaf == trainLeaf &&
                testLeaf.getInBagIndices() != null) {

            return testLeaf.getInBagIndices()
                    .size();
        }

        return 0;
    }

    public static ArrayList<Integer> getTrainIndicesInLeaf(
            ProximityTree.Node leaf) {

        if (leaf == null ||
                leaf.getInBagIndices() == null) {

            return new ArrayList<>();
        }

        return leaf.getInBagIndices();
    }

    public static ArrayList<Integer> getTestIndicesInLeaf(
            ProximityTree.Node leaf) {

        if (leaf == null ||
                leaf.TestIndices == null) {

            return new ArrayList<>();
        }

        return leaf.TestIndices;
    }
}