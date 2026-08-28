package proximities;

import trees.ProximityForest;
import trees.ProximityTree;

public class BreimanProximity {

    public static double compute(
            Integer i,
            Integer j,
            ProximityForest pf) {

        int matches = 0;

        ProximityTree[] trees =
                pf.getTrees();

        for (ProximityTree tree : trees) {

            ProximityTree.Node leafI =
                    ProximityUtils.getLeafContainingTrainIndex(
                            i,
                            tree);

            ProximityTree.Node leafJ =
                    ProximityUtils.getLeafContainingTrainIndex(
                            j,
                            tree);

            if (leafI != null &&
                    leafI == leafJ) {

                matches++;
            }
        }

        return (double) matches /
                trees.length;
    }

    public static double computeTestTrain(
            Integer testIndex,
            Integer trainIndex,
            ProximityForest pf) {

        int matches = 0;

        ProximityTree[] trees =
                pf.getTrees();

        for (ProximityTree tree : trees) {

            ProximityTree.Node testLeaf =
                    ProximityUtils.getLeafContainingTestIndex(
                            testIndex,
                            tree);

            ProximityTree.Node trainLeaf =
                    ProximityUtils.getLeafContainingTrainIndex(
                            trainIndex,
                            tree);

            if (testLeaf != null &&
                    testLeaf == trainLeaf) {

                matches++;
            }
        }

        return (double) matches /
                trees.length;
    }
}