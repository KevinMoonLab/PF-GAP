package core;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import org.apache.commons.lang3.time.DurationFormatUtils;
import trees.ProximityForest;
import trees.ProximityTree;
import util.Statistics;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.io.Serial;
import java.io.Serializable;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/**
 * Forest-wide training and evaluation results.
 *
 * <p>The legacy {@link #Predictions} list remains available for callers that
 * request ordinary prediction output. Structured per-instance output is stored
 * separately in {@link #PredictionResults} when enhanced prediction, OOD-only,
 * or combined evaluation is requested.</p>
 *
 * <p>The forest reference is transient so JSON and Java serialization do not
 * embed the complete trained model. Tree statistics should be collated before
 * exporting a detached result.</p>
 */
public class ProximityForestResult implements Serializable {

    @Serial
    private static final long serialVersionUID = 2L;

    /** Legacy aggregate predictions in test-instance order. */
    public ArrayList<Object> Predictions;

    /**
     * Optional structured per-instance results in evaluation-instance order.
     * Empty means that enhanced evaluation has not populated this result.
     */
    public ArrayList<ForestPredictionResult> PredictionResults;

    private transient ProximityForest forest;

    public boolean results_collated = false;

    // FILLED BY FOREST CLASS
    public int forest_id = -1;
    public int majority_vote_match_count = 0;
    public long startTimeTrain = 0;
    public long endTimeTrain = 0;
    public long elapsedTimeTrain = 0;
    public long startTimeTest = 0;
    public long endTimeTest = 0;
    public long elapsedTimeTest = 0;
    public int errors = 0;
    public int correct = 0;
    public double score = 0;
    public double error_rate = 0;

    // FILLED BY STAT COLLECTOR CLASS
    public int total_num_trees = -1;
    public double mean_num_nodes_per_tree = -1;
    public double sd_num_nodes_per_tree = -1;
    public double mean_depth_per_tree = -1;
    public double sd_depth_per_tree = -1;
    public double mean_weighted_depth_per_tree = -1;
    public double sd_weighted_depth_per_tree = -1;

    public ProximityForestResult(ProximityForest forest) {
        attachForest(forest);
        this.Predictions = new ArrayList<>();
        this.PredictionResults = new ArrayList<>();
    }

    /**
     * Reattaches a runtime forest after deserialization when further topology
     * collation is required.
     */
    public final void attachForest(ProximityForest forest) {
        this.forest = Objects.requireNonNull(
                forest,
                "ProximityForestResult requires a non-null forest."
        );
        this.forest_id = forest.getForestID();
    }

    /** Replaces the legacy prediction output with a defensive list copy. */
    public void setPredictions(List<?> predictions) {
        Objects.requireNonNull(predictions, "Predictions cannot be null.");
        this.Predictions = new ArrayList<>(predictions.size());
        this.Predictions.addAll(predictions);
        validatePredictionAlignmentIfBothPopulated();
    }

    /** Replaces structured enhanced results with a defensive list copy. */
    public void setPredictionResults(
            List<ForestPredictionResult> predictionResults
    ) {
        Objects.requireNonNull(
                predictionResults,
                "Structured prediction results cannot be null."
        );
        this.PredictionResults = new ArrayList<>(predictionResults.size());
        for (int index = 0; index < predictionResults.size(); index++) {
            this.PredictionResults.add(
                    Objects.requireNonNull(
                            predictionResults.get(index),
                            "Structured prediction result cannot be null at index "
                                    + index + "."
                    )
            );
        }
        validatePredictionAlignmentIfBothPopulated();
    }

    /** Clears structured output without changing legacy predictions. */
    public void clearPredictionResults() {
        PredictionResults = new ArrayList<>();
    }

    public boolean hasPredictionResults() {
        return PredictionResults != null && !PredictionResults.isEmpty();
    }

    /**
     * Validates that structured prediction-bearing entries reproduce the legacy
     * prediction list when both representations are populated.
     */
    public void validatePredictionAlignment() {
        if (Predictions == null) {
            throw new IllegalStateException("Legacy Predictions list is null.");
        }
        if (PredictionResults == null) {
            throw new IllegalStateException(
                    "Structured PredictionResults list is null."
            );
        }
        validatePredictionAlignmentIfBothPopulated();
    }

    private void validatePredictionAlignmentIfBothPopulated() {
        if (Predictions == null || PredictionResults == null
                || Predictions.isEmpty() || PredictionResults.isEmpty()) {
            return;
        }
        if (Predictions.size() != PredictionResults.size()) {
            throw new IllegalStateException(
                    "Legacy and structured prediction counts differ: "
                            + Predictions.size() + " versus "
                            + PredictionResults.size() + "."
            );
        }
        for (int index = 0; index < Predictions.size(); index++) {
            ForestPredictionResult structured = PredictionResults.get(index);
            if (!structured.hasPrediction()) {
                throw new IllegalStateException(
                        "Structured result at index " + index
                                + " omits prediction output while legacy "
                                + "Predictions is populated."
                );
            }
            if (!Objects.equals(
                    Predictions.get(index),
                    structured.prediction()
            )) {
                throw new IllegalStateException(
                        "Legacy and structured predictions differ at index "
                                + index + "."
                );
            }
        }
    }

    public void collateResults() {
        if (results_collated) {
            return;
        }
        if (forest == null) {
            throw new IllegalStateException(
                    "Cannot collate tree statistics because the transient forest "
                            + "reference is unavailable. Collate before export or "
                            + "reattach the forest first."
            );
        }

        ProximityTree[] trees = forest.getTrees();
        total_num_trees = trees.length;

        int[] nodes = new int[total_num_trees];
        double[] depths = new double[total_num_trees];
        double[] weightedDepths = new double[total_num_trees];

        for (int index = 0; index < total_num_trees; index++) {
            TreeStatCollector treeStats =
                    trees[index].getTreeStatCollection();
            nodes[index] = treeStats.num_nodes;
            depths[index] = treeStats.depth;
            weightedDepths[index] = treeStats.weighted_depth;
        }

        mean_num_nodes_per_tree = Statistics.mean(nodes);
        sd_num_nodes_per_tree =
                Statistics.standard_deviation_population(nodes);
        mean_depth_per_tree = Statistics.mean(depths);
        sd_depth_per_tree =
                Statistics.standard_deviation_population(depths);
        mean_weighted_depth_per_tree = Statistics.mean(weightedDepths);
        sd_weighted_depth_per_tree =
                Statistics.standard_deviation_population(weightedDepths);

        results_collated = true;
    }

    private void updateDerivedMetrics() {
        if (AppContext.isClassificationMode()) {
            int total = correct + errors;
            if (total > 0) {
                score = (double) correct / total;
                error_rate = (double) errors / total;
            } else {
                score = Double.NaN;
                error_rate = Double.NaN;
            }
        }
    }

    public void printResults(
            String datasetName,
            int experiment_id,
            String prefix
    ) {
        updateDerivedMetrics();

        if (AppContext.verbosity > 0) {
            String duration = DurationFormatUtils.formatDuration(
                    (long) (elapsedTimeTrain / 1e6),
                    "H:m:s.SSS"
            );
            System.out.format(
                    "%sTraining Time: %fms (%s)%n",
                    prefix,
                    elapsedTimeTrain / 1e6,
                    duration
            );

            duration = DurationFormatUtils.formatDuration(
                    (long) (elapsedTimeTest / 1e6),
                    "H:m:s.SSS"
            );
            System.out.format(
                    "%sPrediction Time: %fms (%s)%n",
                    prefix,
                    elapsedTimeTest / 1e6,
                    duration
            );
            System.out.format(
                    "%sCorrect(TP+TN): %d vs Incorrect(FP+FN): %d%n",
                    prefix,
                    correct,
                    errors
            );
            System.out.println(prefix + "Score: " + score);
            System.out.println(prefix + "Error Rate: " + error_rate);
        }

        collateResults();

        String pre = "REPEAT:" + (experiment_id + 1) + " ,";
        System.out.print(pre + datasetName);
        System.out.print(", " + score);
        System.out.print(", " + elapsedTimeTrain / 1e6);
        System.out.print(", " + elapsedTimeTest / 1e6);
        System.out.print(", " + mean_depth_per_tree);
        System.out.println();
    }

    public String exportJSON(
            String datasetName,
            int experiment_id
    ) throws Exception {
        if (!results_collated && forest != null) {
            collateResults();
        }
        validatePredictionAlignmentIfBothPopulated();

        String timestamp = LocalDateTime.now().format(
                DateTimeFormatter.ofPattern("yyyyMMdd_HHmmss_SSS")
        );
        String file = AppContext.output_dir
                + File.separator
                + forest_id
                + timestamp;

        File fileObject = new File(file);
        File parent = fileObject.getParentFile();
        if (parent != null && !parent.exists() && !parent.mkdirs()) {
            throw new IOException(
                    "Unable to create result directory: " + parent
            );
        }

        Gson gson = new GsonBuilder()
                .serializeSpecialFloatingPointValues()
                .serializeNulls()
                .create();

        try (BufferedWriter writer =
                     new BufferedWriter(new FileWriter(fileObject))) {
            writer.write(gson.toJson(this));
        }

        return file;
    }
}
