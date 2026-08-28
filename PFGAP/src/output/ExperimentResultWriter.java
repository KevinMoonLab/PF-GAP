package output;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;

import java.io.BufferedWriter;
import java.io.IOException;
import java.io.Writer;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.time.OffsetDateTime;
import java.time.ZoneOffset;
import java.time.format.DateTimeFormatter;
import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;

/**
 * Collects repetition-level experiment records and writes one JSON result
 * document.
 *
 * The output document contains:
 *
 *     formatVersion
 *     generatedAt
 *     dataset
 *     forestMode
 *     numRepeats
 *     results
 *     aggregateMetrics
 *
 * Example:
 *
 * {
 *   "formatVersion": 1,
 *   "generatedAt": "2026-08-23T22:15:30.123Z",
 *   "dataset": "JapaneseVowels",
 *   "forestMode": "classification",
 *   "numRepeats": 3,
 *   "results": [
 *     {
 *       "repetition": 1,
 *       "metrics": {
 *         "accuracy": 0.92
 *       }
 *     },
 *     {
 *       "repetition": 2,
 *       "metrics": {
 *         "accuracy": 0.94
 *       }
 *     }
 *   ],
 *   "aggregateMetrics": {
 *     "accuracy": {
 *       "count": 3,
 *       "mean": 0.93,
 *       "populationStandardDeviation": 0.01,
 *       "minimum": 0.92,
 *       "maximum": 0.94
 *     }
 *   }
 * }
 *
 * Only finite metric values contribute to aggregate statistics. A metric
 * whose value is NaN or infinite remains present in its repetition record
 * but is excluded from its aggregate calculation.
 *
 * The writer does not serialize ProximityForest, ProximityTree, datasets,
 * predictions, outlier-score arrays, or proximity matrices. Those large
 * artifacts are written separately and referenced by paths stored in each
 * {@link ExperimentResultRecord}.
 *
 * This class is mutable and is not thread-safe. ExperimentRunner should own
 * one instance per training or evaluation run and add records after each
 * repetition completes.
 */
public final class ExperimentResultWriter {

    /**
     * Version of the top-level experiment-results JSON format.
     */
    public static final int CURRENT_FORMAT_VERSION =
            1;

    public static final String DEFAULT_FILE_NAME =
            "experiment_results.json";

    private static final DateTimeFormatter TIMESTAMP_FORMATTER =
            DateTimeFormatter.ISO_OFFSET_DATE_TIME;

    private final List<ExperimentResultRecord> records =
            new ArrayList<>();

    /**
     * Creates an empty result collector.
     */
    public ExperimentResultWriter() {
    }

    /**
     * Creates a result collector initialized with records.
     *
     * @param records initial repetition records
     */
    public ExperimentResultWriter(
            List<ExperimentResultRecord> records
    ) {
        addAll(records);
    }

    /**
     * Adds one repetition-level result.
     *
     * Every collected record must describe the same dataset and forest mode.
     *
     * Repetition numbers must also be unique.
     *
     * @param record repetition result
     */
    public void add(
            ExperimentResultRecord record
    ) {
        Objects.requireNonNull(
                record,
                "ExperimentResultRecord cannot be null."
        );

        validateCompatibility(record);
        validateUniqueRepetition(record);

        records.add(record);
    }

    /**
     * Adds multiple repetition-level results.
     *
     * @param values result records
     */
    public void addAll(
            List<ExperimentResultRecord> values
    ) {
        if (values == null) {
            return;
        }

        for (ExperimentResultRecord record : values) {
            add(record);
        }
    }

    /**
     * Removes all collected records.
     */
    public void clear() {
        records.clear();
    }

    /**
     * Returns the number of collected repetition results.
     *
     * @return result count
     */
    public int size() {
        return records.size();
    }

    /**
     * Returns whether no results have been collected.
     *
     * @return true when empty
     */
    public boolean isEmpty() {
        return records.isEmpty();
    }

    /**
     * Returns an immutable copy of the collected records.
     *
     * @return repetition results
     */
    public List<ExperimentResultRecord> getRecords() {
        return Collections.unmodifiableList(
                new ArrayList<>(records)
        );
    }

    /**
     * Writes the collected results to the default filename inside an output
     * directory:
     *
     *     experiment_results.json
     *
     * @param outputDirectory output directory
     * @return normalized absolute output path
     * @throws IOException if writing fails
     */
    public Path writeToDirectory(
            Path outputDirectory
    ) throws IOException {
        Objects.requireNonNull(
                outputDirectory,
                "Experiment output directory cannot be null."
        );

        return write(
                outputDirectory.resolve(
                        DEFAULT_FILE_NAME
                )
        );
    }

    /**
     * Writes one JSON document containing all collected repetition results
     * and aggregate metrics.
     *
     * Any existing file is replaced.
     *
     * @param path output JSON path
     * @return normalized absolute output path
     * @throws IOException if writing fails
     */
    public Path write(
            Path path
    ) throws IOException {
        Objects.requireNonNull(
                path,
                "Experiment-result output path cannot be null."
        );

        if (records.isEmpty()) {
            throw new IllegalStateException(
                    "Cannot write experiment results because no "
                            + "repetition records have been collected."
            );
        }

        Path outputPath =
                path.toAbsolutePath()
                        .normalize();

        Path parent =
                outputPath.getParent();

        if (parent != null) {
            Files.createDirectories(parent);
        }

        ExperimentResultDocument document =
                buildDocument();

        Gson gson =
                createGson();

        try (BufferedWriter writer =
                     Files.newBufferedWriter(
                             outputPath,
                             StandardCharsets.UTF_8,
                             StandardOpenOption.CREATE,
                             StandardOpenOption.WRITE,
                             StandardOpenOption.TRUNCATE_EXISTING
                     )) {

            gson.toJson(
                    document,
                    writer
            );
        }

        return outputPath;
    }

    /**
     * Writes one record directly without requiring callers to create and
     * maintain a collector explicitly.
     *
     * @param path output JSON path
     * @param record result record
     * @return normalized absolute output path
     * @throws IOException if writing fails
     */
    public static Path writeSingle(
            Path path,
            ExperimentResultRecord record
    ) throws IOException {
        ExperimentResultWriter resultWriter =
                new ExperimentResultWriter();

        resultWriter.add(record);

        return resultWriter.write(path);
    }

    /**
     * Builds the complete JSON-facing result document.
     *
     * @return result document
     */
    private ExperimentResultDocument buildDocument() {
        ExperimentResultRecord first =
                records.get(0);

        return new ExperimentResultDocument(
                CURRENT_FORMAT_VERSION,
                OffsetDateTime.now(
                        ZoneOffset.UTC
                ).format(
                        TIMESTAMP_FORMATTER
                ),
                first.getDataset(),
                first.getForestMode(),
                records.size(),
                new ArrayList<>(records),
                calculateAggregateMetrics(),
                calculateAggregateTimings(),
                calculateAggregateForestStatistics()
        );
    }

    /**
     * Calculates aggregate statistics for named evaluation metrics.
     *
     * A metric does not have to appear in every repetition. The aggregate
     * count identifies how many finite values actually contributed.
     */
    private Map<String, AggregateNumericSummary>
    calculateAggregateMetrics() {
        return calculateAggregateValues(
                ExperimentResultRecord::getMetrics
        );
    }

    /**
     * Calculates aggregate timing statistics across repetitions.
     */
    private Map<String, AggregateNumericSummary>
    calculateAggregateTimings() {
        return calculateAggregateValues(
                ExperimentResultRecord::getTimingMilliseconds
        );
    }

    /**
     * Calculates aggregate forest statistics across repetitions.
     */
    private Map<String, AggregateNumericSummary>
    calculateAggregateForestStatistics() {
        return calculateAggregateValues(
                ExperimentResultRecord::getForestStatistics
        );
    }

    /**
     * Aggregates all recurring numeric values obtained from one record map.
     */
    private Map<String, AggregateNumericSummary>
    calculateAggregateValues(
            NumericMapExtractor extractor
    ) {
        Set<String> names =
                new LinkedHashSet<>();

        for (ExperimentResultRecord record : records) {
            Map<String, Double> values =
                    extractor.extract(record);

            if (values != null) {
                names.addAll(
                        values.keySet()
                );
            }
        }

        Map<String, AggregateNumericSummary> aggregates =
                new LinkedHashMap<>();

        for (String name : names) {
            NumericAccumulator accumulator =
                    new NumericAccumulator();

            for (ExperimentResultRecord record : records) {
                Map<String, Double> values =
                        extractor.extract(record);

                if (values == null) {
                    continue;
                }

                Double value =
                        values.get(name);

                if (value == null
                        || !Double.isFinite(value)) {

                    continue;
                }

                accumulator.add(value);
            }

            if (!accumulator.isEmpty()) {
                aggregates.put(
                        name,
                        accumulator.toSummary()
                );
            }
        }

        return Collections.unmodifiableMap(
                aggregates
        );
    }

    private void validateCompatibility(
            ExperimentResultRecord candidate
    ) {
        if (records.isEmpty()) {
            return;
        }

        ExperimentResultRecord first =
                records.get(0);

        if (!first.getDataset().equals(
                candidate.getDataset()
        )) {
            throw new IllegalArgumentException(
                    "Experiment result dataset mismatch. Existing "
                            + "records describe dataset '"
                            + first.getDataset()
                            + "', but the new record describes '"
                            + candidate.getDataset()
                            + "'."
            );
        }

        if (!first.getForestMode().equalsIgnoreCase(
                candidate.getForestMode()
        )) {
            throw new IllegalArgumentException(
                    "Experiment result forest-mode mismatch. Existing "
                            + "records use mode '"
                            + first.getForestMode()
                            + "', but the new record uses '"
                            + candidate.getForestMode()
                            + "'."
            );
        }
    }

    private void validateUniqueRepetition(
            ExperimentResultRecord candidate
    ) {
        for (ExperimentResultRecord existing : records) {
            if (existing.getRepetition()
                    == candidate.getRepetition()) {

                throw new IllegalArgumentException(
                        "An experiment result already exists for "
                                + "repetition "
                                + candidate.getRepetition()
                                + "."
                );
            }
        }
    }

    private static Gson createGson() {
        return new GsonBuilder()
                .setPrettyPrinting()
                .serializeNulls()
                .serializeSpecialFloatingPointValues()
                .create();
    }

    /**
     * Extracts one named numeric map from a repetition record.
     */
    @FunctionalInterface
    private interface NumericMapExtractor {

        Map<String, Double> extract(
                ExperimentResultRecord record
        );
    }

    /**
     * Top-level object serialized to experiment_results.json.
     */
    private static final class ExperimentResultDocument {

        private final int formatVersion;

        private final String generatedAt;

        private final String dataset;

        private final String forestMode;

        private final int numRepeats;

        private final List<ExperimentResultRecord> results;

        private final Map<String, AggregateNumericSummary>
                aggregateMetrics;

        private final Map<String, AggregateNumericSummary>
                aggregateTimingMilliseconds;

        private final Map<String, AggregateNumericSummary>
                aggregateForestStatistics;

        private ExperimentResultDocument(
                int formatVersion,
                String generatedAt,
                String dataset,
                String forestMode,
                int numRepeats,
                List<ExperimentResultRecord> results,
                Map<String, AggregateNumericSummary> aggregateMetrics,
                Map<String, AggregateNumericSummary>
                        aggregateTimingMilliseconds,
                Map<String, AggregateNumericSummary>
                        aggregateForestStatistics
        ) {
            this.formatVersion =
                    formatVersion;

            this.generatedAt =
                    generatedAt;

            this.dataset =
                    dataset;

            this.forestMode =
                    forestMode;

            this.numRepeats =
                    numRepeats;

            this.results =
                    Collections.unmodifiableList(
                            new ArrayList<>(results)
                    );

            this.aggregateMetrics =
                    aggregateMetrics;

            this.aggregateTimingMilliseconds =
                    aggregateTimingMilliseconds;

            this.aggregateForestStatistics =
                    aggregateForestStatistics;
        }
    }

    /**
     * JSON-facing numeric aggregate.
     */
    public static final class AggregateNumericSummary {

        private final long count;

        private final double mean;

        private final double populationStandardDeviation;

        private final double minimum;

        private final double maximum;

        private AggregateNumericSummary(
                long count,
                double mean,
                double populationStandardDeviation,
                double minimum,
                double maximum
        ) {
            this.count =
                    count;

            this.mean =
                    mean;

            this.populationStandardDeviation =
                    populationStandardDeviation;

            this.minimum =
                    minimum;

            this.maximum =
                    maximum;
        }

        public long getCount() {
            return count;
        }

        public double getMean() {
            return mean;
        }

        public double getPopulationStandardDeviation() {
            return populationStandardDeviation;
        }

        public double getMinimum() {
            return minimum;
        }

        public double getMaximum() {
            return maximum;
        }
    }

    /**
     * Numerically stable one-pass accumulator used for aggregate metrics.
     *
     * This is intentionally local to the output package rather than using
     * preprocessing.standardization.OnlineMoments. Experiment reporting
     * should not depend on the standardization subsystem.
     */
    private static final class NumericAccumulator {

        private long count =
                0L;

        private double mean =
                0.0;

        private double m2 =
                0.0;

        private double minimum =
                Double.POSITIVE_INFINITY;

        private double maximum =
                Double.NEGATIVE_INFINITY;

        private void add(
                double value
        ) {
            if (!Double.isFinite(value)) {
                return;
            }

            if (count == Long.MAX_VALUE) {
                throw new ArithmeticException(
                        "Experiment aggregate count overflow."
                );
            }

            count++;

            double delta =
                    value - mean;

            mean +=
                    delta / count;

            double deltaFromUpdatedMean =
                    value - mean;

            m2 +=
                    delta * deltaFromUpdatedMean;

            minimum =
                    Math.min(
                            minimum,
                            value
                    );

            maximum =
                    Math.max(
                            maximum,
                            value
                    );
        }

        private boolean isEmpty() {
            return count == 0L;
        }

        private AggregateNumericSummary toSummary() {
            if (isEmpty()) {
                throw new IllegalStateException(
                        "Cannot summarize an empty numeric accumulator."
                );
            }

            double normalizedM2 =
                    normalizeM2(m2);

            double populationVariance =
                    normalizedM2 / count;

            double populationStandardDeviation =
                    Math.sqrt(
                            populationVariance
                    );

            return new AggregateNumericSummary(
                    count,
                    mean,
                    populationStandardDeviation,
                    minimum,
                    maximum
            );
        }

        private static double normalizeM2(
                double value
        ) {
            if (value >= 0.0) {
                return value;
            }

            double tolerance =
                    32.0
                            * Math.ulp(
                            Math.max(
                                    1.0,
                                    Math.abs(value)
                            )
                    );

            if (value >= -tolerance) {
                return 0.0;
            }

            throw new ArithmeticException(
                    "Aggregate squared deviation became negative: "
                            + value
            );
        }
    }
}