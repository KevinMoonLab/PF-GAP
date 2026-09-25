package datasets;

import core.contracts.ObjectDataset;
import datasets.readers.lazy.LazySeriesRef;
import imputation.util.MissingIndices;
import imputation.util.MissingIndicesBuilder;
import purity.classification.Entropy;
import purity.classification.Gini;
import purity.regression.MAD;
import purity.regression.Variance;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Random;
import java.util.Set;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * General-purpose dataset backed by a list of observations.
 *
 * <p>Numeric observations may be stored as {@code double[]},
 * {@code double[][]}, {@code float[]}, or {@code float[][]}. Numeric missing
 * values are represented by the corresponding NaN value. Object arrays are
 * reserved for nonnumeric data and may use {@code null} for missing values.</p>
 *
 * <p>Shallow derived datasets share their observation objects. A deep clone
 * copies supported eager array observations recursively and shares immutable
 * {@link LazySeriesRef} instances.</p>
 */
public class ListObjectDataset implements ObjectDataset, Serializable {

    private static final long serialVersionUID = 1L;

    private List<Object> data;
    private boolean is2D;
    private List<Object> labels;
    private Map<Object, Integer> classMap;
    private Map<Object, Integer> initialClassLabels;
    private ArrayList<Integer> indices;
    private MissingIndices missingIndices;
    private int length;
    private boolean isReordered;

    public ListObjectDataset() {
        this(0);
    }

    public ListObjectDataset(int expectedSize) {
        if (expectedSize < 0) {
            throw new IllegalArgumentException(
                    "expectedSize cannot be negative."
            );
        }
        this.data = new ArrayList<>(expectedSize);
        this.labels = new ArrayList<>(expectedSize);
        this.classMap = new LinkedHashMap<>();
        this.indices = new ArrayList<>(expectedSize);
    }

    @Override
    public int size() {
        return data.size();
    }

    @Override
    public int getLength() {
        return length;
    }

    @Override
    public void setLength(int len) {
        this.length = len;
    }

    @Override
    public int length() {
        return getLength();
    }

    @Override
    public List<Object> getData() {
        return data;
    }

    @Override
    @SuppressWarnings("unchecked")
    public void setData(List<?> newData) {
        Objects.requireNonNull(newData, "newData cannot be null.");
        this.data = (List<Object>) newData;
        this.missingIndices = null;
    }

    @Override
    public MissingIndices getMissingIndices() {
        return missingIndices;
    }

    @Override
    public void setMissingIndices(MissingIndices mi) {
        this.missingIndices = mi;
    }

    @Override
    public void add(Object label, Object series, Integer index) {
        data.add(series);
        labels.add(label);
        indices.add(index);
        if (label instanceof Integer || label instanceof String) {
            classMap.put(label, classMap.getOrDefault(label, 0) + 1);
        }
        missingIndices = null;
    }

    @Override
    public void remove(int i) {
        Object label = labels.get(i);
        if (classMap.containsKey(label)) {
            int updatedCount = classMap.get(label) - 1;
            if (updatedCount <= 0) {
                classMap.remove(label);
            } else {
                classMap.put(label, updatedCount);
            }
        }

        data.remove(i);
        labels.remove(i);
        indices.remove(i);
        rebuildMissingIndicesIfPresent();
    }

    @Override
    public Object get_series(int i) {
        return data.get(i);
    }

    @Override
    public Object get_class(int i) {
        return labels.get(i);
    }

    @Override
    public Integer get_index(int i) {
        return indices.get(i);
    }

    @Override
    public int get_num_classes() {
        return classMap.size();
    }

    @Override
    public int get_class_size(Object label) {
        return classMap.getOrDefault(label, 0);
    }

    @Override
    public Map<Object, Integer> get_class_map() {
        return classMap;
    }

    @Override
    public Map<Integer, Object> invertLabelMap(
            Map<Object, Integer> originalToNew
    ) {
        Map<Integer, Object> newToOriginal = new HashMap<>();
        for (Map.Entry<Object, Integer> entry : originalToNew.entrySet()) {
            newToOriginal.put(entry.getValue(), entry.getKey());
        }
        return newToOriginal;
    }

    @Override
    public void set_indices(ArrayList<Integer> indices) {
        this.indices = Objects.requireNonNull(
                indices,
                "indices cannot be null."
        );
    }

    @Override
    public Object[] get_unique_classes() {
        return classMap.keySet().toArray();
    }

    @Override
    public Set<Object> get_unique_classes_as_set() {
        return classMap.keySet();
    }

    @Override
    public Map<Object, ListObjectDataset> split_classes() {
        Map<Object, ListObjectDataset> split = new LinkedHashMap<>();
        for (int i = 0; i < size(); i++) {
            Object label = labels.get(i);
            ListObjectDataset classDataset = split.computeIfAbsent(
                    label,
                    ignored -> new ListObjectDataset(classMap.get(label))
            );
            classDataset.add(label, data.get(i), indices.get(i));
        }

        for (ListObjectDataset classDataset : split.values()) {
            copyStructuralMetadataTo(classDataset);
            if (missingIndices != null && !classDataset.data.isEmpty()) {
                classDataset.missingIndices =
                        MissingIndicesBuilder.buildFromDataset(
                                classDataset.data
                        );
            }
        }
        return split;
    }

    @Override
    public double purity(String method) {
        return switch (method.toLowerCase()) {
            case "gini" -> Gini.compute(labels);
            case "variance" -> Variance.compute(labels);
            case "entropy" -> Entropy.compute(labels);
            case "mad" -> MAD.compute(labels);
            default -> throw new IllegalArgumentException(
                    "Unknown purity method: " + method
            );
        };
    }

    @Override
    public List<Object> _internal_data_list() {
        return data;
    }

    @Override
    public List<Object> _internal_class_list() {
        return labels;
    }

    @Override
    public Object[] _internal_data_array() {
        return data.toArray();
    }

    @Override
    public ArrayList<Integer> _internal_indices_list() {
        return indices;
    }

    @Override
    public Object[] _internal_class_array() {
        return labels.toArray();
    }

    @Override
    public ListObjectDataset reorder_class_labels(
            Map<Object, Integer> newOrder
    ) {
        Map<Object, Integer> effectiveOrder =
                newOrder == null
                        ? new LinkedHashMap<>()
                        : new LinkedHashMap<>(newOrder);
        AtomicInteger nextLabel =
                new AtomicInteger(effectiveOrder.size());
        ListObjectDataset reordered = new ListObjectDataset(size());

        for (int i = 0; i < size(); i++) {
            Object oldLabel = labels.get(i);
            Integer mappedLabel = effectiveOrder.computeIfAbsent(
                    oldLabel,
                    ignored -> nextLabel.getAndIncrement()
            );
            reordered.add(mappedLabel, data.get(i), indices.get(i));
        }

        copyStructuralMetadataTo(reordered);
        reordered.initialClassLabels = effectiveOrder;
        reordered.isReordered = true;
        reordered.missingIndices = missingIndices;
        return reordered;
    }

    @Override
    public Map<Object, Integer> _get_initial_class_labels() {
        return initialClassLabels;
    }

    public void setReordered(boolean status) {
        this.isReordered = status;
    }

    public void setInitialClassOrder(
            Map<Object, Integer> initialOrder
    ) {
        this.initialClassLabels = initialOrder;
    }

    @Override
    public void shuffle() {
        shuffle(System.nanoTime());
    }

    @Override
    public void shuffle(long seed) {
        Random random = new Random(seed);
        List<Integer> order = new ArrayList<>(size());
        for (int i = 0; i < size(); i++) {
            order.add(i);
        }
        Collections.shuffle(order, random);

        List<Object> shuffledData = new ArrayList<>(size());
        List<Object> shuffledLabels = new ArrayList<>(size());
        ArrayList<Integer> shuffledIndices = new ArrayList<>(size());
        for (int position : order) {
            shuffledData.add(data.get(position));
            shuffledLabels.add(labels.get(position));
            shuffledIndices.add(indices.get(position));
        }

        data = shuffledData;
        labels = shuffledLabels;
        indices = shuffledIndices;
        rebuildMissingIndicesIfPresent();
    }

    @Override
    public ListObjectDataset shallow_clone() {
        ListObjectDataset clone = new ListObjectDataset(size());
        clone.data = new ArrayList<>(data);
        clone.labels = new ArrayList<>(labels);
        clone.indices = new ArrayList<>(indices);
        clone.classMap = new LinkedHashMap<>(classMap);
        copyStructuralMetadataTo(clone);
        clone.initialClassLabels = copyNullableMap(initialClassLabels);
        clone.isReordered = isReordered;
        clone.missingIndices = missingIndices;
        return clone;
    }

    @Override
    public ListObjectDataset deep_clone() {
        ListObjectDataset clone = new ListObjectDataset(size());
        for (int i = 0; i < size(); i++) {
            clone.add(
                    labels.get(i),
                    deepCopyObservation(data.get(i)),
                    indices.get(i)
            );
        }

        copyStructuralMetadataTo(clone);
        clone.initialClassLabels = copyNullableMap(initialClassLabels);
        clone.isReordered = isReordered;
        if (missingIndices != null && !clone.data.isEmpty()) {
            clone.missingIndices =
                    MissingIndicesBuilder.buildFromDataset(clone.data);
        }
        return clone;
    }

    @Override
    public ListObjectDataset sample_n(
            int nItems,
            Random random
    ) {
        Objects.requireNonNull(random, "random cannot be null.");
        if (nItems < 0) {
            throw new IllegalArgumentException(
                    "nItems cannot be negative."
            );
        }

        int sampleSize = Math.min(nItems, size());
        ListObjectDataset sample = new ListObjectDataset(sampleSize);
        List<Integer> order = new ArrayList<>(size());
        for (int i = 0; i < size(); i++) {
            order.add(i);
        }
        Collections.shuffle(order, random);

        for (int i = 0; i < sampleSize; i++) {
            int sourceIndex = order.get(i);
            sample.add(
                    labels.get(sourceIndex),
                    data.get(sourceIndex),
                    indices.get(sourceIndex)
            );
        }

        copyStructuralMetadataTo(sample);
        sample.initialClassLabels = copyNullableMap(initialClassLabels);
        sample.isReordered = isReordered;
        if (missingIndices != null && !sample.data.isEmpty()) {
            sample.missingIndices =
                    MissingIndicesBuilder.buildFromDataset(sample.data);
        }
        return sample;
    }

    @Override
    public boolean isNumeric(int i) {
        Object value = get_series(i);
        return value instanceof double[]
                || value instanceof double[][]
                || value instanceof float[]
                || value instanceof float[][];
    }

    @Override
    public boolean isCategorical(int i) {
        Object value = get_series(i);
        return value instanceof Object[]
                || value instanceof Object[][];
    }

    @Override
    public boolean isBoolean(int i) {
        Object value = get_series(i);
        if (value instanceof Boolean[]) {
            return true;
        }
        if (value instanceof Object[] values) {
            for (Object element : values) {
                if (element != null && !(element instanceof Boolean)) {
                    return false;
                }
            }
            return true;
        }
        return false;
    }

    @Override
    public boolean isDate(int i) {
        Object value = get_series(i);
        if (!(value instanceof Object[] values)) {
            return false;
        }
        for (Object element : values) {
            if (element != null
                    && !(element instanceof java.util.Date)
                    && !(element instanceof java.time.temporal.Temporal)) {
                return false;
            }
        }
        return true;
    }

    private void copyStructuralMetadataTo(
            ListObjectDataset target
    ) {
        target.length = length;
        target.is2D = is2D;
    }

    private void rebuildMissingIndicesIfPresent() {
        if (missingIndices == null) {
            return;
        }
        missingIndices = data.isEmpty()
                ? null
                : MissingIndicesBuilder.buildFromDataset(data);
    }

    private static Object deepCopyObservation(Object original) {
        if (original == null) {
            return null;
        }
        if (original instanceof LazySeriesRef) {
            return original;
        }
        if (original instanceof double[] values) {
            return values.clone();
        }
        if (original instanceof float[] values) {
            return values.clone();
        }
        if (original instanceof int[] values) {
            return values.clone();
        }
        if (original instanceof long[] values) {
            return values.clone();
        }
        if (original instanceof boolean[] values) {
            return values.clone();
        }
        if (original instanceof byte[] values) {
            return values.clone();
        }
        if (original instanceof short[] values) {
            return values.clone();
        }
        if (original instanceof char[] values) {
            return values.clone();
        }
        if (original instanceof double[][] values) {
            double[][] copy = new double[values.length][];
            for (int i = 0; i < values.length; i++) {
                copy[i] = values[i] == null ? null : values[i].clone();
            }
            return copy;
        }
        if (original instanceof float[][] values) {
            float[][] copy = new float[values.length][];
            for (int i = 0; i < values.length; i++) {
                copy[i] = values[i] == null ? null : values[i].clone();
            }
            return copy;
        }
        if (original instanceof Object[][] values) {
            Object[][] copy = Arrays.copyOf(values, values.length);
            for (int i = 0; i < values.length; i++) {
                copy[i] = values[i] == null
                        ? null
                        : Arrays.copyOf(values[i], values[i].length);
            }
            return copy;
        }
        if (original instanceof Object[] values) {
            return Arrays.copyOf(values, values.length);
        }
        throw new IllegalArgumentException(
                "Unsupported eager observation type for deep cloning: "
                        + original.getClass().getTypeName()
        );
    }

    private static Map<Object, Integer> copyNullableMap(
            Map<Object, Integer> source
    ) {
        return source == null
                ? null
                : new LinkedHashMap<>(source);
    }
}
