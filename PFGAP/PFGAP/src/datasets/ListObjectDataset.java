package datasets;

import java.io.Serializable;
import java.util.*;
import java.util.concurrent.atomic.AtomicInteger;

import core.contracts.ObjectDataset;
import imputation.MissingIndices;
import util.ObjectDataUtils; // is this needed here?

public class ListObjectDataset implements ObjectDataset, Serializable {

    //private List<Object[]> data;
    private List<Object> data;
    private boolean is2D; //when each data instance is 2d, like multivariate time series.
    private List<Integer> labels;
    private Map<Integer, Integer> classMap;
    private Map<Integer, Integer> initialClassLabels;
    private ArrayList<Integer> indices;
    private MissingIndices missingIndices;
    private int length;
    private boolean isReordered = false;

    public ListObjectDataset() {
        this.data = new ArrayList<>();
        this.labels = new ArrayList<>();
        this.classMap = new LinkedHashMap<>();
        this.indices = new ArrayList<>();
    }

    public ListObjectDataset(int expectedSize) {
        this.data = new ArrayList<>(expectedSize);
        this.labels = new ArrayList<>(expectedSize);
        this.classMap = new LinkedHashMap<>();
        this.indices = new ArrayList<>();
    }

    /*public ListObjectDataset(int expectedSize, int length) {
        this.length = length;
        this.data = new ArrayList<>(expectedSize);
        this.labels = new ArrayList<>(expectedSize);
        this.classMap = new LinkedHashMap<>();
        this.indices = new ArrayList<>();
    }*/

    @Override
    public int size() {
        return this.data.size();
    }

    @Override
    public int getLength(){
        return this.length;
    };

    @Override
    public void setLength(int len){
        this.length = len;
    }

    @Override
    public int length() { //careful when referring to this: data instances may not all have the same length.
        return getLength();
        //if (data.isEmpty()) return 0;
        //if (is2D) return ((Object[][]) data.get(0)).length;
        //else return ((Object[]) data.get(0)).length;
    }


    //public int length() {
    //    return this.data.isEmpty() ? length : this.data.get(0).length;
    //}

    @Override
    public List<Object> getData() { // this is needed for imputation.
        return this.data;
    }

    @Override
    public void setData(List<?> newData) {
        this.data = (List<Object>) newData;
    }

    @Override
    public MissingIndices getMissingIndices() { // this is needed for imputation.
        return this.missingIndices;
    }

    @Override
    public void setMissingIndices(MissingIndices mi) {
        this.missingIndices = mi;
    }

    @Override
    public void add(Integer label, Object series, Integer index) {
        this.data.add(series);
        this.labels.add(label);
        this.indices.add(index);
        classMap.put(label, classMap.getOrDefault(label, 0) + 1);
    }


    /*@Override
    public void add(Integer label, Object[] series, Integer index) {
        this.data.add(series);
        this.labels.add(label);
        this.indices.add(index);

        classMap.put(label, classMap.getOrDefault(label, 0) + 1);
    }

    @Override
    public void add(Integer label, Object[][] series, Integer index) {
        this.data.add(series);
        this.labels.add(label);
        this.indices.add(index);

        classMap.put(label, classMap.getOrDefault(label, 0) + 1);
    }*/

    @Override
    public void remove(int i) {
        Integer label = this.labels.get(i);
        classMap.put(label, classMap.get(label) - 1);
        if (classMap.get(label) <= 0) {
            classMap.remove(label);
        }

        this.data.remove(i);
        this.labels.remove(i);
        this.indices.remove(i);
    }

    @Override
    public Object get_series(int i) {
        return this.data.get(i);
    }

    @Override
    public Integer get_class(int i) {
        return this.labels.get(i);
    }

    @Override
    public Integer get_index(int i) {
        return this.indices.get(i);
    }

    @Override
    public int get_num_classes() {
        return this.classMap.size();
    }

    @Override
    public int get_class_size(Integer label) {
        return this.classMap.getOrDefault(label, 0);
    }

    @Override
    public Map<Integer, Integer> get_class_map() {
        return this.classMap;
    }

    @Override
    public Map<Integer, Integer> invertLabelMap(Map<Integer, Integer> originalToNew) {
        Map<Integer, Integer> newToOriginal = new HashMap<>();
        for (Map.Entry<Integer, Integer> entry : originalToNew.entrySet()) {
            newToOriginal.put(entry.getValue(), entry.getKey());
        }
        return newToOriginal;
    }


    @Override
    public void set_indices(ArrayList<Integer> indices) {
        this.indices = indices;
    }

    @Override
    public int[] get_unique_classes() {
        Set<Integer> keys = this.classMap.keySet();
        int[] unique = new int[keys.size()];
        int i = 0;
        for (Integer key : keys) {
            unique[i++] = key;
        }
        return unique;
    }

    @Override
    public Set<Integer> get_unique_classes_as_set() {
        return this.classMap.keySet();
    }

    @Override
    public Map<Integer, ListObjectDataset> split_classes() {
        Map<Integer, ListObjectDataset> split = new LinkedHashMap<>();
        for (int i = 0; i < this.size(); i++) {
            Integer label = this.labels.get(i);
            split.putIfAbsent(label, new ListObjectDataset(this.classMap.get(label))); //ListObjectDataset(this.classMap.get(label), this.length()));
            split.get(label).add(label, this.data.get(i), this.indices.get(i));
        }
        return split;
    }

    @Override
    public double gini() {
        double sum = 0;
        double p;
        int totalSize = this.size();
        for (Map.Entry<Integer, Integer> entry : classMap.entrySet()) {
            p = (double) entry.getValue() / totalSize;
            sum += p * p;
        }
        return 1 - sum;
    }

    @Override
    public List<Object> _internal_data_list() {
        return this.data;
    }

    @Override
    public List<Integer> _internal_class_list() {
        return this.labels;
    }

    @Override
    public Object[] _internal_data_array() {
        return null;
    }

    @Override
    public ArrayList<Integer> _internal_indices_list() {
        return this.indices;
    }

    @Override
    public int[] _internal_class_array() {
        int[] arr = new int[this.labels.size()];
        for (int i = 0; i < labels.size(); i++) {
            arr[i] = labels.get(i);
        }
        return arr;
    }

    @Override
    public ListObjectDataset reorder_class_labels(Map<Integer, Integer> newOrder) {
        ListObjectDataset newDataset = new ListObjectDataset(this.size()); //ListObjectDataset(this.size(), this.length());
        if (newOrder == null) newOrder = new HashMap<>();

        AtomicInteger newLabel = new AtomicInteger();
        for (int i = 0; i < this.size(); i++) {
            Integer oldLabel = labels.get(i);
            Integer mappedLabel = newOrder.computeIfAbsent(oldLabel, k -> newLabel.getAndIncrement());
            newDataset.add(mappedLabel, data.get(i), indices.get(i));
        }

        newDataset.setInitialClassOrder(newOrder);
        newDataset.setReordered(true);
        return newDataset;
    }

    @Override
    public Map<Integer, Integer> _get_initial_class_labels() {
        return this.initialClassLabels;
    }

    public void setReordered(boolean status) {
        this.isReordered = status;
    }

    public void setInitialClassOrder(Map<Integer, Integer> initialOrder) {
        this.initialClassLabels = initialOrder;
    }

    @Override
    public void shuffle() {
        this.shuffle(System.nanoTime());
    }

    @Override
    public void shuffle(long seed) {
        Random rand = new Random(seed);
        List<Integer> indices = new ArrayList<>();
        for (int i = 0; i < this.size(); i++) indices.add(i);
        Collections.shuffle(indices, rand);

        List<Object> newData = new ArrayList<>();
        List<Integer> newLabels = new ArrayList<>();
        List<Integer> newIndices = new ArrayList<>();

        for (int i : indices) {
            newData.add(data.get(i));
            newLabels.add(labels.get(i));
            newIndices.add(this.indices.get(i));
        }

        this.data = newData;
        this.labels = newLabels;
        this.indices = new ArrayList<>(newIndices); //newIndices;
    }

    @Override
    public ListObjectDataset shallow_clone() {
        ListObjectDataset clone = new ListObjectDataset(this.size()); //ListObjectDataset(this.size(), this.length());
        clone.data = new ArrayList<>(this.data);
        clone.labels = new ArrayList<>(this.labels);
        clone.indices = new ArrayList<>(this.indices);
        clone.classMap = new LinkedHashMap<>(this.classMap);
        return clone;
    }

    @Override
    public ListObjectDataset deep_clone() {
        ListObjectDataset clone = new ListObjectDataset(this.size()); //ListObjectDataset(this.size(), this.length());
        for (int i = 0; i < this.size(); i++) {
            Object original = this.data.get(i);
            //Object copy = Arrays.copyOf(original, original.length);
            Object copy;
            if (is2D) {
                Object[][] originalArray = (Object[][]) original;
                copy = Arrays.copyOf(originalArray, originalArray.length);
            } else {
                Object[] originalArray = (Object[]) original;
                copy = Arrays.copyOf(originalArray, originalArray.length);
            }

            clone.add(this.labels.get(i), copy, this.indices.get(i));
        }
        return clone;
    }

    @Override
    public ListObjectDataset sample_n(int n_items, Random rand) {
        int n = Math.min(n_items, this.size());
        ListObjectDataset sample = new ListObjectDataset(n); //ListObjectDataset(n, this.length());
        List<Integer> indices = new ArrayList<>();
        for (int i = 0; i < this.size(); i++) indices.add(i);
        Collections.shuffle(indices, rand);

        for (int i = 0; i < n; i++) {
            int idx = indices.get(i);
            sample.add(this.labels.get(idx), this.data.get(idx), this.indices.get(idx));
        }

        return sample;
    }

    /*@Override
    public ListObjectDataset sort_on(int timestamp) {
        // Optional: define sorting logic based on timestamp index
        // This assumes timestamp is a numeric value at a fixed column index
        List<Integer> sortedIndices = new ArrayList<>();
        for (int i = 0; i < this.size(); i++) sortedIndices.add(i);

        sortedIndices.sort(Comparator.comparingDouble(i -> {
            Object val = data.get(i)[timestamp];
            return (val instanceof Number) ? ((Number) val).doubleValue() : Double.NaN;
        }));

        ListObjectDataset sorted = new ListObjectDataset(this.size(), this.length());
        for (int i : sortedIndices) {
            sorted.add(labels.get(i), data.get(i), indices.get(i));
        }

        return sorted;
    }*/

    @Override
    public boolean isNumeric(int i) {
        //TODO
        return false;
    }

    @Override
    public boolean isCategorical(int i) {
        //TODO
        return false;
    }

    public boolean isBoolean(int i) {
        //TODO
        return false;
    }

    @Override
    public boolean isDate(int i) {
        //TODO
        return false;
    }
}

// example use of ObjectDataUtils
//        Object val = row[i];
//        if (ObjectDataUtils.isNumeric(val)) {
//        double num = ObjectDataUtils.getAsDouble(val);
//        }

