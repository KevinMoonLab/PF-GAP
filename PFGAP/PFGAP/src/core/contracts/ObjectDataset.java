package core.contracts;

import java.util.*;

//This whole thing might need to be redone.
import datasets.ListObjectDataset;
import imputation.MissingIndices;

public interface ObjectDataset {

    public int size();

    public int length();

    //public int length(); //is it fair to have a length attribute? Is it even needed?

    public List<Object> getData();

    public MissingIndices getMissingIndices();

    public void setMissingIndices(MissingIndices mi);

    public void setData(List<?> newData);

    public void setLength(int len);

    public int getLength();

    public void add(Integer label, Object series, Integer index);

    //public void add(Integer label, Object[] series, Integer index);

   // public void add(Integer label, Object[][] series, Integer index);

    public void remove(int i);

    public Object get_series(int i);

    public Integer get_class(int i);

    public Integer get_index(int i);

    public int get_num_classes();

    public void set_indices(ArrayList<Integer> indices);

    public int get_class_size(Integer class_label);

    public Map<Integer, Integer> get_class_map();

    public int[] get_unique_classes();

    public Set<Integer> get_unique_classes_as_set();

    public Map<Integer, ListObjectDataset> split_classes();

    public double gini(); // we may need to define how gini works with mixed types

    public List<Object> _internal_data_list();

    public List<Integer> _internal_class_list();

    public Object[] _internal_data_array();

    public ArrayList<Integer> _internal_indices_list();

    public int[] _internal_class_array();

    public ObjectDataset reorder_class_labels(Map<Integer, Integer> new_order);

    public Map<Integer, Integer> _get_initial_class_labels();

    public void shuffle();

    public void shuffle(long seed);

    public ListObjectDataset shallow_clone();

    public ListObjectDataset deep_clone();

    public ListObjectDataset sample_n(int n_items, Random rand);

    //public ListObjectDataset sort_on(int timestamp); // we may need to define how sorting works with mixed types

    public boolean isNumeric(int i);
    public boolean isCategorical(int i);
    public boolean isBoolean(int i);
    public boolean isDate(int i);

}
