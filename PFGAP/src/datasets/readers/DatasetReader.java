package datasets.readers;

import datasets.ListObjectDataset;
import java.io.IOException;

/**
 * Common interface implemented by all dataset readers.
 *
 * Every reader is responsible for converting some external
 * representation (CSV, TSV, TS, Parquet, HDF5, directory-based
 * datasets, etc.) into a ListObjectDataset.
 *
 * Downstream algorithms should interact only with
 * ListObjectDataset and remain agnostic to the original
 * storage format.
 */
public interface DatasetReader {

    /**
     * Reads data from the configured source and returns
     * a populated ListObjectDataset.
     *
     * @return dataset containing all instances and labels
     * @throws IOException if the underlying source cannot be read
     */
    ListObjectDataset read() throws IOException;
}