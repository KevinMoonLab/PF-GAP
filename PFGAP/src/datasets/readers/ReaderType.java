package datasets.readers;

/**
 * Enumerates the supported dataset reader types.
 *
 * ReaderType is intended to be used by ReaderOptions and
 * DatasetReaderFactory to determine which DatasetReader
 * implementation should be constructed.
 */
public enum ReaderType {

    /**
     * Standard delimited files such as CSV, TSV, TXT, or row-encoded
     * multivariate time series where each dataset instance appears on
     * one row.
     */
    DELIMITED,

    /**
     * .ts files, such as those used by the UEA/UCR time-series archive.
     * Supports univariate and multivariate time series, with optional
     * embedded class labels.
     */
    TS,

    /**
     * Long-format CSV, TSV, or similar delimited data.
     *
     * Example:
     *
     *      id,time,feature1,feature2,label
     *      A,0,1.2,5.1,class1
     *      A,1,1.5,5.3,class1
     *      B,0,3.2,7.4,class2
     *
     * Rows are grouped by an ID column into time-series instances.
     */
    LONG_FORMAT_DELIMITED,

    /**
     * Long-format Parquet data.
     *
     * Conceptually similar to LONG_FORMAT_DELIMITED, but backed by
     * a Parquet file instead of a text-delimited file.
     */
    LONG_FORMAT_PARQUET,

    /**
     * Directory-based dataset where each file represents one dataset
     * instance. The individual files may themselves be delimited files,
     * Parquet files, or other supported formats.
     */
    DIRECTORY,

    /**
     * HDF5 file containing array-like data.
     *
     * Initial support should probably be limited to common array shapes,
     * for example:
     *
     *      [N, T]
     *      [N, D, T]
     *      [N, H, W, C]
     */
    HDF5,

    /**
     * Parquet format where each row is one time-series instance and
     * feature columns contain sequence/list values.
     *
     * Example:
     *
     *      id,feature1,feature2,label
     *      A,[1,2,3],[4,5,6],class1
     *      B,[2,3,4],[7,8,9],class2
     *
     * This is lower priority than long-format Parquet but useful enough
     * to reserve as an explicit reader type.
     */
    NESTED_PARQUET,
    LAZY_PER_FILE_PARQUET,
    PER_FILE_PARQUET,
    PER_FILE_DELIMITED,
    LAZY_PER_FILE_DELIMITED,
    LAZY_PER_FILE_NUMERIC_DELIMITED,
    PER_FILE_NUMERIC_DELIMITED,
    NUMERIC_LONG_FORMAT,
    PER_FILE_NUMERIC_PARQUET,
    LAZY_PER_FILE_NUMERIC_PARQUET,
    PARQUET_COLUMN_FILE_READER,
    PER_FILE_CUSTOM,
    LAZY_PER_FILE_CUSTOM,
}