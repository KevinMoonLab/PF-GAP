package datasets.readers.lazy;

public interface LazySeriesReader {

    Object read(LazySeriesRef ref);

    default void clearCache() {
    }
}