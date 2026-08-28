package datasets.readers.api;

import datasets.readers.lazy.LazySeriesRef;

import java.io.IOException;

/**
 * Public plugin interface for user-defined instance readers.
 *
 * <p>A CustomSeriesReader materializes exactly one dataset instance from a
 * {@link LazySeriesRef}. The same implementation can be used by both eager
 * and lazy per-file dataset readers:</p>
 *
 * <ul>
 *     <li>
 *         An eager custom per-file reader invokes this method while building
 *         the dataset.
 *     </li>
 *     <li>
 *         A lazy custom per-file reader invokes this method only when a
 *         distance calculation requests the referenced instance.
 *     </li>
 * </ul>
 *
 * <p>The initial supported use case is one file per instance, including
 * proprietary binary formats. In that case, the implementation normally
 * reads:</p>
 *
 * <pre>
 * reference.getFile()
 * </pre>
 *
 * <p>The complete reference is supplied instead of only a Path so this API
 * remains compatible with future additions to LazySeriesRef, such as
 * instance metadata, byte offsets, record identifiers, or other
 * serializable locator information.</p>
 *
 * <p>Reader-specific configuration is supplied through
 * {@link CustomReaderContext}. Implementations should not access mutable
 * PFGAP global configuration when the required information can be supplied
 * through the context.</p>
 *
 * <p>The returned object may use any representation understood by the
 * configured distance functions. Common PFGAP representations include:</p>
 *
 * <pre>
 * double[]
 * Double[]
 * Object[]
 * double[][]
 * Double[][]
 * Object[][]
 * </pre>
 *
 * <p>The interface does not require Serializable. Runtime reader instances,
 * class loaders, open files, memory mappings, and caches are not serialized.
 * Saved lazy models instead retain the plugin descriptor and reader
 * parameters, then reconstruct the plugin in a new JVM.</p>
 *
 * <p>Implementations must provide an accessible no-argument constructor so
 * the Java plugin loader can instantiate them reflectively.</p>
 *
 * <h2>Thread safety</h2>
 *
 * <p>PFGAP may eventually invoke this method concurrently for distinct
 * references when parallel execution is enabled. Implementations should keep
 * per-read mutable state inside this method. Shared immutable lookup tables
 * are safe. A later adapter option will allow non-thread-safe plugins to be
 * synchronized without changing this interface.</p>
 */
public interface CustomSeriesReader {

    /**
     * Materializes one dataset instance.
     *
     * @param reference reference identifying the instance to read
     * @param context immutable configuration for this reader
     * @return non-null instance representation
     * @throws IOException if the underlying data cannot be read
     */
    Object read(
            LazySeriesRef reference,
            CustomReaderContext context
    ) throws IOException;
}