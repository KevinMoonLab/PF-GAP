package datasets.readers.api;

import datasets.readers.lazy.LazySeriesRef;

import java.io.IOException;

/**
 * Public plugin contract for materializing one dataset observation.
 *
 * <p>The same plugin can be used by eager and lazy custom dataset readers.
 * Eager readers invoke {@link #read(LazySeriesRef, CustomReaderContext)} while
 * constructing the dataset. Lazy readers invoke it only when an observation
 * is requested.</p>
 *
 * <h2>Source location</h2>
 *
 * <p>The complete {@link LazySeriesRef} is supplied rather than only a file
 * path. Current per-file plugins normally read {@code reference.getFile()}.
 * Plugins should also use {@code reference.getIndex()} when the observation
 * index is meaningful. Supplying the complete reference keeps this contract
 * compatible with future non-file locators, byte ranges, record identifiers,
 * and shared-container readers.</p>
 *
 * <h2>Configuration</h2>
 *
 * <p>Reader-specific configuration is supplied through the immutable
 * {@link CustomReaderContext}. Implementations should not read mutable global
 * PFGAP configuration when the required setting can be obtained from the
 * context.</p>
 *
 * <h2>Returned representation</h2>
 *
 * <p>The returned object must be non-null and contain raw observation values.
 * A custom reader must not apply PFGAP standardization itself. When built-in
 * standardization is configured, PFGAP applies the prepared training
 * statistics after this method returns.</p>
 *
 * <p>Standard numeric PFGAP representations are:</p>
 *
 * <pre>
 * float[time]
 * double[time]
 * float[dimension][time]
 * double[dimension][time]
 * </pre>
 *
 * <p>Built-in standardization supports these primitive one-dimensional and
 * two-dimensional numeric representations. Numeric missing values should be
 * represented by primitive {@code NaN}. Boxed numeric arrays are not part of
 * the standard numeric contract.</p>
 *
 * <p>Standard generic PFGAP representations are:</p>
 *
 * <pre>
 * Object[time]
 * Object[dimension][time]
 * </pre>
 *
 * <p>Generic missing values should be represented by {@code null}.</p>
 *
 * <h3>Per-file custom readers</h3>
 *
 * <p>When this plugin is used through {@code CustomPerFileReader} or
 * {@code LazyCustomPerFileReader}, the per-file time-series contract is more
 * specific: the plugin must preserve the dimension axis and return a
 * two-dimensional representation:</p>
 *
 * <pre>
 * float[dimension][time]
 * double[dimension][time]
 * Object[dimension][time]
 * </pre>
 *
 * <p>A univariate per-file series therefore remains {@code [1][time]}. The
 * per-file coordinators are responsible for enforcing this narrower contract.</p>
 *
 * <p>A proprietary representation remains valid when built-in
 * standardization is disabled and every configured consumer, including
 * distance functions, explicitly supports that representation.</p>
 *
 * <h2>Serialization and lifecycle</h2>
 *
 * <p>The plugin implementation does not need to implement
 * {@link java.io.Serializable}. Runtime instances, class loaders, open files,
 * memory mappings, and caches are not serialized. Saved lazy models retain
 * the plugin descriptor and serializable reader configuration, then recreate
 * the plugin in a new JVM.</p>
 *
 * <p>Implementations must provide an accessible no-argument constructor so
 * the plugin loader can instantiate them reflectively. Resources owned by the
 * runtime adapter are released when the registered reader is replaced or the
 * lazy-reader registry is cleared.</p>
 *
 * <h2>Thread safety</h2>
 *
 * <p>PFGAP may invoke one plugin instance concurrently for different
 * references. Thread-safe implementations should keep mutable per-read state
 * local to this method and use immutable or safely concurrent shared caches.
 * Plugins declared non-thread-safe are synchronized by the runtime adapter;
 * implementations must not add unnecessary global synchronization.</p>
 *
 * <p>Implementations should preserve interruption status when wrapping an
 * interruption-related failure and should include useful source information
 * in thrown exceptions. The runtime adapter adds reader-key and reference
 * context when propagating plugin failures.</p>
 */
@FunctionalInterface
public interface CustomSeriesReader {

    /**
     * Materializes one raw dataset observation.
     *
     * @param reference non-null reference identifying the observation
     * @param context non-null immutable plugin configuration
     * @return non-null raw observation representation
     * @throws IOException if the source cannot be read or decoded
     */
    Object read(
            LazySeriesRef reference,
            CustomReaderContext context
    ) throws IOException;
}
