package datasets.readers.api;

import datasets.ListObjectDataset;

import java.io.IOException;

/**
 * Public plugin contract for eagerly reading one complete dataset.
 *
 * <p>Unlike {@link CustomSeriesReader}, which materializes one observation
 * per invocation, a custom dataset reader reads and assembles every
 * observation represented by the supplied {@link CustomReaderContext} in one
 * invocation. The configured data path may identify a single file, a
 * directory, an archive, or another dataset-level source understood by the
 * plugin.</p>
 *
 * <h2>Labels</h2>
 *
 * <p>The returned dataset may be labeled or unlabeled. A plugin may read
 * labels embedded in the primary data source, load labels from
 * {@link CustomReaderContext#getLabelPath()}, or obtain label-source details
 * from reader-specific parameters. When a sidecar label path is configured,
 * implementations may delegate to PFGAP's shared sidecar-label reader rather
 * than duplicating its label parsing and validation.</p>
 *
 * <p>The context supplied for each invocation contains the data and label
 * paths associated with that specific training, validation, or test read.
 * An isolation-mode reader may return an unlabeled dataset when no labels are
 * supplied.</p>
 *
 * <h2>Values and preprocessing</h2>
 *
 * <p>Implementations should return raw observation values. A plugin must not
 * apply PFGAP standardization itself; configured standardization and later
 * preprocessing remain the responsibility of the ordinary PFGAP dataset
 * preparation pipeline.</p>
 *
 * <p>A whole-dataset reader may return observations using the standard PFGAP
 * representations supported by its task, including one-dimensional or
 * two-dimensional primitive numeric arrays and their generic object-array
 * counterparts. Unlike the per-file time-series custom-reader coordinators,
 * this contract does not require every observation to preserve a separate
 * dimension axis.</p>
 *
 * <h2>Lifecycle</h2>
 *
 * <p>Implementations must provide an accessible public no-argument
 * constructor so the plugin loader can instantiate them reflectively. A
 * plugin that owns resources beyond the duration of {@link #read} may also
 * implement {@link AutoCloseable}; the runtime adapter closes such plugins
 * before releasing their class loader.</p>
 *
 * <p>Because this is an eager reader, returned datasets should contain
 * ordinary PFGAP-compatible observations and labels that remain usable after
 * the plugin and its class loader are closed.</p>
 */
@FunctionalInterface
public interface CustomDatasetReader {

    /**
     * Reads and assembles one complete dataset.
     *
     * @param context non-null immutable dataset-reader configuration
     * @return non-null eagerly materialized dataset, labeled or unlabeled
     * @throws IOException if the configured source cannot be read or decoded
     */
    ListObjectDataset read(
            CustomReaderContext context
    ) throws IOException;
}
