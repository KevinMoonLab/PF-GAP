package datasets.writers;

import datasets.ListObjectDataset;

import java.io.IOException;
import java.nio.file.Path;

/**
 * Writes a materialized {@link ListObjectDataset} in one supported external
 * dataset format.
 *
 * <p>A writer implementation owns format-specific encoding, shape validation,
 * and any streaming conversion needed by that format. Experiment orchestration
 * decides whether output is requested and obtains the appropriate writer from
 * {@code DatasetWriterFactory}; it should not contain format-specific writing
 * logic.</p>
 *
 * <p>Implementations must not mutate the supplied dataset. In particular, when
 * {@link DatasetWriteOptions} contains reusable or per-series standardization
 * parameters, a writer should inverse-transform each numeric value while
 * emitting it rather than inverse-transforming the source arrays in place or
 * constructing a complete inverse-transformed dataset copy.</p>
 *
 * <p>Implementations should preserve primitive numeric storage semantics where
 * the destination format supports them. A float-backed dataset should not be
 * widened into a double-backed intermediate merely for output.</p>
 */
public interface DatasetWriter {

    /**
     * Writes one materialized dataset according to the supplied immutable
     * options.
     *
     * @param dataset materialized dataset to write; implementations may reject
     *        lazy references when their format does not support streaming from
     *        a lazy source
     * @param options output path, labels, schema metadata, and optional inverse
     *        standardization parameters
     * @return the path of the completed output artifact
     * @throws IOException if output creation or writing fails
     * @throws IllegalArgumentException if the dataset representation is not
     *         supported by the concrete writer or is incompatible with the
     *         supplied options
     */
    Path write(
            ListObjectDataset dataset,
            DatasetWriteOptions options
    ) throws IOException;

    /**
     * Returns whether this writer can encode the supplied materialized dataset
     * representation under the given options without beginning output.
     *
     * <p>This is intended for factory diagnostics and early validation. The
     * write method must still validate the complete dataset because later
     * instances may differ from the first instance.</p>
     *
     * @param dataset dataset whose representation should be checked
     * @param options proposed output options
     * @return true when the representation is supported
     */
    boolean supports(
            ListObjectDataset dataset,
            DatasetWriteOptions options
    );

    /**
     * Returns a stable short format name for diagnostics and manifests.
     *
     * @return writer format name
     */
    String formatName();
}
