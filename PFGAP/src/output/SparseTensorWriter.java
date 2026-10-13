package output;

import java.io.BufferedWriter;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.Objects;

/**
 * Format-level writer for sparse third-order tensor coordinate files.
 *
 * <p>Each non-comment record contains a one-based instance index, dimension
 * index, time index, and finite real value:</p>
 *
 * <pre>
 * instance dimension time value
 * </pre>
 *
 * <p>The representation follows the conventional sparse {@code .tns}
 * coordinate layout. Callers provide the logical tensor envelope, the exact
 * number of entries, and a callback that streams zero-based coordinates to a
 * sink. The sink validates coordinates and converts them to one-based file
 * indices.</p>
 *
 * <p>The full tensor envelope and declared entry count are written as comments
 * because imputed-only output may not contain coordinates at the maximum extent
 * of every mode. Exact zero values are retained because entry presence denotes
 * an originally missing value, not a mathematically nonzero value.</p>
 *
 * <p>For ragged time-series data, the declared time count is the maximum time
 * length. The file is a sparse patch whose coordinates are interpreted against
 * the original dataset; it does not independently encode every instance's
 * ragged shape.</p>
 */
public final class SparseTensorWriter {

    /** Receives one zero-based sparse tensor coordinate from the caller. */
    @FunctionalInterface
    public interface CoordinateSink {
        void write(
                int zeroBasedInstance,
                int zeroBasedDimension,
                int zeroBasedTime,
                double value
        ) throws IOException;
    }

    /** Streams sparse tensor coordinates to a writer-provided sink. */
    @FunctionalInterface
    public interface CoordinateEntrySource {
        void writeEntries(CoordinateSink sink) throws IOException;
    }

    private SparseTensorWriter() {
        // Utility class.
    }

    /**
     * Writes a sparse real third-order tensor in coordinate form.
     *
     * <p>The entry callback must emit exactly {@code entryCount} entries.
     * Coordinates supplied to the sink are zero-based; coordinates written to
     * the file are one-based.</p>
     *
     * @param path output {@code .tns} path
     * @param instanceCount logical instance count
     * @param dimensionCount maximum logical dimension count
     * @param timeCount maximum logical time length
     * @param entryCount exact number of entries the callback will emit
     * @param description optional comment, null or blank to omit
     * @param entries streaming entry callback
     * @return normalized absolute output path
     * @throws IOException if output fails or the callback emits an invalid
     *         coordinate, value, or entry count
     */
    public static Path writeCoordinate(
            Path path,
            int instanceCount,
            int dimensionCount,
            int timeCount,
            long entryCount,
            String description,
            CoordinateEntrySource entries
    ) throws IOException {
        Path outputPath = normalizeAndPreparePath(path);
        validateShape(
                instanceCount,
                dimensionCount,
                timeCount,
                entryCount
        );
        Objects.requireNonNull(
                entries,
                "Sparse tensor coordinate entry source cannot be null."
        );

        try (BufferedWriter writer = newUtf8Writer(outputPath)) {
            writer.write("% PFGAP sparse tensor coordinate file");
            writer.write('\n');
            writer.write("% shape: ");
            writer.write(Integer.toString(instanceCount));
            writer.write(' ');
            writer.write(Integer.toString(dimensionCount));
            writer.write(' ');
            writer.write(Integer.toString(timeCount));
            writer.write('\n');
            writer.write("% entries: ");
            writer.write(Long.toString(entryCount));
            writer.write('\n');
            writer.write("% coordinates: instance dimension time value");
            writer.write('\n');
            writeComment(writer, description);

            CountingCoordinateSink sink = new CountingCoordinateSink(
                    writer,
                    instanceCount,
                    dimensionCount,
                    timeCount,
                    entryCount
            );
            entries.writeEntries(sink);
            sink.validateFinalCount();
        }

        return outputPath;
    }

    private static void validateShape(
            int instanceCount,
            int dimensionCount,
            int timeCount,
            long entryCount
    ) {
        if (instanceCount < 0
                || dimensionCount < 0
                || timeCount < 0) {
            throw new IllegalArgumentException(
                    "Sparse tensor dimensions cannot be negative: "
                            + instanceCount
                            + " x "
                            + dimensionCount
                            + " x "
                            + timeCount
                            + "."
            );
        }
        if (entryCount < 0) {
            throw new IllegalArgumentException(
                    "Sparse tensor entry count cannot be negative: "
                            + entryCount
                            + "."
            );
        }

        long maximumEntries;
        try {
            maximumEntries = Math.multiplyExact(
                    Math.multiplyExact(
                            (long) instanceCount,
                            (long) dimensionCount
                    ),
                    (long) timeCount
            );
        } catch (ArithmeticException exception) {
            throw new IllegalArgumentException(
                    "Sparse tensor shape capacity exceeds long range: "
                            + instanceCount
                            + " x "
                            + dimensionCount
                            + " x "
                            + timeCount
                            + ".",
                    exception
            );
        }

        if (entryCount > maximumEntries) {
            throw new IllegalArgumentException(
                    "Declared sparse tensor entry count "
                            + entryCount
                            + " exceeds the rectangular capacity "
                            + maximumEntries
                            + " for shape "
                            + instanceCount
                            + " x "
                            + dimensionCount
                            + " x "
                            + timeCount
                            + "."
            );
        }
    }

    private static void writeComment(
            BufferedWriter writer,
            String description
    ) throws IOException {
        if (description == null || description.isBlank()) {
            return;
        }
        String normalized = description
                .replace('\r', ' ')
                .replace('\n', ' ')
                .trim();
        if (!normalized.isEmpty()) {
            writer.write("% ");
            writer.write(normalized);
            writer.write('\n');
        }
    }

    private static Path normalizeAndPreparePath(Path path) throws IOException {
        Path outputPath = Objects.requireNonNull(
                path,
                "Sparse tensor output path cannot be null."
        ).toAbsolutePath().normalize();
        Path parent = outputPath.getParent();
        if (parent != null) {
            Files.createDirectories(parent);
        }
        return outputPath;
    }

    private static BufferedWriter newUtf8Writer(Path outputPath)
            throws IOException {
        return Files.newBufferedWriter(
                outputPath,
                StandardCharsets.UTF_8,
                StandardOpenOption.CREATE,
                StandardOpenOption.WRITE,
                StandardOpenOption.TRUNCATE_EXISTING
        );
    }

    private static final class CountingCoordinateSink
            implements CoordinateSink {
        private final BufferedWriter writer;
        private final int instanceCount;
        private final int dimensionCount;
        private final int timeCount;
        private final long expectedEntryCount;
        private long writtenEntryCount;

        private CountingCoordinateSink(
                BufferedWriter writer,
                int instanceCount,
                int dimensionCount,
                int timeCount,
                long expectedEntryCount
        ) {
            this.writer = writer;
            this.instanceCount = instanceCount;
            this.dimensionCount = dimensionCount;
            this.timeCount = timeCount;
            this.expectedEntryCount = expectedEntryCount;
        }

        @Override
        public void write(
                int zeroBasedInstance,
                int zeroBasedDimension,
                int zeroBasedTime,
                double value
        ) throws IOException {
            if (writtenEntryCount >= expectedEntryCount) {
                throw new IOException(
                        "Sparse tensor coordinate source emitted more than "
                                + "the declared "
                                + expectedEntryCount
                                + " entries."
                );
            }
            if (zeroBasedInstance < 0
                    || zeroBasedInstance >= instanceCount
                    || zeroBasedDimension < 0
                    || zeroBasedDimension >= dimensionCount
                    || zeroBasedTime < 0
                    || zeroBasedTime >= timeCount) {
                throw new IOException(
                        "Sparse tensor coordinate is out of bounds: instance="
                                + zeroBasedInstance
                                + ", dimension="
                                + zeroBasedDimension
                                + ", time="
                                + zeroBasedTime
                                + ", shape="
                                + instanceCount
                                + " x "
                                + dimensionCount
                                + " x "
                                + timeCount
                                + "."
                );
            }
            if (!Double.isFinite(value)) {
                throw new IOException(
                        "Sparse tensor coordinate value must be finite at "
                                + "instance="
                                + zeroBasedInstance
                                + ", dimension="
                                + zeroBasedDimension
                                + ", time="
                                + zeroBasedTime
                                + ": "
                                + value
                                + "."
                );
            }

            writer.write(Integer.toString(zeroBasedInstance + 1));
            writer.write(' ');
            writer.write(Integer.toString(zeroBasedDimension + 1));
            writer.write(' ');
            writer.write(Integer.toString(zeroBasedTime + 1));
            writer.write(' ');
            writer.write(Double.toString(value));
            writer.write('\n');
            writtenEntryCount++;
        }

        private void validateFinalCount() throws IOException {
            if (writtenEntryCount != expectedEntryCount) {
                throw new IOException(
                        "Sparse tensor coordinate source emitted "
                                + writtenEntryCount
                                + " entries, but declared "
                                + expectedEntryCount
                                + "."
                );
            }
        }
    }
}
