package output;

import java.io.BufferedWriter;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.Objects;

/**
 * Format-level writer for Matrix Market coordinate and array files.
 *
 * <p>This class is independent of proximity, imputation, and any particular
 * sparse-matrix implementation. Callers provide dimensions, the declared
 * coordinate-entry count, and a callback that streams zero-based entries to a
 * sink. The sink converts indices to Matrix Market's one-based convention.</p>
 *
 * <p>Coordinate output deliberately permits exact zero values. This is needed
 * for structural sparse results such as originally missing cells whose imputed
 * value is zero. A caller whose sparse representation forbids retained zeros
 * should enforce that rule before invoking this writer.</p>
 */
public final class MatrixMarketWriter {

    /** Matrix Market coordinate symmetry declarations supported by this API. */
    public enum Symmetry {
        GENERAL("general"),
        SYMMETRIC("symmetric");

        private final String token;

        Symmetry(String token) {
            this.token = token;
        }

        private String token() {
            return token;
        }
    }

    /**
     * Receives zero-based coordinate entries from a caller-supplied traversal.
     */
    @FunctionalInterface
    public interface CoordinateSink {
        void write(
                int zeroBasedRow,
                int zeroBasedColumn,
                double value
        ) throws IOException;
    }

    /**
     * Streams coordinate entries to a Matrix Market sink.
     */
    @FunctionalInterface
    public interface CoordinateEntrySource {
        void writeEntries(
                CoordinateSink sink
        ) throws IOException;
    }

    private MatrixMarketWriter() {
        // Utility class.
    }

    /**
     * Writes a general real coordinate matrix.
     */
    public static Path writeCoordinate(
            Path path,
            int rowCount,
            int columnCount,
            long entryCount,
            String description,
            CoordinateEntrySource entries
    ) throws IOException {
        return writeCoordinate(
                path,
                rowCount,
                columnCount,
                entryCount,
                Symmetry.GENERAL,
                description,
                entries
        );
    }

    /**
     * Writes a real coordinate matrix by streaming caller-provided entries.
     *
     * <p>The entry callback must emit exactly {@code entryCount} entries.
     * Indices supplied to the sink are zero-based. For symmetric output, every
     * supplied entry must lie in the upper triangle, including the diagonal.
     * This class validates storage shape and emitted entries, but determining
     * whether a source matrix is mathematically symmetric remains the caller's
     * responsibility.</p>
     *
     * @param path output .mtx path
     * @param rowCount logical row count
     * @param columnCount logical column count
     * @param entryCount exact number of entries the callback will emit
     * @param symmetry Matrix Market symmetry declaration
     * @param description optional comment, null or blank to omit
     * @param entries streaming entry callback
     * @return normalized absolute output path
     * @throws IOException if output fails or the entry callback emits an
     *         invalid coordinate/value/count
     */
    public static Path writeCoordinate(
            Path path,
            int rowCount,
            int columnCount,
            long entryCount,
            Symmetry symmetry,
            String description,
            CoordinateEntrySource entries
    ) throws IOException {
        Path outputPath = normalizeAndPreparePath(path);
        validateCoordinateShape(
                rowCount,
                columnCount,
                entryCount,
                symmetry
        );
        Objects.requireNonNull(entries, "Coordinate entry source cannot be null.");

        try (BufferedWriter writer = newUtf8Writer(outputPath)) {
            writer.write("%%MatrixMarket matrix coordinate real ");
            writer.write(symmetry.token());
            writer.write('\n');
            writeComment(writer, description);
            writeCoordinateShape(
                    writer,
                    rowCount,
                    columnCount,
                    entryCount
            );

            CountingCoordinateSink sink = new CountingCoordinateSink(
                    writer,
                    rowCount,
                    columnCount,
                    entryCount,
                    symmetry
            );
            entries.writeEntries(sink);
            sink.validateFinalCount();
        }

        return outputPath;
    }

    /**
     * Writes a rectangular dense double matrix in Matrix Market array format.
     * Values are emitted in the column-major order required by the format.
     */
    public static Path writeArray(
            Path path,
            double[][] matrix,
            String description
    ) throws IOException {
        DenseShape shape = validateDense(matrix);
        Path outputPath = normalizeAndPreparePath(path);

        try (BufferedWriter writer = newUtf8Writer(outputPath)) {
            writer.write("%%MatrixMarket matrix array real general");
            writer.write('\n');
            writeComment(writer, description);
            writer.write(Integer.toString(shape.rowCount()));
            writer.write(' ');
            writer.write(Integer.toString(shape.columnCount()));
            writer.write('\n');

            for (int column = 0; column < shape.columnCount(); column++) {
                for (int row = 0; row < shape.rowCount(); row++) {
                    writer.write(Double.toString(matrix[row][column]));
                    writer.write('\n');
                }
            }
        }
        return outputPath;
    }

    /** Writes a dense float matrix in Matrix Market array format. */
    public static Path writeArray(
            Path path,
            float[][] matrix,
            String description
    ) throws IOException {
        DenseShape shape = validateDense(matrix);
        Path outputPath = normalizeAndPreparePath(path);

        try (BufferedWriter writer = newUtf8Writer(outputPath)) {
            writer.write("%%MatrixMarket matrix array real general");
            writer.write('\n');
            writeComment(writer, description);
            writer.write(Integer.toString(shape.rowCount()));
            writer.write(' ');
            writer.write(Integer.toString(shape.columnCount()));
            writer.write('\n');

            for (int column = 0; column < shape.columnCount(); column++) {
                for (int row = 0; row < shape.rowCount(); row++) {
                    writer.write(Float.toString(matrix[row][column]));
                    writer.write('\n');
                }
            }
        }
        return outputPath;
    }

    private static void validateCoordinateShape(
            int rowCount,
            int columnCount,
            long entryCount,
            Symmetry symmetry
    ) {
        Objects.requireNonNull(symmetry, "Matrix Market symmetry cannot be null.");
        if (rowCount < 0 || columnCount < 0) {
            throw new IllegalArgumentException(
                    "Matrix dimensions cannot be negative: "
                            + rowCount
                            + " x "
                            + columnCount
                            + "."
            );
        }
        if (entryCount < 0) {
            throw new IllegalArgumentException(
                    "Matrix Market entry count cannot be negative: "
                            + entryCount
                            + "."
            );
        }
        if (symmetry == Symmetry.SYMMETRIC && rowCount != columnCount) {
            throw new IllegalArgumentException(
                    "Symmetric Matrix Market output requires a square matrix, "
                            + "but received "
                            + rowCount
                            + " x "
                            + columnCount
                            + "."
            );
        }
        long maximumEntries = symmetry == Symmetry.SYMMETRIC
                ? triangularCapacity(rowCount)
                : rectangularCapacity(rowCount, columnCount);
        if (entryCount > maximumEntries) {
            throw new IllegalArgumentException(
                    "Declared Matrix Market entry count "
                            + entryCount
                            + " exceeds the maximum storage capacity "
                            + maximumEntries
                            + " for "
                            + rowCount
                            + " x "
                            + columnCount
                            + " "
                            + symmetry.token()
                            + " storage."
            );
        }
    }

    private static void writeCoordinateShape(
            BufferedWriter writer,
            int rowCount,
            int columnCount,
            long entryCount
    ) throws IOException {
        writer.write(Integer.toString(rowCount));
        writer.write(' ');
        writer.write(Integer.toString(columnCount));
        writer.write(' ');
        writer.write(Long.toString(entryCount));
        writer.write('\n');
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

    private static DenseShape validateDense(double[][] matrix) {
        Objects.requireNonNull(matrix, "Dense matrix cannot be null.");
        int columns = matrix.length == 0
                ? 0
                : requireRow(matrix[0], 0).length;
        for (int row = 0; row < matrix.length; row++) {
            double[] values = requireRow(matrix[row], row);
            if (values.length != columns) {
                throw new IllegalArgumentException(
                        "Dense matrix is ragged at row " + row + "."
                );
            }
            for (int column = 0; column < columns; column++) {
                if (!Double.isFinite(values[column])) {
                    throw new IllegalArgumentException(
                            "Dense matrix contains non-finite value at row "
                                    + row
                                    + ", column "
                                    + column
                                    + "."
                    );
                }
            }
        }
        return new DenseShape(matrix.length, columns);
    }

    private static DenseShape validateDense(float[][] matrix) {
        Objects.requireNonNull(matrix, "Dense matrix cannot be null.");
        int columns = matrix.length == 0
                ? 0
                : requireRow(matrix[0], 0).length;
        for (int row = 0; row < matrix.length; row++) {
            float[] values = requireRow(matrix[row], row);
            if (values.length != columns) {
                throw new IllegalArgumentException(
                        "Dense matrix is ragged at row " + row + "."
                );
            }
            for (int column = 0; column < columns; column++) {
                if (!Float.isFinite(values[column])) {
                    throw new IllegalArgumentException(
                            "Dense matrix contains non-finite value at row "
                                    + row
                                    + ", column "
                                    + column
                                    + "."
                    );
                }
            }
        }
        return new DenseShape(matrix.length, columns);
    }

    private static double[] requireRow(double[] row, int index) {
        return Objects.requireNonNull(
                row,
                "Dense matrix row " + index + " cannot be null."
        );
    }

    private static float[] requireRow(float[] row, int index) {
        return Objects.requireNonNull(
                row,
                "Dense matrix row " + index + " cannot be null."
        );
    }

    private static Path normalizeAndPreparePath(Path path) throws IOException {
        Path outputPath = Objects.requireNonNull(
                path,
                "Matrix Market output path cannot be null."
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

    private static long rectangularCapacity(int rows, int columns) {
        return Math.multiplyExact((long) rows, (long) columns);
    }

    private static long triangularCapacity(int size) {
        return Math.multiplyExact((long) size, (long) size + 1L) / 2L;
    }

    private static final class CountingCoordinateSink
            implements CoordinateSink {
        private final BufferedWriter writer;
        private final int rowCount;
        private final int columnCount;
        private final long expectedEntryCount;
        private final Symmetry symmetry;
        private long writtenEntryCount;

        private CountingCoordinateSink(
                BufferedWriter writer,
                int rowCount,
                int columnCount,
                long expectedEntryCount,
                Symmetry symmetry
        ) {
            this.writer = writer;
            this.rowCount = rowCount;
            this.columnCount = columnCount;
            this.expectedEntryCount = expectedEntryCount;
            this.symmetry = symmetry;
        }

        @Override
        public void write(
                int zeroBasedRow,
                int zeroBasedColumn,
                double value
        ) throws IOException {
            if (writtenEntryCount >= expectedEntryCount) {
                throw new IOException(
                        "Coordinate source emitted more than the declared "
                                + expectedEntryCount
                                + " entries."
                );
            }
            if (zeroBasedRow < 0 || zeroBasedRow >= rowCount
                    || zeroBasedColumn < 0
                    || zeroBasedColumn >= columnCount) {
                throw new IOException(
                        "Matrix Market coordinate is out of bounds: row="
                                + zeroBasedRow
                                + ", column="
                                + zeroBasedColumn
                                + ", shape="
                                + rowCount
                                + " x "
                                + columnCount
                                + "."
                );
            }
            if (symmetry == Symmetry.SYMMETRIC
                    && zeroBasedRow > zeroBasedColumn) {
                throw new IOException(
                        "Symmetric coordinate output accepts only upper-"
                                + "triangle entries, but received row="
                                + zeroBasedRow
                                + ", column="
                                + zeroBasedColumn
                                + "."
                );
            }
            if (!Double.isFinite(value)) {
                throw new IOException(
                        "Matrix Market coordinate value must be finite at row="
                                + zeroBasedRow
                                + ", column="
                                + zeroBasedColumn
                                + ": "
                                + value
                                + "."
                );
            }

            writer.write(Integer.toString(zeroBasedRow + 1));
            writer.write(' ');
            writer.write(Integer.toString(zeroBasedColumn + 1));
            writer.write(' ');
            writer.write(Double.toString(value));
            writer.write('\n');
            writtenEntryCount++;
        }

        private void validateFinalCount() throws IOException {
            if (writtenEntryCount != expectedEntryCount) {
                throw new IOException(
                        "Coordinate source emitted "
                                + writtenEntryCount
                                + " entries, but declared "
                                + expectedEntryCount
                                + "."
                );
            }
        }
    }

    private record DenseShape(int rowCount, int columnCount) {
    }
}
