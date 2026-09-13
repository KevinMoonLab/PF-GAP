package output;

import proximity.CompressedSparseProximityMatrix;

import java.io.BufferedWriter;
import java.io.IOException;
import java.io.Writer;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.Objects;

/**
 * Writes dense and compressed-sparse proximity matrices.
 *
 * <p>Dense matrices are written as CSV. Sparse matrices are streamed directly
 * from {@link CompressedSparseProximityMatrix} to Matrix Market coordinate
 * format without constructing maps, triplet collections, or dense
 * intermediates.</p>
 *
 * <p>Matrix Market coordinate indices are one-based, while PFGAP and the CSR
 * representation use zero-based indices internally. This writer adds one to
 * each row and column index during output.</p>
 */
public final class ProximityWriter {

    private static final String DEFAULT_SPARSE_DESCRIPTION =
            "PFGAP sparse proximity matrix";

    private static final String DEFAULT_DENSE_DESCRIPTION =
            "PFGAP dense proximity matrix";

    private ProximityWriter() {
    }

    /**
     * Writes a dense proximity matrix as CSV.
     *
     * <p>The CSV contains one header row with column indices, one row-index
     * column, and one numeric field per matrix entry.</p>
     *
     * @param path output CSV path
     * @param matrix rectangular dense proximity matrix
     * @return normalized absolute output path
     * @throws IOException if writing fails
     */
    public static Path writeDenseCsv(
            Path path,
            double[][] matrix
    ) throws IOException {
        DenseShape shape =
                validateDenseMatrix(
                        path,
                        matrix
                );

        Path outputPath =
                normalizeAndPreparePath(
                        path
                );

        try (BufferedWriter writer =
                     CsvUtils.newWriter(
                             outputPath
                     )) {

            writeDenseHeader(
                    writer,
                    shape.columnCount()
            );

            for (int rowIndex = 0;
                 rowIndex < shape.rowCount();
                 rowIndex++) {

                writeDenseRow(
                        writer,
                        rowIndex,
                        matrix[rowIndex]
                );
            }
        }

        return outputPath;
    }

    /**
     * Writes a CSR proximity matrix in Matrix Market coordinate format.
     *
     * <p>The matrix is written as a general matrix. This is the safe default
     * because proximity matrices may be rectangular or asymmetric. Symmetric
     * storage must be selected explicitly with
     * {@link #writeSparseMatrix(Path, CompressedSparseProximityMatrix, boolean,
     * String)}.</p>
     *
     * @param path output Matrix Market path
     * @param matrix compressed sparse proximity matrix
     * @return normalized absolute output path
     * @throws IOException if writing fails
     */
    public static Path writeSparseMatrix(
            Path path,
            CompressedSparseProximityMatrix matrix
    ) throws IOException {
        return writeSparseMatrix(
                path,
                matrix,
                false,
                DEFAULT_SPARSE_DESCRIPTION
        );
    }

    /**
     * Writes a CSR proximity matrix in Matrix Market coordinate format.
     *
     * <p>When {@code symmetric} is false, every retained CSR entry is written
     * and the Matrix Market header declares {@code general}. When
     * {@code symmetric} is true, the matrix must be square and exactly
     * symmetric; only the upper triangle is written and the header declares
     * {@code symmetric}. Exact validation prevents accidental data loss when a
     * directed proximity such as RF-GAP is supplied.</p>
     *
     * @param path output Matrix Market path
     * @param matrix compressed sparse proximity matrix
     * @param symmetric whether to validate and emit symmetric storage
     * @param description optional Matrix Market comment
     * @return normalized absolute output path
     * @throws IOException if writing fails
     */
    public static Path writeSparseMatrix(
            Path path,
            CompressedSparseProximityMatrix matrix,
            boolean symmetric,
            String description
    ) throws IOException {
        Objects.requireNonNull(
                path,
                "Proximity output path cannot be null."
        );

        Objects.requireNonNull(
                matrix,
                "Sparse proximity matrix cannot be null."
        );

        if (symmetric) {
            validateSymmetric(
                    matrix
            );
        }

        long entryCount =
                symmetric
                        ? countUpperTriangleEntries(matrix)
                        : matrix.nonZeroCountLong();

        Path outputPath =
                normalizeAndPreparePath(
                        path
                );

        try (BufferedWriter writer =
                     newUtf8Writer(
                             outputPath
                     )) {

            writeMatrixMarketCoordinateHeader(
                    writer,
                    symmetric,
                    description
            );

            writeCoordinateShape(
                    writer,
                    matrix.rowCount(),
                    matrix.columnCount(),
                    entryCount
            );

            writeSparseEntries(
                    writer,
                    matrix,
                    symmetric
            );
        }

        return outputPath;
    }

    /**
     * Writes a dense matrix in Matrix Market array format.
     *
     * <p>Matrix Market array storage is column-major, so values are emitted one
     * column at a time.</p>
     */
    public static Path writeDenseMatrixMarket(
            Path path,
            double[][] matrix,
            String description
    ) throws IOException {
        DenseShape shape =
                validateDenseMatrix(
                        path,
                        matrix
                );

        Path outputPath =
                normalizeAndPreparePath(
                        path
                );

        try (BufferedWriter writer =
                     newUtf8Writer(
                             outputPath
                     )) {

            writer.write(
                    "%%MatrixMarket matrix array real general"
            );
            writer.write('\n');

            writeMatrixMarketComment(
                    writer,
                    description
            );

            writer.write(
                    Integer.toString(
                            shape.rowCount()
                    )
            );
            writer.write(' ');
            writer.write(
                    Integer.toString(
                            shape.columnCount()
                    )
            );
            writer.write('\n');

            for (int columnIndex = 0;
                 columnIndex < shape.columnCount();
                 columnIndex++) {

                for (int rowIndex = 0;
                     rowIndex < shape.rowCount();
                     rowIndex++) {

                    double value =
                            matrix[rowIndex][columnIndex];

                    writer.write(
                            Double.toString(
                                    value
                            )
                    );
                    writer.write('\n');
                }
            }
        }

        return outputPath;
    }

    /**
     * Writes a dense matrix in Matrix Market array format using the default
     * description.
     */
    public static Path writeDenseMatrixMarket(
            Path path,
            double[][] matrix
    ) throws IOException {
        return writeDenseMatrixMarket(
                path,
                matrix,
                DEFAULT_DENSE_DESCRIPTION
        );
    }

    private static void writeSparseEntries(
            Writer writer,
            CompressedSparseProximityMatrix matrix,
            boolean upperTriangleOnly
    ) throws IOException {
        for (int rowIndex = 0;
             rowIndex < matrix.rowCount();
             rowIndex++) {

            int entryCount =
                    matrix.rowEntryCount(
                            rowIndex
                    );

            for (int rowOffset = 0;
                 rowOffset < entryCount;
                 rowOffset++) {

                int columnIndex =
                        matrix.columnIndexAt(
                                rowIndex,
                                rowOffset
                        );

                if (upperTriangleOnly
                        && rowIndex > columnIndex) {
                    continue;
                }

                double value =
                        matrix.valueAt(
                                rowIndex,
                                rowOffset
                        );

                if (value == 0.0) {
                    throw new IllegalStateException(
                            "CSR matrix retains an exact zero at row "
                                    + rowIndex
                                    + ", column "
                                    + columnIndex
                                    + "."
                    );
                }

                writer.write(
                        Integer.toString(
                                rowIndex + 1
                        )
                );
                writer.write(' ');
                writer.write(
                        Integer.toString(
                                columnIndex + 1
                        )
                );
                writer.write(' ');
                writer.write(
                        Double.toString(
                                value
                        )
                );
                writer.write('\n');
            }
        }
    }

    private static void validateSymmetric(
            CompressedSparseProximityMatrix matrix
    ) {
        if (matrix.rowCount() != matrix.columnCount()) {
            throw new IllegalArgumentException(
                    "Symmetric Matrix Market output requires a square matrix, "
                            + "but received "
                            + matrix.rowCount()
                            + " x "
                            + matrix.columnCount()
                            + "."
            );
        }

        for (int rowIndex = 0;
             rowIndex < matrix.rowCount();
             rowIndex++) {

            int entryCount =
                    matrix.rowEntryCount(
                            rowIndex
                    );

            for (int rowOffset = 0;
                 rowOffset < entryCount;
                 rowOffset++) {

                int columnIndex =
                        matrix.columnIndexAt(
                                rowIndex,
                                rowOffset
                        );

                if (columnIndex <= rowIndex) {
                    continue;
                }

                double value =
                        matrix.valueAt(
                                rowIndex,
                                rowOffset
                        );

                if (!matrix.contains(columnIndex, rowIndex)) {
                    throw new IllegalArgumentException(
                            "Sparse matrix is not symmetric: entry ("
                                    + rowIndex
                                    + ", "
                                    + columnIndex
                                    + ") has no reflected entry."
                    );
                }

                double reflectedValue =
                        matrix.get(
                                columnIndex,
                                rowIndex
                        );

                if (Double.compare(value, reflectedValue) != 0) {
                    throw new IllegalArgumentException(
                            "Sparse matrix is not exactly symmetric at entries ("
                                    + rowIndex
                                    + ", "
                                    + columnIndex
                                    + ") and ("
                                    + columnIndex
                                    + ", "
                                    + rowIndex
                                    + ")."
                    );
                }
            }
        }
    }

    private static long countUpperTriangleEntries(
            CompressedSparseProximityMatrix matrix
    ) {
        long count = 0L;

        for (int rowIndex = 0;
             rowIndex < matrix.rowCount();
             rowIndex++) {

            int entryCount =
                    matrix.rowEntryCount(
                            rowIndex
                    );

            for (int rowOffset = 0;
                 rowOffset < entryCount;
                 rowOffset++) {

                int columnIndex =
                        matrix.columnIndexAt(
                                rowIndex,
                                rowOffset
                        );

                if (rowIndex <= columnIndex) {
                    count = Math.addExact(
                            count,
                            1L
                    );
                }
            }
        }

        return count;
    }

    private static void writeCoordinateShape(
            Writer writer,
            int rowCount,
            int columnCount,
            long entryCount
    ) throws IOException {
        writer.write(
                Integer.toString(
                        rowCount
                )
        );
        writer.write(' ');
        writer.write(
                Integer.toString(
                        columnCount
                )
        );
        writer.write(' ');
        writer.write(
                Long.toString(
                        entryCount
                )
        );
        writer.write('\n');
    }

    private static void writeDenseHeader(
            BufferedWriter writer,
            int columnCount
    ) throws IOException {
        writer.write(
                CsvUtils.escape(
                        "instance_index"
                )
        );

        for (int columnIndex = 0;
             columnIndex < columnCount;
             columnIndex++) {

            writer.write(
                    CsvUtils.DEFAULT_DELIMITER
            );
            writer.write(
                    CsvUtils.escape(
                            columnIndex
                    )
            );
        }

        writer.write(
                CsvUtils.RECORD_SEPARATOR
        );
    }

    private static void writeDenseRow(
            BufferedWriter writer,
            int rowIndex,
            double[] row
    ) throws IOException {
        writer.write(
                CsvUtils.escape(
                        rowIndex
                )
        );

        for (double value : row) {
            writer.write(
                    CsvUtils.DEFAULT_DELIMITER
            );
            writer.write(
                    CsvUtils.escape(
                            CsvUtils.numericField(
                                    value
                            )
                    )
            );
        }

        writer.write(
                CsvUtils.RECORD_SEPARATOR
        );
    }

    private static void writeMatrixMarketCoordinateHeader(
            Writer writer,
            boolean symmetric,
            String description
    ) throws IOException {
        writer.write(
                symmetric
                        ? "%%MatrixMarket matrix coordinate real symmetric"
                        : "%%MatrixMarket matrix coordinate real general"
        );
        writer.write('\n');

        writeMatrixMarketComment(
                writer,
                description
        );

        writer.write(
                "% Internal zero-based indices were converted to one-based."
        );
        writer.write('\n');
    }

    private static void writeMatrixMarketComment(
            Writer writer,
            String description
    ) throws IOException {
        if (description == null || description.isBlank()) {
            return;
        }

        String normalizedDescription =
                description.trim()
                        .replace('\r', ' ')
                        .replace('\n', ' ');

        writer.write("% ");
        writer.write(normalizedDescription);
        writer.write('\n');
    }

    private static DenseShape validateDenseMatrix(
            Path path,
            double[][] matrix
    ) {
        Objects.requireNonNull(
                path,
                "Proximity output path cannot be null."
        );
        Objects.requireNonNull(
                matrix,
                "Dense proximity matrix cannot be null."
        );

        if (matrix.length == 0) {
            throw new IllegalArgumentException(
                    "Dense proximity matrix must contain at least one row."
            );
        }
        if (matrix[0] == null || matrix[0].length == 0) {
            throw new IllegalArgumentException(
                    "Dense proximity matrix must contain at least one column."
            );
        }

        int columnCount = matrix[0].length;

        for (int rowIndex = 0;
             rowIndex < matrix.length;
             rowIndex++) {

            double[] row = matrix[rowIndex];
            if (row == null) {
                throw new IllegalArgumentException(
                        "Dense proximity matrix row "
                                + rowIndex
                                + " cannot be null."
                );
            }
            if (row.length != columnCount) {
                throw new IllegalArgumentException(
                        "Dense proximity matrix is ragged at row "
                                + rowIndex
                                + "."
                );
            }

            for (int columnIndex = 0;
                 columnIndex < columnCount;
                 columnIndex++) {

                if (!Double.isFinite(row[columnIndex])) {
                    throw new IllegalArgumentException(
                            "Proximity matrix contains non-finite value "
                                    + row[columnIndex]
                                    + " at row "
                                    + rowIndex
                                    + ", column "
                                    + columnIndex
                                    + "."
                    );
                }
            }
        }

        return new DenseShape(
                matrix.length,
                columnCount
        );
    }

    private static Path normalizeAndPreparePath(
            Path path
    ) throws IOException {
        Path outputPath =
                path.toAbsolutePath()
                        .normalize();

        Path parent = outputPath.getParent();
        if (parent != null) {
            Files.createDirectories(parent);
        }

        return outputPath;
    }

    private static BufferedWriter newUtf8Writer(
            Path outputPath
    ) throws IOException {
        return Files.newBufferedWriter(
                outputPath,
                StandardCharsets.UTF_8,
                StandardOpenOption.CREATE,
                StandardOpenOption.WRITE,
                StandardOpenOption.TRUNCATE_EXISTING
        );
    }

    private record DenseShape(
            int rowCount,
            int columnCount
    ) {
    }
}
