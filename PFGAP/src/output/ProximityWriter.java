package output;

import java.io.BufferedWriter;
import java.io.IOException;
import java.io.Writer;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.Map;
import java.util.Objects;
import java.util.TreeMap;

/**
 * Writes dense and sparse proximity matrices.
 *
 * Supported output formats:
 *
 *     Dense matrices:
 *         CSV
 *
 *     Sparse matrices:
 *         Matrix Market coordinate format
 *
 * Dense CSV example:
 *
 *     instance_index,0,1,2
 *     0,1.0,0.4,0.2
 *     1,0.4,1.0,0.7
 *     2,0.2,0.7,1.0
 *
 * Sparse Matrix Market example:
 *
 *     %%MatrixMarket matrix coordinate real symmetric
 *     % PFGAP training proximity matrix
 *     3 3 5
 *     1 1 1.0
 *     1 2 0.4
 *     2 2 1.0
 *     2 3 0.7
 *     3 3 1.0
 *
 * Matrix Market uses one-based row and column indices. PFGAP's internal
 * sparse maps are expected to use zero-based indices, so this writer adds
 * one to each index.
 *
 * Sparse training proximity matrices are usually square and symmetric.
 * Sparse test-to-training proximity matrices are rectangular and general.
 *
 * The writer does not construct a dense intermediate matrix when writing
 * sparse output.
 */
public final class ProximityWriter {

    private ProximityWriter() {
    }

    /**
     * Writes a dense proximity matrix as CSV.
     *
     * The CSV contains:
     *
     *     one header row containing column indices
     *     one row-index column
     *     one numeric field per matrix entry
     *
     * @param path output CSV path
     * @param matrix dense proximity matrix
     * @return normalized absolute output path
     * @throws IOException if writing fails
     */
    public static Path writeDenseCsv(
            Path path,
            double[][] matrix
    ) throws IOException {
        validateDenseMatrix(
                path,
                matrix
        );

        Path outputPath =
                normalizePath(path);

        int rowCount =
                matrix.length;

        int columnCount =
                matrix[0].length;

        try (BufferedWriter writer =
                     CsvUtils.newWriter(outputPath)) {

            writeDenseHeader(
                    writer,
                    columnCount
            );

            for (int rowIndex = 0;
                 rowIndex < rowCount;
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
     * Writes a sparse training-to-training proximity matrix in Matrix Market
     * coordinate format.
     *
     * Training proximity matrices are expected to be square and symmetric.
     * By default, only the upper triangle, including the diagonal, is
     * written. The Matrix Market header declares the matrix symmetric, so
     * compliant readers reconstruct the reflected lower triangle.
     *
     * @param path output Matrix Market path
     * @param sparseMatrix zero-based sparse proximity map
     * @param size number of training instances
     * @return normalized absolute output path
     * @throws IOException if writing fails
     */
    public static Path writeSparseTrainingMatrix(
            Path path,
            Map<Integer, ? extends Map<Integer, Double>> sparseMatrix,
            int size
    ) throws IOException {
        return writeSparseMatrixMarket(
                path,
                sparseMatrix,
                size,
                size,
                false,
                false,
                "PFGAP training proximity matrix"
        );
    }

    /**
     * Writes a sparse test-to-training proximity matrix in Matrix Market
     * coordinate format.
     *
     * Test-to-training matrices are rectangular and are therefore written
     * as general matrices.
     *
     * @param path output Matrix Market path
     * @param sparseMatrix zero-based sparse proximity map
     * @param testSize number of testing instances
     * @param trainingSize number of training instances
     * @return normalized absolute output path
     * @throws IOException if writing fails
     */
    public static Path writeSparseTestTrainMatrix(
            Path path,
            Map<Integer, ? extends Map<Integer, Double>> sparseMatrix,
            int testSize,
            int trainingSize
    ) throws IOException {
        return writeSparseMatrixMarket(
                path,
                sparseMatrix,
                testSize,
                trainingSize,
                false,
                false,
                "PFGAP test-to-training proximity matrix"
        );
    }

    /**
     * Writes a sparse matrix in Matrix Market coordinate format.
     *
     * The internal sparse representation is expected to use:
     *
     *     outer map key:
     *         zero-based row index
     *
     *     inner map key:
     *         zero-based column index
     *
     *     inner map value:
     *         proximity value
     *
     * When symmetric is true, rowCount and columnCount must be equal.
     *
     * When upperTriangleOnly is true, only entries satisfying:
     *
     *     rowIndex <= columnIndex
     *
     * are written. This should normally be paired with symmetric=true.
     *
     * Zero-valued entries are omitted. Non-finite values are rejected
     * because Matrix Market numeric data should contain usable real values.
     *
     * @param path output Matrix Market path
     * @param sparseMatrix zero-based sparse matrix map
     * @param rowCount matrix row count
     * @param columnCount matrix column count
     * @param symmetric whether the Matrix Market matrix is symmetric
     * @param upperTriangleOnly whether to write only the upper triangle
     * @param description optional Matrix Market comment
     * @return normalized absolute output path
     * @throws IOException if writing fails
     */
    public static Path writeSparseMatrixMarket(
            Path path,
            Map<Integer, ? extends Map<Integer, Double>> sparseMatrix,
            int rowCount,
            int columnCount,
            boolean symmetric,
            boolean upperTriangleOnly,
            String description
    ) throws IOException {
        validateSparseArguments(
                path,
                sparseMatrix,
                rowCount,
                columnCount,
                symmetric,
                upperTriangleOnly
        );

        long nonzeroCount =
                countMatrixMarketEntries(
                        sparseMatrix,
                        rowCount,
                        columnCount,
                        upperTriangleOnly
                );

        Path outputPath =
                normalizePath(path);

        createParentDirectory(
                outputPath
        );

        try (BufferedWriter writer =
                     Files.newBufferedWriter(
                             outputPath,
                             StandardCharsets.UTF_8,
                             StandardOpenOption.CREATE,
                             StandardOpenOption.WRITE,
                             StandardOpenOption.TRUNCATE_EXISTING
                     )) {

            writeMatrixMarketHeader(
                    writer,
                    symmetric,
                    description
            );

            writer.write(
                    Integer.toString(rowCount)
            );

            writer.write(' ');

            writer.write(
                    Integer.toString(columnCount)
            );

            writer.write(' ');

            writer.write(
                    Long.toString(nonzeroCount)
            );

            writer.write('\n');

            writeMatrixMarketEntries(
                    writer,
                    sparseMatrix,
                    rowCount,
                    columnCount,
                    upperTriangleOnly
            );
        }

        return outputPath;
    }

    /**
     * Writes a dense matrix in Matrix Market array format.
     *
     * This optional output is useful when users want one standard matrix
     * format for both dense and sparse results.
     *
     * Matrix Market array storage is column-major. Values are therefore
     * written one column at a time.
     *
     * @param path output Matrix Market path
     * @param matrix dense proximity matrix
     * @param description optional Matrix Market comment
     * @return normalized absolute output path
     * @throws IOException if writing fails
     */
    public static Path writeDenseMatrixMarket(
            Path path,
            double[][] matrix,
            String description
    ) throws IOException {
        validateDenseMatrix(
                path,
                matrix
        );

        Path outputPath =
                normalizePath(path);

        createParentDirectory(
                outputPath
        );

        int rowCount =
                matrix.length;

        int columnCount =
                matrix[0].length;

        try (BufferedWriter writer =
                     Files.newBufferedWriter(
                             outputPath,
                             StandardCharsets.UTF_8,
                             StandardOpenOption.CREATE,
                             StandardOpenOption.WRITE,
                             StandardOpenOption.TRUNCATE_EXISTING
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
                    Integer.toString(rowCount)
            );

            writer.write(' ');

            writer.write(
                    Integer.toString(columnCount)
            );

            writer.write('\n');

            /*
             * Matrix Market array format is column-major.
             */
            for (int columnIndex = 0;
                 columnIndex < columnCount;
                 columnIndex++) {

                for (int rowIndex = 0;
                     rowIndex < rowCount;
                     rowIndex++) {

                    double value =
                            matrix[rowIndex][columnIndex];

                    validateFiniteValue(
                            value,
                            rowIndex,
                            columnIndex
                    );

                    writer.write(
                            Double.toString(value)
                    );

                    writer.write('\n');
                }
            }
        }

        return outputPath;
    }

    /**
     * Writes a dense matrix in Matrix Market array format without a custom
     * description.
     *
     * @param path output Matrix Market path
     * @param matrix dense proximity matrix
     * @return normalized absolute output path
     * @throws IOException if writing fails
     */
    public static Path writeDenseMatrixMarket(
            Path path,
            double[][] matrix
    ) throws IOException {
        return writeDenseMatrixMarket(
                path,
                matrix,
                "PFGAP dense proximity matrix"
        );
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

    private static void writeMatrixMarketHeader(
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
                "% Internal PFGAP indices were converted from "
                        + "zero-based to one-based."
        );

        writer.write('\n');
    }

    private static void writeMatrixMarketComment(
            Writer writer,
            String description
    ) throws IOException {
        if (description == null
                || description.isBlank()) {

            return;
        }

        /*
         * Matrix Market comments are single-line records beginning with %.
         * Replace embedded line separators to avoid accidentally producing
         * uncommented data lines.
         */
        String normalizedDescription =
                description.trim()
                        .replace('\r', ' ')
                        .replace('\n', ' ');

        writer.write("% ");

        writer.write(
                normalizedDescription
        );

        writer.write('\n');
    }

    /**
     * Counts entries before opening the output file because Matrix Market
     * coordinate format requires the nonzero count in its size line.
     *
     * This operation visits the sparse map once. Writing visits it a second
     * time, but no dense matrix or entry collection is allocated.
     */
    private static long countMatrixMarketEntries(
            Map<Integer, ? extends Map<Integer, Double>> sparseMatrix,
            int rowCount,
            int columnCount,
            boolean upperTriangleOnly
    ) {
        long count =
                0L;

        for (Map.Entry<Integer, ? extends Map<Integer, Double>>
                rowEntry : sparseMatrix.entrySet()) {

            Integer rowIndexObject =
                    rowEntry.getKey();

            if (rowIndexObject == null) {
                throw new IllegalArgumentException(
                        "Sparse proximity matrix contains a null row index."
                );
            }

            int rowIndex =
                    rowIndexObject;

            validateIndex(
                    rowIndex,
                    rowCount,
                    "row"
            );

            Map<Integer, Double> columns =
                    rowEntry.getValue();

            if (columns == null) {
                throw new IllegalArgumentException(
                        "Sparse proximity matrix row "
                                + rowIndex
                                + " has a null column map."
                );
            }

            for (Map.Entry<Integer, Double>
                    columnEntry : columns.entrySet()) {

                Integer columnIndexObject =
                        columnEntry.getKey();

                if (columnIndexObject == null) {
                    throw new IllegalArgumentException(
                            "Sparse proximity matrix row "
                                    + rowIndex
                                    + " contains a null column index."
                    );
                }

                int columnIndex =
                        columnIndexObject;

                validateIndex(
                        columnIndex,
                        columnCount,
                        "column"
                );

                Double value =
                        columnEntry.getValue();

                if (value == null) {
                    throw new IllegalArgumentException(
                            "Sparse proximity matrix entry at row "
                                    + rowIndex
                                    + ", column "
                                    + columnIndex
                                    + " has a null value."
                    );
                }

                validateFiniteValue(
                        value,
                        rowIndex,
                        columnIndex
                );

                if (value == 0.0) {
                    continue;
                }

                if (upperTriangleOnly
                        && rowIndex > columnIndex) {

                    continue;
                }

                if (count == Long.MAX_VALUE) {
                    throw new ArithmeticException(
                            "Sparse proximity matrix contains too many "
                                    + "entries to count."
                    );
                }

                count++;
            }
        }

        return count;
    }

    private static void writeMatrixMarketEntries(
            Writer writer,
            Map<Integer, ? extends Map<Integer, Double>> sparseMatrix,
            int rowCount,
            int columnCount,
            boolean upperTriangleOnly
    ) throws IOException {
        /*
         * Use sorted views for deterministic output. This creates maps of
         * references to the existing entries, but it does not construct a
         * dense matrix.
         */
        Map<Integer, ? extends Map<Integer, Double>> sortedRows =
                sparseMatrix instanceof TreeMap
                        ? sparseMatrix
                        : new TreeMap<>(sparseMatrix);

        for (Map.Entry<Integer, ? extends Map<Integer, Double>>
                rowEntry : sortedRows.entrySet()) {

            int rowIndex =
                    rowEntry.getKey();

            validateIndex(
                    rowIndex,
                    rowCount,
                    "row"
            );

            Map<Integer, Double> columns =
                    rowEntry.getValue();

            Map<Integer, Double> sortedColumns =
                    columns instanceof TreeMap
                            ? columns
                            : new TreeMap<>(columns);

            for (Map.Entry<Integer, Double>
                    columnEntry : sortedColumns.entrySet()) {

                int columnIndex =
                        columnEntry.getKey();

                validateIndex(
                        columnIndex,
                        columnCount,
                        "column"
                );

                double value =
                        columnEntry.getValue();

                validateFiniteValue(
                        value,
                        rowIndex,
                        columnIndex
                );

                if (value == 0.0) {
                    continue;
                }

                if (upperTriangleOnly
                        && rowIndex > columnIndex) {

                    continue;
                }

                /*
                 * Matrix Market coordinate indices are one-based.
                 */
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
                        Double.toString(value)
                );

                writer.write('\n');
            }
        }
    }

    private static void validateDenseMatrix(
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

        if (matrix[0] == null) {
            throw new IllegalArgumentException(
                    "Dense proximity matrix row 0 cannot be null."
            );
        }

        if (matrix[0].length == 0) {
            throw new IllegalArgumentException(
                    "Dense proximity matrix must contain at least "
                            + "one column."
            );
        }

        int columnCount =
                matrix[0].length;

        for (int rowIndex = 0;
             rowIndex < matrix.length;
             rowIndex++) {

            double[] row =
                    matrix[rowIndex];

            if (row == null) {
                throw new IllegalArgumentException(
                        "Dense proximity matrix row "
                                + rowIndex
                                + " cannot be null."
                );
            }

            if (row.length != columnCount) {
                throw new IllegalArgumentException(
                        "Dense proximity matrix is ragged. Row 0 contains "
                                + columnCount
                                + " columns, but row "
                                + rowIndex
                                + " contains "
                                + row.length
                                + "."
                );
            }

            for (int columnIndex = 0;
                 columnIndex < row.length;
                 columnIndex++) {

                validateFiniteValue(
                        row[columnIndex],
                        rowIndex,
                        columnIndex
                );
            }
        }
    }

    private static void validateSparseArguments(
            Path path,
            Map<Integer, ? extends Map<Integer, Double>> sparseMatrix,
            int rowCount,
            int columnCount,
            boolean symmetric,
            boolean upperTriangleOnly
    ) {
        Objects.requireNonNull(
                path,
                "Proximity output path cannot be null."
        );

        Objects.requireNonNull(
                sparseMatrix,
                "Sparse proximity matrix cannot be null."
        );

        if (rowCount <= 0) {
            throw new IllegalArgumentException(
                    "Sparse matrix row count must be positive, but "
                            + "received "
                            + rowCount
                            + "."
            );
        }

        if (columnCount <= 0) {
            throw new IllegalArgumentException(
                    "Sparse matrix column count must be positive, but "
                            + "received "
                            + columnCount
                            + "."
            );
        }

        if (symmetric && rowCount != columnCount) {
            throw new IllegalArgumentException(
                    "A symmetric sparse matrix must be square, but "
                            + "received shape "
                            + rowCount
                            + " x "
                            + columnCount
                            + "."
            );
        }

        if (upperTriangleOnly && !symmetric) {
            throw new IllegalArgumentException(
                    "upperTriangleOnly=true requires symmetric=true. "
                            + "Writing only half of a general matrix would "
                            + "discard data."
            );
        }
    }

    private static void validateIndex(
            int index,
            int limit,
            String indexDescription
    ) {
        if (index < 0 || index >= limit) {
            throw new IllegalArgumentException(
                    "Sparse proximity matrix "
                            + indexDescription
                            + " index "
                            + index
                            + " is outside the valid zero-based range [0, "
                            + (limit - 1)
                            + "]."
            );
        }
    }

    private static void validateFiniteValue(
            double value,
            int rowIndex,
            int columnIndex
    ) {
        if (!Double.isFinite(value)) {
            throw new IllegalArgumentException(
                    "Proximity matrix contains non-finite value "
                            + value
                            + " at row "
                            + rowIndex
                            + ", column "
                            + columnIndex
                            + "."
            );
        }
    }

    private static Path normalizePath(
            Path path
    ) {
        return path.toAbsolutePath()
                .normalize();
    }

    private static void createParentDirectory(
            Path path
    ) throws IOException {
        Path parent =
                path.getParent();

        if (parent != null) {
            Files.createDirectories(parent);
        }
    }
}