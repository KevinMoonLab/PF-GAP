package proximity;

import java.util.Objects;

/**
 * Typed result of a proximity-matrix computation.
 *
 * <p>A result preserves the matrix representation selected for the operation:
 * dense results contain a rectangular {@code double[][]}, while sparse results
 * contain a {@link CompressedSparseProximityMatrix}. This removes ambiguous
 * {@code Object} return values and avoids nullable dense/sparse field pairs.</p>
 *
 * <p>The contained matrix is transferred by ownership and is not copied.
 * Callers must treat it as immutable after publication. In particular, callers
 * must not modify rows returned by {@link Dense#values()}.</p>
 */
//public sealed public interface ProximityMatrixResult
//        permits ProximityMatrixResult.Dense,
//                ProximityMatrixResult.Sparse {
public interface ProximityMatrixResult {

    /** Returns the number of matrix rows. */
    int rowCount();

    /** Returns the number of matrix columns. */
    int columnCount();

    /** Returns whether this result uses dense storage. */
    default boolean isDense() {
        return this instanceof Dense;
    }

    /** Returns whether this result uses compressed sparse row storage. */
    default boolean isSparse() {
        return this instanceof Sparse;
    }

    /**
     * Returns the dense result or throws when this result is sparse.
     */
    default Dense requireDense() {
        if (this instanceof Dense dense) {
            return dense;
        }

        throw new IllegalStateException(
                "Proximity result uses sparse storage, not dense storage."
        );
    }

    /**
     * Returns the sparse result or throws when this result is dense.
     */
    default Sparse requireSparse() {
        if (this instanceof Sparse sparse) {
            return sparse;
        }

        throw new IllegalStateException(
                "Proximity result uses dense storage, not sparse storage."
        );
    }

    /**
     * Dense proximity-matrix result.
     *
     * <p>The matrix must be rectangular and contain only finite values. Empty
     * matrices are supported. Because Java cannot infer the column count from
     * an empty outer array, an empty dense result has shape {@code 0 x 0}.</p>
     *
     * @param values owned rectangular dense matrix
     */
    record Dense(
            double[][] values
    ) implements ProximityMatrixResult {

        public Dense {
            Objects.requireNonNull(
                    values,
                    "Dense proximity matrix cannot be null."
            );

            int columnCount = values.length == 0
                    ? 0
                    : requireFirstRow(values);

            for (int rowIndex = 0;
                 rowIndex < values.length;
                 rowIndex++) {

                double[] row = values[rowIndex];

                if (row == null) {
                    throw new IllegalArgumentException(
                            "Dense proximity matrix row "
                                    + rowIndex
                                    + " cannot be null."
                    );
                }

                if (row.length != columnCount) {
                    throw new IllegalArgumentException(
                            "Dense proximity matrix is ragged. Row "
                                    + rowIndex
                                    + " has length "
                                    + row.length
                                    + ", but expected "
                                    + columnCount
                                    + "."
                    );
                }

                for (int columnIndex = 0;
                     columnIndex < row.length;
                     columnIndex++) {

                    if (!Double.isFinite(row[columnIndex])) {
                        throw new IllegalArgumentException(
                                "Dense proximity matrix contains non-finite "
                                        + "value "
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
        }

        @Override
        public int rowCount() {
            return values.length;
        }

        @Override
        public int columnCount() {
            return values.length == 0
                    ? 0
                    : values[0].length;
        }

        private static int requireFirstRow(
                double[][] values
        ) {
            if (values[0] == null) {
                throw new IllegalArgumentException(
                        "Dense proximity matrix row 0 cannot be null."
                );
            }

            return values[0].length;
        }
    }

    /**
     * Compressed sparse row proximity-matrix result.
     *
     * @param values owned immutable CSR matrix
     */
    record Sparse(
            CompressedSparseProximityMatrix values
    ) implements ProximityMatrixResult {

        public Sparse {
            Objects.requireNonNull(
                    values,
                    "Sparse proximity matrix cannot be null."
            );
        }

        @Override
        public int rowCount() {
            return values.rowCount();
        }

        @Override
        public int columnCount() {
            return values.columnCount();
        }
    }
}
