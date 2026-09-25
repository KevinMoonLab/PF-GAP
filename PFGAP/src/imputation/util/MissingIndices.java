package imputation.util;

import java.io.Serial;
import java.io.Serializable;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;

/**
 * Compact dataset-level metadata identifying positions that were originally
 * missing.
 *
 * <p>The representation is CSR-like and fully primitive. All missing positions
 * are stored in one sorted {@code int[]} array. Offset arrays identify the
 * contiguous range belonging to one instance or to one instance/dimension
 * pair. No object is allocated per missing position, instance, or empty row.</p>
 *
 * <p>For one-dimensional observations:</p>
 *
 * <pre>
 * positions[instanceOffsets[i] ... instanceOffsets[i + 1])
 * </pre>
 *
 * <p>For dimension-major observations, {@code instanceOffsets} maps an
 * instance to its flattened dimensions, and {@code groupOffsets} maps each
 * flattened dimension to its range in {@code positions}:</p>
 *
 * <pre>
 * flattenedDimension = instanceOffsets[instance] + dimension
 * positions[groupOffsets[flattenedDimension]
 *           ... groupOffsets[flattenedDimension + 1])
 * </pre>
 *
 * <p>Positions within each range are strictly increasing and nonnegative.
 * The metadata describes the dataset's original missing targets. It may remain
 * attached after initial imputation so iterative imputers can update only those
 * positions; a recorded position does not imply that its current value remains
 * NaN or null.</p>
 */
public final class MissingIndices implements Serializable {

    @Serial
    private static final long serialVersionUID = 1L;

    private final boolean twoDimensional;
    private final int instanceCount;

    /**
     * In 1D, offsets directly into {@link #positions} for each instance.
     * In 2D, offsets into the flattened dimension groups for each instance.
     */
    private final int[] instanceOffsets;

    /**
     * In 2D, offsets into {@link #positions} for each flattened dimension.
     * Null for 1D metadata.
     */
    private final int[] groupOffsets;

    /** All original missing positions in contiguous primitive storage. */
    private final int[] positions;

    private MissingIndices(
            boolean twoDimensional,
            int instanceCount,
            int[] instanceOffsets,
            int[] groupOffsets,
            int[] positions
    ) {
        this.twoDimensional = twoDimensional;
        this.instanceCount = instanceCount;
        this.instanceOffsets = instanceOffsets;
        this.groupOffsets = groupOffsets;
        this.positions = positions;
    }
    /**
     * Creates 1D metadata by taking ownership of builder-created CSR arrays.
     * Package-private to keep the no-copy contract internal to this package.
     */
    static MissingIndices takeOwnership1D(
            int[] instanceOffsets,
            int[] positions
    ) {
        validateCsr1D(instanceOffsets, positions);
        return new MissingIndices(
                false,
                instanceOffsets.length - 1,
                instanceOffsets,
                null,
                positions
        );
    }

    /**
     * Creates 2D metadata by taking ownership of builder-created CSR arrays.
     * Package-private to keep the no-copy contract internal to this package.
     */
    static MissingIndices takeOwnership2D(
            int[] instanceOffsets,
            int[] groupOffsets,
            int[] positions
    ) {
        validateCsr2D(instanceOffsets, groupOffsets, positions);
        return new MissingIndices(
                true,
                instanceOffsets.length - 1,
                instanceOffsets,
                groupOffsets,
                positions
        );
    }

    /**
     * Builds compact 1D metadata from the legacy nested-list representation.
     */
    public static MissingIndices from1D(
            List<? extends List<Integer>> indices
    ) {
        Objects.requireNonNull(indices, "1D missing indices cannot be null.");

        int instanceCount = indices.size();
        int[] instanceOffsets = new int[checkedOffsetLength(instanceCount)];
        long total = 0L;

        for (int instance = 0; instance < instanceCount; instance++) {
            List<Integer> row = Objects.requireNonNull(
                    indices.get(instance),
                    "Missing-position list cannot be null at instance "
                            + instance + "."
            );
            total = checkedAdd(total, row.size());
            instanceOffsets[instance + 1] = checkedPositionCount(total);
        }

        int[] positions = new int[(int) total];
        for (int instance = 0; instance < instanceCount; instance++) {
            writeValidatedPositions(
                    indices.get(instance),
                    positions,
                    instanceOffsets[instance],
                    "instance " + instance
            );
        }

        return new MissingIndices(
                false,
                instanceCount,
                instanceOffsets,
                null,
                positions
        );
    }

    /**
     * Builds compact 2D metadata from the legacy nested-list representation.
     */
    public static MissingIndices from2D(
            List<? extends List<? extends List<Integer>>> indices
    ) {
        Objects.requireNonNull(indices, "2D missing indices cannot be null.");

        int instanceCount = indices.size();
        int[] instanceOffsets = new int[checkedOffsetLength(instanceCount)];
        long totalGroups = 0L;

        for (int instance = 0; instance < instanceCount; instance++) {
            List<? extends List<Integer>> dimensions = Objects.requireNonNull(
                    indices.get(instance),
                    "Dimension metadata cannot be null at instance "
                            + instance + "."
            );
            totalGroups = checkedAdd(totalGroups, dimensions.size());
            instanceOffsets[instance + 1] = checkedPositionCount(totalGroups);
        }

        int groupCount = (int) totalGroups;
        int[] groupOffsets = new int[checkedOffsetLength(groupCount)];
        long totalPositions = 0L;
        int group = 0;

        for (int instance = 0; instance < instanceCount; instance++) {
            List<? extends List<Integer>> dimensions = indices.get(instance);
            for (int dimension = 0;
                    dimension < dimensions.size();
                    dimension++) {
                List<Integer> row = Objects.requireNonNull(
                        dimensions.get(dimension),
                        "Missing-position list cannot be null at instance "
                                + instance
                                + ", dimension "
                                + dimension
                                + "."
                );
                totalPositions = checkedAdd(totalPositions, row.size());
                groupOffsets[++group] =
                        checkedPositionCount(totalPositions);
            }
        }

        int[] positions = new int[(int) totalPositions];
        group = 0;
        for (int instance = 0; instance < instanceCount; instance++) {
            List<? extends List<Integer>> dimensions = indices.get(instance);
            for (int dimension = 0;
                    dimension < dimensions.size();
                    dimension++) {
                writeValidatedPositions(
                        dimensions.get(dimension),
                        positions,
                        groupOffsets[group],
                        "instance "
                                + instance
                                + ", dimension "
                                + dimension
                );
                group++;
            }
        }

        return new MissingIndices(
                true,
                instanceCount,
                instanceOffsets,
                groupOffsets,
                positions
        );
    }

    /**
     * Builds compact 1D metadata from primitive per-instance position arrays.
     */
    public static MissingIndices fromPrimitive1D(int[][] indices) {
        Objects.requireNonNull(indices, "1D missing indices cannot be null.");

        int[] offsets = new int[checkedOffsetLength(indices.length)];
        long total = 0L;
        for (int instance = 0; instance < indices.length; instance++) {
            int[] row = Objects.requireNonNull(
                    indices[instance],
                    "Missing positions cannot be null at instance "
                            + instance + "."
            );
            validateSortedUniqueNonnegative(row, "instance " + instance);
            total = checkedAdd(total, row.length);
            offsets[instance + 1] = checkedPositionCount(total);
        }

        int[] positions = new int[(int) total];
        for (int instance = 0; instance < indices.length; instance++) {
            System.arraycopy(
                    indices[instance],
                    0,
                    positions,
                    offsets[instance],
                    indices[instance].length
            );
        }

        return new MissingIndices(
                false,
                indices.length,
                offsets,
                null,
                positions
        );
    }

    /**
     * Builds compact 2D metadata from primitive per-dimension position arrays.
     */
    public static MissingIndices fromPrimitive2D(int[][][] indices) {
        Objects.requireNonNull(indices, "2D missing indices cannot be null.");

        int[] instanceOffsets = new int[checkedOffsetLength(indices.length)];
        long totalGroups = 0L;
        for (int instance = 0; instance < indices.length; instance++) {
            int[][] dimensions = Objects.requireNonNull(
                    indices[instance],
                    "Dimension metadata cannot be null at instance "
                            + instance + "."
            );
            totalGroups = checkedAdd(totalGroups, dimensions.length);
            instanceOffsets[instance + 1] =
                    checkedPositionCount(totalGroups);
        }

        int groupCount = (int) totalGroups;
        int[] groupOffsets = new int[checkedOffsetLength(groupCount)];
        long totalPositions = 0L;
        int group = 0;
        for (int instance = 0; instance < indices.length; instance++) {
            for (int dimension = 0;
                    dimension < indices[instance].length;
                    dimension++) {
                int[] row = Objects.requireNonNull(
                        indices[instance][dimension],
                        "Missing positions cannot be null at instance "
                                + instance
                                + ", dimension "
                                + dimension
                                + "."
                );
                validateSortedUniqueNonnegative(
                        row,
                        "instance "
                                + instance
                                + ", dimension "
                                + dimension
                );
                totalPositions = checkedAdd(totalPositions, row.length);
                groupOffsets[++group] =
                        checkedPositionCount(totalPositions);
            }
        }

        int[] positions = new int[(int) totalPositions];
        group = 0;
        for (int instance = 0; instance < indices.length; instance++) {
            for (int dimension = 0;
                    dimension < indices[instance].length;
                    dimension++) {
                int[] row = indices[instance][dimension];
                System.arraycopy(
                        row,
                        0,
                        positions,
                        groupOffsets[group],
                        row.length
                );
                group++;
            }
        }

        return new MissingIndices(
                true,
                indices.length,
                instanceOffsets,
                groupOffsets,
                positions
        );
    }

    public boolean is2D() {
        return twoDimensional;
    }

    public boolean is1D() {
        return !twoDimensional;
    }

    public int instanceCount() {
        return instanceCount;
    }

    public int missingValueCount() {
        return positions.length;
    }

    public boolean isEmpty() {
        return positions.length == 0;
    }

    /** Returns the inclusive position-array offset for one 1D instance. */
    public int start1D(int instance) {
        require1D();
        requireInstance(instance);
        return instanceOffsets[instance];
    }

    /** Returns the exclusive position-array offset for one 1D instance. */
    public int end1D(int instance) {
        require1D();
        requireInstance(instance);
        return instanceOffsets[instance + 1];
    }

    /** Returns the inclusive position-array offset for one 2D dimension. */
    public int start2D(int instance, int dimension) {
        return groupOffsets[flatGroup(instance, dimension)];
    }

    /** Returns the exclusive position-array offset for one 2D dimension. */
    public int end2D(int instance, int dimension) {
        return groupOffsets[flatGroup(instance, dimension) + 1];
    }

    /** Returns one missing position by global compact offset. */
    public int positionAt(int offset) {
        if (offset < 0 || offset >= positions.length) {
            throw new IndexOutOfBoundsException(
                    "Missing-position offset "
                            + offset
                            + " is outside [0, "
                            + (positions.length - 1)
                            + "]."
            );
        }
        return positions[offset];
    }

    public int missingCount1D(int instance) {
        return end1D(instance) - start1D(instance);
    }

    public int missingCount2D(int instance, int dimension) {
        return end2D(instance, dimension) - start2D(instance, dimension);
    }

    public int dimensionCount(int instance) {
        require2D();
        requireInstance(instance);
        return instanceOffsets[instance + 1] - instanceOffsets[instance];
    }

    public boolean contains1D(int instance, int position) {
        return binarySearch(
                positions,
                start1D(instance),
                end1D(instance),
                position
        ) >= 0;
    }

    public boolean contains2D(
            int instance,
            int dimension,
            int position
    ) {
        return binarySearch(
                positions,
                start2D(instance, dimension),
                end2D(instance, dimension),
                position
        ) >= 0;
    }

    /** Returns a defensive copy of all compact missing positions. */
    public int[] copyPositions() {
        return positions.clone();
    }

    /** Returns a defensive copy of the instance-level offsets. */
    public int[] copyInstanceOffsets() {
        return instanceOffsets.clone();
    }

    /** Returns a defensive copy of 2D group offsets, or null for 1D. */
    public int[] copyGroupOffsets() {
        return groupOffsets == null ? null : groupOffsets.clone();
    }

    private int flatGroup(int instance, int dimension) {
        require2D();
        requireInstance(instance);
        int dimensionCount = dimensionCount(instance);
        if (dimension < 0 || dimension >= dimensionCount) {
            throw new IndexOutOfBoundsException(
                    "Dimension index "
                            + dimension
                            + " is outside [0, "
                            + (dimensionCount - 1)
                            + "] for instance "
                            + instance
                            + "."
            );
        }
        return instanceOffsets[instance] + dimension;
    }

    private void require1D() {
        if (twoDimensional) {
            throw new IllegalStateException(
                    "MissingIndices contains 2D metadata, not 1D metadata."
            );
        }
    }

    private void require2D() {
        if (!twoDimensional) {
            throw new IllegalStateException(
                    "MissingIndices contains 1D metadata, not 2D metadata."
            );
        }
    }

    private void requireInstance(int instance) {
        if (instance < 0 || instance >= instanceCount) {
            throw new IndexOutOfBoundsException(
                    "Instance index "
                            + instance
                            + " is outside [0, "
                            + (instanceCount - 1)
                            + "]."
            );
        }
    }

    private static int binarySearch(
            int[] values,
            int fromInclusive,
            int toExclusive,
            int target
    ) {
        int low = fromInclusive;
        int high = toExclusive - 1;
        while (low <= high) {
            int middle = (low + high) >>> 1;
            int value = values[middle];
            if (value < target) {
                low = middle + 1;
            } else if (value > target) {
                high = middle - 1;
            } else {
                return middle;
            }
        }
        return -(low + 1);
    }

    private static void writeValidatedPositions(
            List<Integer> source,
            int[] destination,
            int destinationOffset,
            String location
    ) {
        int previous = -1;
        for (int index = 0; index < source.size(); index++) {
            Integer boxed = source.get(index);
            if (boxed == null) {
                throw new IllegalArgumentException(
                        "Missing position cannot be null at "
                                + location
                                + ", entry "
                                + index
                                + "."
                );
            }
            int position = boxed;
            validatePosition(position, previous, index, location);
            destination[destinationOffset + index] = position;
            previous = position;
        }
    }

    private static void validateSortedUniqueNonnegative(
            int[] values,
            String location
    ) {
        int previous = -1;
        for (int index = 0; index < values.length; index++) {
            validatePosition(values[index], previous, index, location);
            previous = values[index];
        }
    }

    private static void validatePosition(
            int position,
            int previous,
            int index,
            String location
    ) {
        if (position < 0) {
            throw new IllegalArgumentException(
                    "Missing position cannot be negative at "
                            + location
                            + ": "
                            + position
                            + "."
            );
        }
        if (index > 0 && position <= previous) {
            throw new IllegalArgumentException(
                    "Missing positions must be strictly increasing at "
                            + location
                            + ". Found "
                            + previous
                            + " followed by "
                            + position
                            + "."
            );
        }
    }

    private static void validateCsr1D(
            int[] instanceOffsets,
            int[] positions
    ) {
        Objects.requireNonNull(
                instanceOffsets,
                "Instance offsets cannot be null."
        );
        Objects.requireNonNull(positions, "Positions cannot be null.");
        validateOffsets(
                instanceOffsets,
                positions.length,
                "instance offsets"
        );
        validatePositionRanges(instanceOffsets, positions, "instance");
    }

    private static void validateCsr2D(
            int[] instanceOffsets,
            int[] groupOffsets,
            int[] positions
    ) {
        Objects.requireNonNull(
                instanceOffsets,
                "Instance offsets cannot be null."
        );
        Objects.requireNonNull(
                groupOffsets,
                "Group offsets cannot be null."
        );
        Objects.requireNonNull(positions, "Positions cannot be null.");
        if (instanceOffsets.length == 0) {
            throw new IllegalArgumentException(
                    "Instance offsets must contain at least the zero boundary."
            );
        }
        validateOffsets(
                instanceOffsets,
                groupOffsets.length - 1,
                "instance offsets"
        );
        validateOffsets(
                groupOffsets,
                positions.length,
                "group offsets"
        );
        validatePositionRanges(groupOffsets, positions, "dimension group");
    }

    private static void validateOffsets(
            int[] offsets,
            int expectedFinalOffset,
            String description
    ) {
        if (offsets.length == 0 || offsets[0] != 0) {
            throw new IllegalArgumentException(
                    description + " must begin with zero."
            );
        }
        int previous = 0;
        for (int index = 1; index < offsets.length; index++) {
            int current = offsets[index];
            if (current < previous) {
                throw new IllegalArgumentException(
                        description + " must be nondecreasing."
                );
            }
            previous = current;
        }
        if (offsets[offsets.length - 1] != expectedFinalOffset) {
            throw new IllegalArgumentException(
                    description
                            + " final offset must equal "
                            + expectedFinalOffset
                            + ", but was "
                            + offsets[offsets.length - 1]
                            + "."
            );
        }
    }

    private static void validatePositionRanges(
            int[] offsets,
            int[] positions,
            String description
    ) {
        for (int group = 0; group < offsets.length - 1; group++) {
            int previous = -1;
            for (int offset = offsets[group];
                    offset < offsets[group + 1];
                    offset++) {
                int position = positions[offset];
                if (position < 0 || position <= previous) {
                    throw new IllegalArgumentException(
                            "Positions in "
                                    + description
                                    + " "
                                    + group
                                    + " must be nonnegative and strictly "
                                    + "increasing."
                    );
                }
                previous = position;
            }
        }
    }

    private static int checkedOffsetLength(int groupCount) {
        if (groupCount == Integer.MAX_VALUE) {
            throw new IllegalArgumentException(
                    "Too many missing-index groups for an offset array."
            );
        }
        return groupCount + 1;
    }

    private static long checkedAdd(long current, int addition) {
        if (Long.MAX_VALUE - current < addition) {
            throw new ArithmeticException("Missing-position count overflow.");
        }
        return current + addition;
    }

    private static int checkedPositionCount(long count) {
        if (count > Integer.MAX_VALUE) {
            throw new IllegalArgumentException(
                    "MissingIndices currently supports at most "
                            + Integer.MAX_VALUE
                            + " stored missing positions or groups."
            );
        }
        return (int) count;
    }
}
