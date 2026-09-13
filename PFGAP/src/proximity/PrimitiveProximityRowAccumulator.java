package proximity;

import java.util.Arrays;

/**
 * Reusable primitive open-addressing accumulator for one sparse proximity row.
 *
 * <p>The table stores target-column keys and accumulated {@code double} values
 * without boxing. Clearing is proportional to the number of retained entries,
 * not to the target-column count or table capacity, because only occupied slots
 * are recorded in {@code occupiedSlots}.</p>
 *
 * <p>This class is mutable and not thread-safe. One instance should belong to
 * one worker task and may be reused across source rows by calling
 * {@link #clear()} after the previous row has been exported.</p>
 *
 * <p>Iteration order is unspecified. CSR construction should export entries,
 * apply its configured sparsity threshold, and sort retained column indices
 * before committing a row.</p>
 */
public final class PrimitiveProximityRowAccumulator
        implements ProximityRowAccumulator {

    private static final int EMPTY_KEY =
            -1;

    private static final double DEFAULT_LOAD_FACTOR =
            0.65;

    private static final int MINIMUM_CAPACITY =
            16;

    private final int targetCount;
    private final double loadFactor;

    private int[] keys;
    private double[] values;
    private int[] occupiedSlots;

    private int size;
    private int resizeThreshold;

    /**
     * Creates an accumulator with a modest default expected row size.
     */
    public PrimitiveProximityRowAccumulator(
            int targetCount
    ) {
        this(
                targetCount,
                Math.min(
                        targetCount,
                        MINIMUM_CAPACITY
                )
        );
    }

    /**
     * Creates an accumulator sized for an expected number of retained entries.
     *
     * @param targetCount valid target-column count
     * @param expectedEntries expected number of explicitly retained targets
     */
    public PrimitiveProximityRowAccumulator(
            int targetCount,
            int expectedEntries
    ) {
        if (targetCount < 0) {
            throw new IllegalArgumentException(
                    "targetCount cannot be negative."
            );
        }

        if (expectedEntries < 0) {
            throw new IllegalArgumentException(
                    "expectedEntries cannot be negative."
            );
        }

        this.targetCount =
                targetCount;

        this.loadFactor =
                DEFAULT_LOAD_FACTOR;

        int boundedExpectedEntries =
                Math.min(
                        targetCount,
                        expectedEntries
                );

        allocateTable(
                capacityForExpectedEntries(
                        boundedExpectedEntries
                )
        );
    }

    @Override
    public void add(
            int targetIndex,
            double contribution
    ) {
        checkTargetIndex(targetIndex);
        checkFinite(contribution, "contribution");

        if (contribution == 0.0) {
            return;
        }

        int slot =
                findSlot(
                        targetIndex
                );

        if (keys[slot] == targetIndex) {
            double updatedValue =
                    values[slot]
                            + contribution;

            checkFinite(
                    updatedValue,
                    "accumulated value"
            );

            if (updatedValue == 0.0) {
                removeSlot(slot);
            } else {
                values[slot] =
                        updatedValue;
            }

            return;
        }

        ensureCapacityForInsertion();

        slot =
                findSlot(
                        targetIndex
                );

        insertAt(
                slot,
                targetIndex,
                contribution
        );
    }

    @Override
    public void set(
            int targetIndex,
            double value
    ) {
        checkTargetIndex(targetIndex);
        checkFinite(value, "value");

        int slot =
                findSlot(
                        targetIndex
                );

        if (keys[slot] == targetIndex) {
            if (value == 0.0) {
                removeSlot(slot);
            } else {
                values[slot] =
                        value;
            }

            return;
        }

        if (value == 0.0) {
            return;
        }

        ensureCapacityForInsertion();

        slot =
                findSlot(
                        targetIndex
                );

        insertAt(
                slot,
                targetIndex,
                value
        );
    }

    @Override
    public double get(
            int targetIndex
    ) {
        checkTargetIndex(targetIndex);

        int slot =
                findSlot(
                        targetIndex
                );

        return keys[slot] == targetIndex
                ? values[slot]
                : 0.0;
    }

    @Override
    public int targetCount() {
        return targetCount;
    }

    @Override
    public int retainedEntryCount() {
        return size;
    }

    @Override
    public void scale(
            double factor
    ) {
        checkFinite(factor, "factor");

        if (size == 0 || factor == 1.0) {
            return;
        }

        if (factor == 0.0) {
            clear();
            return;
        }

        int occupiedCount =
                size;

        for (int position = 0;
             position < occupiedCount;
             position++) {

            int slot =
                    occupiedSlots[position];

            double scaled =
                    values[slot]
                            * factor;

            checkFinite(
                    scaled,
                    "scaled value"
            );

            values[slot] =
                    scaled;
        }
    }

    @Override
    public void clear() {
        for (int position = 0;
             position < size;
             position++) {

            int slot =
                    occupiedSlots[position];

            keys[slot] =
                    EMPTY_KEY;

            values[slot] =
                    0.0;
        }

        size =
                0;
    }

    /**
     * Returns the retained target index at an unspecified entry position.
     *
     * <p>This allocation-free traversal API is intended for CSR builders in
     * the same package. Entries are not sorted by target index.</p>
     */
    int retainedTargetAt(
            int entryPosition
    ) {
        checkEntryPosition(entryPosition);

        return keys[
                occupiedSlots[entryPosition]
                ];
    }

    /**
     * Returns the retained value at an unspecified entry position.
     */
    double retainedValueAt(
            int entryPosition
    ) {
        checkEntryPosition(entryPosition);

        return values[
                occupiedSlots[entryPosition]
                ];
    }

    /**
     * Ensures capacity for at least the requested number of retained entries.
     *
     * <p>This can be used by a worker before processing a row when a leaf-based
     * estimate is available.</p>
     */
    public void ensureExpectedEntries(
            int expectedEntries
    ) {
        if (expectedEntries < 0) {
            throw new IllegalArgumentException(
                    "expectedEntries cannot be negative."
            );
        }

        int boundedExpectedEntries =
                Math.min(
                        targetCount,
                        expectedEntries
                );

        int requiredCapacity =
                capacityForExpectedEntries(
                        boundedExpectedEntries
                );

        if (requiredCapacity > keys.length) {
            rehash(
                    requiredCapacity
            );
        }
    }

    private void insertAt(
            int slot,
            int targetIndex,
            double value
    ) {
        keys[slot] =
                targetIndex;

        values[slot] =
                value;

        occupiedSlots[size] =
                slot;

        size++;
    }

    /**
     * Removes an entry and repairs its probe cluster.
     */
    private void removeSlot(
            int removedSlot
    ) {
        int removedPosition =
                occupiedPositionOf(
                        removedSlot
                );

        int lastPosition =
                size - 1;

        occupiedSlots[removedPosition] =
                occupiedSlots[lastPosition];

        size--;

        keys[removedSlot] =
                EMPTY_KEY;

        values[removedSlot] =
                0.0;

        int mask =
                keys.length - 1;

        int slot =
                (removedSlot + 1)
                        & mask;

        while (keys[slot] != EMPTY_KEY) {
            int keyToReinsert =
                    keys[slot];

            double valueToReinsert =
                    values[slot];

            int occupiedPosition =
                    occupiedPositionOf(
                            slot
                    );

            keys[slot] =
                    EMPTY_KEY;

            values[slot] =
                    0.0;

            int newSlot =
                    findSlot(
                            keyToReinsert
                    );

            keys[newSlot] =
                    keyToReinsert;

            values[newSlot] =
                    valueToReinsert;

            occupiedSlots[occupiedPosition] =
                    newSlot;

            slot =
                    (slot + 1)
                            & mask;
        }
    }

    private int occupiedPositionOf(
            int slot
    ) {
        for (int position = 0;
             position < size;
             position++) {

            if (occupiedSlots[position] == slot) {
                return position;
            }
        }

        throw new IllegalStateException(
                "Occupied slot "
                        + slot
                        + " is missing from the occupancy index."
        );
    }

    private void ensureCapacityForInsertion() {
        if (size + 1 <= resizeThreshold) {
            return;
        }

        if (keys.length >= maximumTableCapacity()) {
            if (size >= targetCount) {
                throw new IllegalStateException(
                        "Accumulator already contains every valid target."
                );
            }

            return;
        }

        rehash(
                keys.length << 1
        );
    }

    private void rehash(
            int requestedCapacity
    ) {
        int newCapacity =
                normalizeCapacity(
                        requestedCapacity
                );

        int[] oldKeys =
                keys;

        double[] oldValues =
                values;

        int[] oldOccupiedSlots =
                occupiedSlots;

        int oldSize =
                size;

        allocateTable(
                newCapacity
        );

        for (int position = 0;
             position < oldSize;
             position++) {

            int oldSlot =
                    oldOccupiedSlots[position];

            int key =
                    oldKeys[oldSlot];

            int newSlot =
                    findSlot(
                            key
                    );

            insertAt(
                    newSlot,
                    key,
                    oldValues[oldSlot]
            );
        }
    }

    private void allocateTable(
            int capacity
    ) {
        keys =
                new int[capacity];

        Arrays.fill(
                keys,
                EMPTY_KEY
        );

        values =
                new double[capacity];

        occupiedSlots =
                new int[capacity];

        size =
                0;

        resizeThreshold =
                Math.min(
                        capacity - 1,
                        Math.max(
                                1,
                                (int) Math.floor(
                                        capacity
                                                * loadFactor
                                )
                        )
                );
    }

    private int findSlot(
            int targetIndex
    ) {
        int mask =
                keys.length - 1;

        int slot =
                mix(
                        targetIndex
                ) & mask;

        while (keys[slot] != EMPTY_KEY
                && keys[slot] != targetIndex) {

            slot =
                    (slot + 1)
                            & mask;
        }

        return slot;
    }

    private int capacityForExpectedEntries(
            int expectedEntries
    ) {
        if (expectedEntries == 0) {
            return MINIMUM_CAPACITY;
        }

        long minimumCapacity =
                (long) Math.ceil(
                        expectedEntries
                                / loadFactor
                );

        if (minimumCapacity > maximumTableCapacity()) {
            throw new IllegalArgumentException(
                    "Expected sparse row size "
                            + expectedEntries
                            + " exceeds supported accumulator capacity."
            );
        }

        return normalizeCapacity(
                (int) minimumCapacity
        );
    }

    private int maximumTableCapacity() {
        return 1 << 30;
    }

    private static int normalizeCapacity(
            int requestedCapacity
    ) {
        int capacity =
                Math.max(
                        MINIMUM_CAPACITY,
                        requestedCapacity
                );

        if (capacity >= (1 << 30)) {
            return 1 << 30;
        }

        int highest =
                Integer.highestOneBit(
                        capacity - 1
                );

        return highest << 1;
    }

    private static int mix(
            int value
    ) {
        int mixed =
                value;

        mixed ^=
                mixed >>> 16;

        mixed *=
                0x7feb352d;

        mixed ^=
                mixed >>> 15;

        mixed *=
                0x846ca68b;

        return mixed
                ^ (mixed >>> 16);
    }

    private void checkEntryPosition(
            int entryPosition
    ) {
        if (entryPosition < 0 || entryPosition >= size) {
            throw new IndexOutOfBoundsException(
                    "Entry position "
                            + entryPosition
                            + " is outside [0, "
                            + size
                            + ")."
            );
        }
    }
}
