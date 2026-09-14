package trees;

import core.contracts.ObjectDataset;

import java.util.Arrays;
import java.util.Objects;

/**
 * Compact node-local mapping from dataset occurrences to distinct training
 * observation identities.
 *
 * <p>A bootstrap sample may contain the same training observation more than
 * once. Those occurrences must continue to contribute independently to split
 * statistics and child datasets, but candidate distance evaluation only needs
 * to materialize and route the underlying observation once. This class
 * separates those two concerns:</p>
 *
 * <ul>
 *     <li>Each local dataset row remains an occurrence.</li>
 *     <li>Rows with the same non-null stable dataset index share one distinct
 *         observation ordinal.</li>
 *     <li>Each distinct observation records one representative local position
 *         and its occurrence multiplicity.</li>
 * </ul>
 *
 * <p>The persistent representation uses primitive arrays. Construction uses a
 * temporary primitive open-addressed table and does not allocate boxed maps.
 * The temporary table is discarded when construction completes.</p>
 *
 * <p>If a dataset row has a null stable index, that row is conservatively
 * treated as a distinct observation. This preserves correctness for datasets
 * that do not provide stable identities, while disabling reuse for those
 * rows.</p>
 */
final class NodeObservationMultiplicity {

    private static final float HASH_LOAD_FACTOR = 0.80f;
    private static final int MINIMUM_HASH_CAPACITY = 4;
    private static final int MAXIMUM_HASH_CAPACITY = 1 << 30;

    private final int occurrenceCount;
    private final int distinctCount;
    private final int[] occurrenceToDistinct;
    private final int[] representativeLocalPositions;
    private final int[] multiplicities;
    private final int[] stableIdentities;
    private final boolean allIdentitiesStable;

    private NodeObservationMultiplicity(
            int occurrenceCount,
            int distinctCount,
            int[] occurrenceToDistinct,
            int[] representativeLocalPositions,
            int[] multiplicities,
            int[] stableIdentities,
            boolean allIdentitiesStable
    ) {
        this.occurrenceCount = occurrenceCount;
        this.distinctCount = distinctCount;
        this.occurrenceToDistinct = occurrenceToDistinct;
        this.representativeLocalPositions = representativeLocalPositions;
        this.multiplicities = multiplicities;
        this.stableIdentities = stableIdentities;
        this.allIdentitiesStable = allIdentitiesStable;
    }

    /**
     * Builds the multiplicity mapping for one node dataset.
     *
     * @param sample node-local dataset containing bootstrap occurrences
     * @return immutable node-local multiplicity mapping
     */
    static NodeObservationMultiplicity from(ObjectDataset sample) {
        Objects.requireNonNull(sample, "Node dataset cannot be null.");

        int occurrenceCount = sample.size();
        if (occurrenceCount < 0) {
            throw new IllegalArgumentException("Dataset size cannot be negative.");
        }
        if (occurrenceCount == 0) {
            return empty();
        }

        int[] occurrenceToDistinct = new int[occurrenceCount];
        int initialDistinctCapacity = Math.min(occurrenceCount, 1_024);
        int[] representativeLocalPositions = new int[initialDistinctCapacity];
        int[] multiplicities = new int[initialDistinctCapacity];
        int[] stableIdentities = new int[initialDistinctCapacity];

        IntOrdinalMap identityToDistinct = new IntOrdinalMap(
                expectedDistinctCount(occurrenceCount)
        );

        int distinctCount = 0;
        boolean allIdentitiesStable = true;

        for (int localPosition = 0;
             localPosition < occurrenceCount;
             localPosition++) {
            Integer stableIdentity = sample.get_index(localPosition);

            if (stableIdentity == null) {
                allIdentitiesStable = false;
                if (distinctCount == representativeLocalPositions.length) {
                    int newCapacity = growCapacity(
                            representativeLocalPositions.length,
                            occurrenceCount
                    );
                    representativeLocalPositions = Arrays.copyOf(
                            representativeLocalPositions,
                            newCapacity
                    );
                    multiplicities = Arrays.copyOf(multiplicities, newCapacity);
                    stableIdentities = Arrays.copyOf(stableIdentities, newCapacity);
                }

                int distinctOrdinal = distinctCount++;
                occurrenceToDistinct[localPosition] = distinctOrdinal;
                representativeLocalPositions[distinctOrdinal] = localPosition;
                multiplicities[distinctOrdinal] = 1;
                // A null identity is represented by its local position only for
                // deterministic diagnostics. It is never inserted in the map.
                stableIdentities[distinctOrdinal] = localPosition;
                continue;
            }

            int identity = stableIdentity;
            int existingOrdinal = identityToDistinct.get(identity);
            if (existingOrdinal >= 0) {
                occurrenceToDistinct[localPosition] = existingOrdinal;
                multiplicities[existingOrdinal]++;
                continue;
            }

            if (distinctCount == representativeLocalPositions.length) {
                int newCapacity = growCapacity(
                        representativeLocalPositions.length,
                        occurrenceCount
                );
                representativeLocalPositions = Arrays.copyOf(
                        representativeLocalPositions,
                        newCapacity
                );
                multiplicities = Arrays.copyOf(multiplicities, newCapacity);
                stableIdentities = Arrays.copyOf(stableIdentities, newCapacity);
            }

            int distinctOrdinal = distinctCount++;
            identityToDistinct.put(identity, distinctOrdinal);
            occurrenceToDistinct[localPosition] = distinctOrdinal;
            representativeLocalPositions[distinctOrdinal] = localPosition;
            multiplicities[distinctOrdinal] = 1;
            stableIdentities[distinctOrdinal] = identity;
        }

        return new NodeObservationMultiplicity(
                occurrenceCount,
                distinctCount,
                occurrenceToDistinct,
                Arrays.copyOf(representativeLocalPositions, distinctCount),
                Arrays.copyOf(multiplicities, distinctCount),
                Arrays.copyOf(stableIdentities, distinctCount),
                allIdentitiesStable
        );
    }

    private static NodeObservationMultiplicity empty() {
        return new NodeObservationMultiplicity(
                0,
                0,
                new int[0],
                new int[0],
                new int[0],
                new int[0],
                true
        );
    }

    int occurrenceCount() {
        return occurrenceCount;
    }

    int distinctCount() {
        return distinctCount;
    }

    int distinctOrdinalAtOccurrence(int localPosition) {
        checkIndex(localPosition, occurrenceCount, "occurrence");
        return occurrenceToDistinct[localPosition];
    }

    int representativeLocalPosition(int distinctOrdinal) {
        checkIndex(distinctOrdinal, distinctCount, "distinct observation");
        return representativeLocalPositions[distinctOrdinal];
    }

    int multiplicity(int distinctOrdinal) {
        checkIndex(distinctOrdinal, distinctCount, "distinct observation");
        return multiplicities[distinctOrdinal];
    }

    int stableIdentity(int distinctOrdinal) {
        checkIndex(distinctOrdinal, distinctCount, "distinct observation");
        return stableIdentities[distinctOrdinal];
    }

    boolean allIdentitiesStable() {
        return allIdentitiesStable;
    }

    long reusedOccurrenceCount() {
        return (long) occurrenceCount - distinctCount;
    }

    boolean hasRepeatedIdentities() {
        return distinctCount < occurrenceCount;
    }

    /**
     * Package-private zero-copy access for splitter hot loops. The returned
     * array is owned by this immutable mapping and must not be modified.
     */
    int[] occurrenceToDistinctUnsafe() {
        return occurrenceToDistinct;
    }

    /**
     * Package-private zero-copy access for splitter hot loops. The returned
     * array is owned by this immutable mapping and must not be modified.
     */
    int[] representativeLocalPositionsUnsafe() {
        return representativeLocalPositions;
    }

    /**
     * Package-private zero-copy access for weighted accumulation. The returned
     * array is owned by this immutable mapping and must not be modified.
     */
    int[] multiplicitiesUnsafe() {
        return multiplicities;
    }

    /**
     * Package-private zero-copy access for deterministic per-observation seeds.
     * The returned array is owned by this immutable mapping and must not be
     * modified.
     */
    int[] stableIdentitiesUnsafe() {
        return stableIdentities;
    }

    private static int expectedDistinctCount(int occurrenceCount) {
        // A size-N bootstrap sample contains approximately
        // N * (1 - exp(-1)) distinct observations. The estimate only controls
        // temporary table sizing; the table grows if the node contains more.
        long estimate = Math.round(occurrenceCount * 0.6321205588285577d);
        return (int) Math.max(1L, Math.min((long) occurrenceCount, estimate));
    }

    private static int growCapacity(int currentCapacity, int maximumCapacity) {
        if (currentCapacity >= maximumCapacity) {
            throw new IllegalStateException(
                    "Distinct-observation storage cannot grow beyond dataset size."
            );
        }
        long doubled = Math.max(1L, (long) currentCapacity * 2L);
        return (int) Math.min((long) maximumCapacity, doubled);
    }

    private static void checkIndex(int index, int length, String kind) {
        if (index < 0 || index >= length) {
            throw new IndexOutOfBoundsException(
                    kind + " index " + index + " is outside [0, " + length + ")."
            );
        }
    }

    /**
     * Primitive int-to-nonnegative-ordinal map using linear probing.
     *
     * <p>The value table stores ordinal + 1, reserving zero as the empty-slot
     * marker. This permits every int value, including zero and Integer.MIN_VALUE,
     * to be used as an observation identity without a separate occupancy
     * array.</p>
     */
    private static final class IntOrdinalMap {
        private int[] keys;
        private int[] encodedOrdinals;
        private int mask;
        private int resizeThreshold;
        private int size;

        private IntOrdinalMap(int expectedSize) {
            int capacity = tableCapacityFor(expectedSize);
            keys = new int[capacity];
            encodedOrdinals = new int[capacity];
            mask = capacity - 1;
            resizeThreshold = resizeThreshold(capacity);
        }

        private int get(int key) {
            int slot = mix(key) & mask;
            while (true) {
                int encodedOrdinal = encodedOrdinals[slot];
                if (encodedOrdinal == 0) {
                    return -1;
                }
                if (keys[slot] == key) {
                    return encodedOrdinal - 1;
                }
                slot = (slot + 1) & mask;
            }
        }

        private void put(int key, int ordinal) {
            if (ordinal < 0 || ordinal == Integer.MAX_VALUE) {
                throw new IllegalArgumentException(
                        "Distinct ordinal must be within [0, Integer.MAX_VALUE)."
                );
            }
            if (size >= resizeThreshold) {
                resize();
            }
            insertWithoutResize(key, ordinal + 1);
        }

        private void insertWithoutResize(int key, int encodedOrdinal) {
            int slot = mix(key) & mask;
            while (encodedOrdinals[slot] != 0) {
                if (keys[slot] == key) {
                    throw new IllegalStateException(
                            "Stable observation identity " + key + " was inserted twice."
                    );
                }
                slot = (slot + 1) & mask;
            }
            keys[slot] = key;
            encodedOrdinals[slot] = encodedOrdinal;
            size++;
        }

        private void resize() {
            int oldCapacity = encodedOrdinals.length;
            if (oldCapacity >= MAXIMUM_HASH_CAPACITY) {
                throw new IllegalStateException(
                        "Too many distinct observation identities for the primitive map."
                );
            }

            int newCapacity = oldCapacity << 1;
            int[] oldKeys = keys;
            int[] oldEncodedOrdinals = encodedOrdinals;

            keys = new int[newCapacity];
            encodedOrdinals = new int[newCapacity];
            mask = newCapacity - 1;
            resizeThreshold = resizeThreshold(newCapacity);
            size = 0;

            for (int slot = 0; slot < oldCapacity; slot++) {
                int encodedOrdinal = oldEncodedOrdinals[slot];
                if (encodedOrdinal != 0) {
                    insertWithoutResize(oldKeys[slot], encodedOrdinal);
                }
            }
        }

        private static int tableCapacityFor(int expectedSize) {
            long required = Math.max(
                    MINIMUM_HASH_CAPACITY,
                    (long) Math.ceil(expectedSize / (double) HASH_LOAD_FACTOR)
            );
            if (required > MAXIMUM_HASH_CAPACITY) {
                throw new IllegalArgumentException(
                        "Expected distinct observation count is too large: "
                                + expectedSize + "."
                );
            }

            int capacity = MINIMUM_HASH_CAPACITY;
            while (capacity < required) {
                capacity <<= 1;
            }
            return capacity;
        }

        private static int resizeThreshold(int capacity) {
            return Math.max(1, (int) (capacity * HASH_LOAD_FACTOR));
        }

        private static int mix(int value) {
            int mixed = value;
            mixed ^= mixed >>> 16;
            mixed *= 0x7feb352d;
            mixed ^= mixed >>> 15;
            mixed *= 0x846ca68b;
            mixed ^= mixed >>> 16;
            return mixed;
        }
    }
}
