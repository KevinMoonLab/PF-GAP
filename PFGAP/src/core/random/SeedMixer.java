package core.random;

/**
 * Deterministic seed derivation for scheduling-independent PFGAP randomness.
 *
 * <p>This utility derives reproducible child seeds from stable logical
 * identities such as forest, tree, node path, candidate index, operation
 * purpose, and instance index. Results depend only on the supplied values and
 * never on task execution order, worker identity, thread identity, or mutable
 * random-number-generator state.</p>
 *
 * <p>The mixer uses the SplitMix64 finalization sequence to provide strong bit
 * diffusion. It is intended for deterministic seed derivation and randomized
 * machine-learning algorithms. It is not a cryptographic primitive and must
 * not be used for passwords, authentication tokens, or other security-sensitive
 * randomness.</p>
 */
public final class SeedMixer {

    /** Odd Weyl-sequence increment used to separate adjacent input values. */
    private static final long GOLDEN_GAMMA =
            0x9E3779B97F4A7C15L;

    private static final long MIX_MULTIPLIER_1 =
            0xBF58476D1CE4E5B9L;

    private static final long MIX_MULTIPLIER_2 =
            0x94D049BB133111EBL;

    private SeedMixer() {
        throw new AssertionError(
                "SeedMixer cannot be instantiated."
        );
    }

    /**
     * Applies a high-diffusion 64-bit finalizer to one value.
     *
     * @param value value to mix
     * @return deterministically mixed value
     */
    public static long mix64(long value) {
        long mixed =
                value;

        mixed =
                (mixed ^ (mixed >>> 30))
                        * MIX_MULTIPLIER_1;

        mixed =
                (mixed ^ (mixed >>> 27))
                        * MIX_MULTIPLIER_2;

        return mixed ^ (mixed >>> 31);
    }

    /**
     * Derives one child seed from a parent seed and a 64-bit discriminator.
     *
     * <p>The discriminator should identify a stable logical choice, such as a
     * branch index, candidate index, purpose constant, or instance identity.</p>
     *
     * @param parentSeed parent or root seed
     * @param discriminator stable logical discriminator
     * @return derived child seed
     */
    public static long derive(
            long parentSeed,
            long discriminator
    ) {
        return mix64(
                parentSeed
                        ^ mix64(
                                discriminator + GOLDEN_GAMMA
                        )
        );
    }

    /**
     * Derives one child seed from a parent seed and a signed integer
     * discriminator without losing its sign information.
     */
    public static long derive(
            long parentSeed,
            int discriminator
    ) {
        return derive(
                parentSeed,
                (long) discriminator
        );
    }

    /**
     * Derives a seed from an ordered sequence of discriminators.
     *
     * <p>Order is significant. For example, deriving from {@code [2, 7]} is
     * intentionally different from deriving from {@code [7, 2]}.</p>
     *
     * @param rootSeed root seed
     * @param discriminators ordered logical discriminators
     * @return derived seed, or {@code rootSeed} when no discriminators are
     *         supplied
     */
    public static long derive(
            long rootSeed,
            long... discriminators
    ) {
        if (discriminators == null) {
            throw new IllegalArgumentException(
                    "Seed discriminators cannot be null."
            );
        }

        long seed =
                rootSeed;

        for (long discriminator : discriminators) {
            seed =
                    derive(
                            seed,
                            discriminator
                    );
        }

        return seed;
    }

    /**
     * Derives a stable child-node path identity from its parent path and branch
     * index.
     *
     * @param parentPathIdentity parent node's stable path identity
     * @param branchIndex zero-based child branch index
     * @return stable child path identity
     */
    public static long childPath(
            long parentPathIdentity,
            int branchIndex
    ) {
        requireNonnegative(
                branchIndex,
                "branchIndex"
        );

        return derive(
                parentPathIdentity,
                branchIndex
        );
    }

    /**
     * Derives a purpose-specific seed for one node.
     *
     * <p>Purpose values should be stable named constants. Distinct operations
     * must use distinct purpose values so adding random draws to one operation
     * cannot perturb another operation's random stream.</p>
     */
    public static long nodePurpose(
            long treeSeed,
            long nodePathIdentity,
            long purpose
    ) {
        return derive(
                treeSeed,
                nodePathIdentity,
                purpose
        );
    }

    /**
     * Derives a candidate-specific seed from stable tree, node, purpose, and
     * candidate identities.
     */
    public static long candidate(
            long treeSeed,
            long nodePathIdentity,
            long purpose,
            int candidateIndex
    ) {
        requireNonnegative(
                candidateIndex,
                "candidateIndex"
        );

        return derive(
                treeSeed,
                nodePathIdentity,
                purpose,
                candidateIndex
        );
    }

    /**
     * Derives a stable per-instance seed from an operation seed and instance
     * identity.
     *
     * <p>The instance identity may be negative when an upstream dataset uses a
     * signed identifier. No nonnegative restriction is applied.</p>
     */
    public static long instance(
            long operationSeed,
            int instanceIdentity
    ) {
        return derive(
                operationSeed,
                instanceIdentity
        );
    }

    /**
     * Derives a stable prediction seed for one forest/tree/query combination.
     */
    public static long prediction(
            long forestSeed,
            int forestId,
            int treeIndex,
            int queryIndex
    ) {
        requireNonnegative(
                treeIndex,
                "treeIndex"
        );

        return derive(
                forestSeed,
                forestId,
                treeIndex,
                queryIndex
        );
    }

    private static void requireNonnegative(
            int value,
            String argumentName
    ) {
        if (value < 0) {
            throw new IllegalArgumentException(
                    argumentName
                            + " cannot be negative. Received: "
                            + value
                            + "."
            );
        }
    }
}
