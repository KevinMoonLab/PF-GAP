package distance.api;

public interface DistanceFunction {

    double compute(Object t1, Object t2);

    /**
     * Computes the distance between two objects, optionally abandoning the
     * calculation when it can no longer produce a result below bestSoFar.
     *
     * <p>The default implementation ignores bestSoFar and performs the full
     * distance calculation. Custom distances that support early abandoning
     * may override this method.</p>
     *
     * @param t1 first object
     * @param t2 second object
     * @param bestSoFar current smallest competing distance
     * @return the exact distance when it is less than bestSoFar; otherwise,
     *         an implementation may return positive infinity after abandoning
     */
    default double compute(
            Object t1,
            Object t2,
            double bestSoFar
    ) {
        return compute(t1, t2);
    }
}