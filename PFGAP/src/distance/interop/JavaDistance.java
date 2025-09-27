package distance.interop;

import distance.api.DistanceFunction;

public class JavaDistance {
    private final DistanceFunction distanceInstance;

    public JavaDistance(String descriptor) throws Exception {
        this.distanceInstance = JavaDistanceLoader.load(descriptor);
    }

    public DistanceFunction getDistanceFunction() {
        return distanceInstance;
    }

    public double distance(Object t1, Object t2) {
        return distanceInstance.compute(t1, t2);
    }
}

/*
for the python wrapper, use
distance = ["javadistance:/path/to/MyCustomDistance.class"]
# or
distance = ["javadistance:/path/to/customdistances.jar:my.package.MyCustomDistance"]

 */