package distance.api;

import datasets.readers.lazy.LazySeriesReader;

public interface LazyDistanceFunction extends DistanceFunction {

    double compute(
            Object t1,
            Object t2,
            LazySeriesReader reader
    );
}