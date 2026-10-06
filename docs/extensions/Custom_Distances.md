# Custom Distances

PFGAP supports custom distance functions implemented in Java, Python, or Maple, as well as meta distances backed by pretrained models or prediction files. Java distances can be distributed as a `.class` file or a separate JAR, including a fat JAR when external dependencies must travel with the implementation.

This guide covers the extension contracts, descriptor syntax, packaging, runtime configuration, dimension selection, deferred data access, external dependencies, model persistence, and common errors.

For built-in distance identifiers and compatibility guidance, see [Distances](../reference/Distances.md).

### Choose an extension type

Use a custom Java distance when:

- the implementation should run in the JVM;
- low invocation overhead matters;
- the distance needs direct access to PFGAP observation objects;
- dimension-subset support is required; or
- the implementation should be packaged as a reusable JAR.

Use a Python or Maple distance when:

- an existing implementation already uses that language;
- the required ecosystem is outside the JVM; or
- development convenience matters more than interlanguage call overhead.

Use a meta distance when observations should be compared through predictions from a pretrained classifier or regressor rather than directly in the original feature space.

## Java distances

### Basic interface

A basic custom Java distance implements:

```java
package distance.api;

public interface DistanceFunction {
    double compute(Object first, Object second);
}
```

The method receives the logical observation objects produced by the configured reader and returns one finite distance value or positive infinity where the algorithm intentionally abandons a comparison.

The implementation is responsible for validating and casting the input representation it supports.

### Minimal numeric-vector example

```java
package example.distance;

import distance.api.DistanceFunction;

public final class EuclideanDistance
        implements DistanceFunction {

    public EuclideanDistance() {
    }

    @Override
    public double compute(
            Object first,
            Object second
    ) {
        double[] a = (double[]) first;
        double[] b = (double[]) second;

        if (a.length != b.length) {
            throw new IllegalArgumentException(
                    "EuclideanDistance requires equal lengths."
            );
        }

        double sum = 0.0;

        for (int index = 0;
             index < a.length;
             index++) {
            double difference =
                    a[index] - b[index];

            sum += difference * difference;
        }

        return Math.sqrt(sum);
    }
}
```

The class uses a public no-argument constructor so PFGAP can instantiate it reflectively.

### Supporting float and double vectors

A custom distance can accept multiple primitive representations explicitly:

```java
package example.distance;

import distance.api.DistanceFunction;

public final class NumericEuclideanDistance
        implements DistanceFunction {

    public NumericEuclideanDistance() {
    }

    @Override
    public double compute(
            Object first,
            Object second
    ) {
        if (first instanceof double[] a
                && second instanceof double[] b) {
            return computeDouble(a, b);
        }

        if (first instanceof float[] a
                && second instanceof float[] b) {
            return computeFloat(a, b);
        }

        throw new IllegalArgumentException(
                "Expected matching float[] or double[] inputs."
        );
    }

    private static double computeDouble(
            double[] a,
            double[] b
    ) {
        requireEqualLength(a.length, b.length);

        double sum = 0.0;
        for (int index = 0;
             index < a.length;
             index++) {
            double difference = a[index] - b[index];
            sum += difference * difference;
        }
        return Math.sqrt(sum);
    }

    private static double computeFloat(
            float[] a,
            float[] b
    ) {
        requireEqualLength(a.length, b.length);

        double sum = 0.0;
        for (int index = 0;
             index < a.length;
             index++) {
            double difference =
                    (double) a[index] - b[index];
            sum += difference * difference;
        }
        return Math.sqrt(sum);
    }

    private static void requireEqualLength(
            int firstLength,
            int secondLength
    ) {
        if (firstLength != secondLength) {
            throw new IllegalArgumentException(
                    "Distance requires equal lengths."
            );
        }
    }
}
```

Do not assume that `numeric_storage=auto` always produces `double[]`. Source formats such as NPY can preserve `float32` data as `float[]` or `float[][]`.

### Multivariate example

A multivariate distance receives a two-dimensional observation when the reader produces one:

```java
package example.distance;

import distance.api.DistanceFunction;

public final class IndependentManhattanDistance
        implements DistanceFunction {

    public IndependentManhattanDistance() {
    }

    @Override
    public double compute(
            Object first,
            Object second
    ) {
        double[][] a = (double[][]) first;
        double[][] b = (double[][]) second;

        if (a.length != b.length) {
            throw new IllegalArgumentException(
                    "Dimension counts must match."
            );
        }

        double total = 0.0;

        for (int dimension = 0;
             dimension < a.length;
             dimension++) {
            if (a[dimension].length
                    != b[dimension].length) {
                throw new IllegalArgumentException(
                        "Dimension lengths must match."
                );
            }

            for (int time = 0;
                 time < a[dimension].length;
                 time++) {
                total += Math.abs(
                        a[dimension][time]
                                - b[dimension][time]
                );
            }
        }

        return total;
    }
}
```

Select a representation contract and document it with the custom implementation. A custom distance is not automatically compatible with every reader or array type.

## Dimension-selectable Java distances

When node-level dimension subsampling is enabled, a custom Java distance must implement the dimension-selection-aware interface.

Conceptually, the interface extends the ordinary custom-distance contract with a computation that receives the selected dimension indices:

```java
package distance.api;

public interface DimensionSelectableDistanceFunction
        extends DistanceFunction {

    double compute(
            Object first,
            Object second,
            int[] selectedDimensions
    );
}
```

A representative implementation is:

```java
package example.distance;

import distance.api.DimensionSelectableDistanceFunction;

public final class SelectedDimensionEuclidean
        implements DimensionSelectableDistanceFunction {

    public SelectedDimensionEuclidean() {
    }

    @Override
    public double compute(
            Object first,
            Object second
    ) {
        double[][] a = (double[][]) first;
        double[][] b = (double[][]) second;

        int[] allDimensions = new int[a.length];
        for (int index = 0;
             index < allDimensions.length;
             index++) {
            allDimensions[index] = index;
        }

        return compute(first, second, allDimensions);
    }

    @Override
    public double compute(
            Object first,
            Object second,
            int[] selectedDimensions
    ) {
        double[][] a = (double[][]) first;
        double[][] b = (double[][]) second;

        double sum = 0.0;

        for (int dimension : selectedDimensions) {
            if (dimension < 0
                    || dimension >= a.length
                    || dimension >= b.length) {
                throw new IllegalArgumentException(
                        "Invalid selected dimension: "
                                + dimension
                );
            }

            if (a[dimension].length
                    != b[dimension].length) {
                throw new IllegalArgumentException(
                        "Dimension lengths must match."
                );
            }

            for (int time = 0;
                 time < a[dimension].length;
                 time++) {
                double difference =
                        a[dimension][time]
                                - b[dimension][time];

                sum += difference * difference;
            }
        }

        return Math.sqrt(sum);
    }
}
```

If dimension subsampling is enabled and a custom Java distance does not implement the selection-aware contract, PFGAP rejects the selected-dimension comparison rather than silently using all dimensions.

For a one-dimensional tabular vector, selected dimensions correspond to feature positions. Do not use dimension subsampling to select time positions in an ordinary univariate series unless that is the intended data contract.

## Java distances and deferred observations

PFGAP normally resolves deferred observation references before invoking a basic custom Java distance. The `DistanceFunction` implementation then receives the materialized observations.

For specialized behavior, a custom implementation can use the deferred-distance API:

```java
package distance.api;

public interface LazyDistanceFunction {
    double compute(
            Object firstStoredValue,
            Object secondStoredValue,
            /* resolver supplied by PFGAP */
    );
}
```

Use the exact interface in the matching PFGAP JAR when implementing this extension. It allows the custom distance to control when stored values are resolved through the provided resolver.

A selection-aware deferred custom-distance interface is not currently part of the public contract. When dimension subsampling is required, use materialized observations and implement `DimensionSelectableDistanceFunction`.

## Missing values

A custom distance must define its own missing-value behavior.

For numeric observations, missing values are represented by primitive `NaN`. An implementation can:

- reject inputs containing `NaN`;
- skip coordinates missing from either observation;
- rescale by the observed-coordinate ratio;
- use a missing-aware alignment method; or
- apply another documented rule.

Do not return `NaN` as a distance. PFGAP rejects `NaN` distance results.

Positive infinity is permitted when it is an intentional algorithm result, including early abandonment.

A custom distance is not automatically accepted by `missing_proximity_distances`. Proximity-first initialization currently accepts only the registered built-in missing-compatible identifiers documented in [Distances](../reference/Distances.md).

## Compile a Java distance

Assume this layout:

```text
custom-distance/
└── src/
    └── example/
        └── distance/
            └── EuclideanDistance.java
```

Compile against the same PFGAP JAR revision used at runtime:

```bash
mkdir -p out

javac \
  -cp /opt/pfgap/PFGAP.jar \
  -d out \
  src/example/distance/EuclideanDistance.java
```

This creates:

```text
out/example/distance/EuclideanDistance.class
```

A compiled class can be used directly when its loading arrangement is supported, but a JAR is normally easier to distribute and archive.

## Build a standard JAR

Package the compiled class:

```bash
jar --create \
  --file user-distances.jar \
  -C out .
```

The JAR contains:

```text
example/distance/EuclideanDistance.class
```

The corresponding PFGAP descriptor is:

```text
javadistance:/absolute/path/to/user-distances.jar:example.distance.EuclideanDistance
```

One JAR can contain multiple distance classes. Add one descriptor per class to the configured distance list.

## Build with Maven

A minimal Maven project is:

```text
custom-distance/
├── pom.xml
└── src/
    └── main/
        └── java/
            └── example/
                └── distance/
                    ├── EuclideanDistance.java
                    └── SpectralDistance.java
```

A basic `pom.xml` can compile against a local PFGAP JAR:

```xml
<project xmlns="http://maven.apache.org/POM/4.0.0"
         xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
         xsi:schemaLocation="http://maven.apache.org/POM/4.0.0
         https://maven.apache.org/xsd/maven-4.0.0.xsd">
    <modelVersion>4.0.0</modelVersion>

    <groupId>example</groupId>
    <artifactId>user-distances</artifactId>
    <version>1.0.0</version>

    <properties>
        <maven.compiler.release>21</maven.compiler.release>
        <project.build.sourceEncoding>
            UTF-8
        </project.build.sourceEncoding>
    </properties>

    <dependencies>
        <dependency>
            <groupId>local.pfgap</groupId>
            <artifactId>pfgap</artifactId>
            <version>1.0.0</version>
            <scope>system</scope>
            <systemPath>
                ${project.basedir}/lib/PFGAP.jar
            </systemPath>
        </dependency>
    </dependencies>
</project>
```

Place `PFGAP.jar` under `lib/`, then build:

```bash
mvn clean package
```

The ordinary JAR is created under `target/`.

`systemPath` is convenient for a local example but is not Maven's preferred dependency-management model. A reusable development setup can install or publish the PFGAP API artifact to a Maven repository instead.

## Build a fat JAR

A fat JAR is useful when a custom distance depends on third-party libraries that should be distributed with it.

The same principle applies to custom reader JARs. It is packaging guidance rather than a distance-specific runtime requirement.

Add the third-party dependency and Maven Shade Plugin:

```xml
<dependencies>
    <dependency>
        <groupId>local.pfgap</groupId>
        <artifactId>pfgap</artifactId>
        <version>1.0.0</version>
        <scope>system</scope>
        <systemPath>
            ${project.basedir}/lib/PFGAP.jar
        </systemPath>
    </dependency>

    <dependency>
        <groupId>org.apache.commons</groupId>
        <artifactId>commons-math3</artifactId>
        <version>3.6.1</version>
    </dependency>
</dependencies>

<build>
    <plugins>
        <plugin>
            <groupId>org.apache.maven.plugins</groupId>
            <artifactId>maven-shade-plugin</artifactId>
            <version>3.6.0</version>
            <executions>
                <execution>
                    <phase>package</phase>
                    <goals>
                        <goal>shade</goal>
                    </goals>
                    <configuration>
                        <finalName>
                            user-distances-all
                        </finalName>
                        <createDependencyReducedPom>
                            false
                        </createDependencyReducedPom>
                        <artifactSet>
                            <excludes>
                                <exclude>
                                    local.pfgap:pfgap
                                </exclude>
                            </excludes>
                        </artifactSet>
                    </configuration>
                </execution>
            </executions>
        </plugin>
    </plugins>
</build>
```

Build with:

```bash
mvn clean package
```

The shaded artifact is:

```text
target/user-distances-all.jar
```

Use it as:

```text
javadistance:/absolute/path/to/user-distances-all.jar:example.distance.SpectralDistance
```

Do not shade PFGAP's own API classes into the extension JAR. The runtime should use the interfaces supplied by the active PFGAP JAR.

If another dependency is not available from a configured repository, install it into a local Maven repository or use an appropriate organizational repository. Keep the complete dependency and version record with the experiment.

## Configure Java distances

### Python training example

```python
import PF_wrapper as PF

status = PF.train(
    train_file="../data/train.csv",
    test_file="../data/test.csv",
    distances=[
        (
            "javadistance:/opt/pfgap/"
            "user-distances.jar:"
            "example.distance.EuclideanDistance"
        ),
        (
            "javadistance:/opt/pfgap/"
            "user-distances-all.jar:"
            "example.distance.SpectralDistance"
        ),
    ],
    forest_mode="classification",
    data_dimension=1,
    numeric_data=True,
    numeric_storage="float64",
    output_directory="../output/custom_distances",
)

if status != 0:
    raise SystemExit(status)
```

### Direct Java example

```bash
java -Xmx4g -jar Application/PFGAP.jar \
  -eval=false \
  -train=data/train.csv \
  '-distances=[javadistance:/opt/pfgap/user-distances.jar:example.distance.EuclideanDistance,javadistance:/opt/pfgap/user-distances-all.jar:example.distance.SpectralDistance]' \
  -forest_mode=classification \
  -is2D=false \
  -isNumeric=true \
  -numeric_storage=float64 \
  -out=output/custom_distances/
```

Quote the list argument when required by the shell.

### Use built-in and custom distances together

```python
distances=[
    "dtw",
    "erp",
    (
        "javadistance:/opt/pfgap/"
        "user-distances.jar:"
        "example.distance.EuclideanDistance"
    ),
]
```

PFGAP selects among the configured candidates according to the ordinary splitter configuration.

## Python distances

The descriptor form is:

```text
python:/absolute/path/to/distance_file.py[:function_name]
```

Example:

```text
python:/opt/pfgap/distances.py:euclidean_distance
```

A Python distance receives the observations through PFGAP's configured Python interoperation layer and returns one numeric distance.

A representative Python function is:

```python
import math


def euclidean_distance(first, second):
    if len(first) != len(second):
        raise ValueError("Lengths must match")

    total = 0.0
    for a, b in zip(first, second):
        difference = float(a) - float(b)
        total += difference * difference

    return math.sqrt(total)
```

Configure it with:

```python
distances=[
    "python:/opt/pfgap/distances.py:euclidean_distance"
]
```

The script path must exist when the command-line distance descriptors are parsed. Preserve the Python runtime, packages, script, and function name with the experiment.

## Maple distances

The descriptor form is:

```text
maple:/absolute/path/to/distance_file.mpl[:function_name]
```

Example:

```text
maple:/opt/pfgap/distances.mpl:EuclideanDistance
```

Configure it with:

```python
distances=[
    "maple:/opt/pfgap/distances.mpl:EuclideanDistance"
]
```

The Maple source file, selected function, compatible Maple installation, and required packages are external model dependencies.

## Meta distances

Meta distances compare observations through outputs from a pretrained model or prediction source.

Registered descriptor types are:

```text
meta_classmatch
meta_file_classmatch
meta_regression
meta_file_regression
```

### Classification match

```text
meta_classmatch:/path/to/model_or_function[:method]
```

This family compares observations according to predicted classes or class-oriented pretrained-model behavior.

### File-backed classification match

```text
meta_file_classmatch:/path/to/predictions[:method]
```

This family uses a file-backed prediction source.

### Regression

```text
meta_regression:/path/to/model_or_function[:method]
```

This family compares numeric predictions from a pretrained regressor or regression function.

### File-backed regression

```text
meta_file_regression:/path/to/predictions[:method]
```

This family uses file-backed numeric predictions.

The referenced file must exist when descriptors are parsed. Retain prediction ordering and observation identity so values remain aligned with PFGAP instances.

## Parameters and state

PFGAP's built-in elastic distances randomize candidate parameters through `DistanceMeasure`. A basic custom Java, Python, or Maple distance does not automatically participate in that built-in parameter-randomization machinery.

A custom implementation can:

- use fixed internal parameters;
- read parameters from its packaged resources;
- derive deterministic parameters from its inputs;
- use configuration supplied by an external implementation; or
- expose several differently configured classes or descriptors.

Avoid hidden nondeterministic parameter selection unless the implementation provides its own reproducibility contract.

## KNN imputation

A custom distance descriptor can be supplied through `knn_distances` when the custom implementation supports the data representation used by the KNN initial imputer.

Example:

```python
initial_imputer="knn"
knn_distances=[
    (
        "javadistance:/opt/pfgap/"
        "user-distances.jar:"
        "example.distance.EuclideanDistance"
    )
]
```

The current KNN initializer uses five neighbors.

Custom distances are not accepted automatically as missing-aware proximity-first distances. Use the registered missing-compatible identifiers for `missing_proximity_distances`.

## Model persistence

A saved PFGAP model retains the selected forest state and custom-distance descriptors, but external implementations remain external dependencies.

### Java JARs

Before deserializing a model, PFGAP examines active `javadistance` descriptors and adds existing JAR paths to a custom class loader. Configure the descriptors and keep the required JARs available when loading the model.

If the custom class is unavailable, deserialization can fail with a class-resolution error.

### Python and Maple

The model does not embed a Python or Maple runtime, package environment, or source file. Preserve the interpreter installation, packages, scripts, and selected function names.

### Meta distances

Preserve the pretrained implementation, prediction files, method selection, observation ordering, and any external model artifacts.

See [Model Persistence](../reference/Model_Persistence.md).

## Reproducibility

Retain:

- the PFGAP JAR and revision;
- custom distance source revision;
- compiled class or JAR checksum;
- external dependency versions;
- exact descriptor strings;
- Java, Python, or Maple runtime versions;
- input representation and numeric storage;
- standardization and missing-value configuration;
- seed and worker count; and
- any external pretrained model or prediction artifacts.

A custom implementation can have its own parallelism or random-number behavior. Document and preserve those settings as part of the extension.

## Performance

Java distances run in the PFGAP process and normally have the lowest interoperation overhead.

Python and Maple distances cross a language boundary and can be substantially more expensive when invoked repeatedly at tree splits. Measure performance on representative data.

Custom Java distances should avoid allocating large temporary arrays for every comparison. Reuse immutable precomputed state where safe, and keep mutable per-call state local when comparisons can run concurrently.

A custom distance used by multiple PFGAP workers must be safe under the runtime's evaluator-copy and invocation behavior. Do not depend on unsynchronized global mutable state.

## Security

Custom JARs, Python scripts, Maple code, and meta-model integrations execute code or consume external artifacts under the PFGAP process's permissions.

Use only trusted extensions and dependencies. Review the implementation before loading it, particularly when a saved model references external custom code.

## Common problems

### The Java class cannot be found

Check:

- the JAR or class path;
- the fully qualified class name;
- the public no-argument constructor;
- package spelling and capitalization; and
- compatibility with the active PFGAP API.

### The implementation does not implement `DistanceFunction`

Compile against `distance.api.DistanceFunction` from the matching PFGAP JAR and implement its exact method signature.

### A dependency class cannot be found

Use a fat JAR or otherwise provide the dependency to the extension class loader. Do not package duplicate PFGAP API classes into the extension JAR.

### The distance receives an unexpected array type

Match the reader, dimensionality, numeric storage, and custom implementation. Support both `float` and `double` forms explicitly when needed.

### Dimension subsampling fails

Implement `DimensionSelectableDistanceFunction`, or disable dimension subsampling for that custom distance.

### A deferred dataset passes references unexpectedly

A basic custom distance normally receives materialized observations. If the implementation intentionally needs stored values and resolver control, implement the matching `LazyDistanceFunction` interface from the active PFGAP JAR.

### The distance returns `NaN`

Correct the algorithm, inputs, missing-value policy, or numerical edge case. `NaN` is not an accepted distance result.

### Proximity-first imputation rejects the custom distance

Use one of the registered built-in missing-compatible distances for `missing_proximity_distances`.

### A Python or Maple source file is not found

Use a path that resolves from the process working directory, or prefer an absolute path.

### A saved model no longer loads

Restore the original custom JARs, classes, scripts, dependencies, and descriptor paths. Use the PFGAP revision that created the model.

### Results differ between machines

Compare runtime versions, dependency versions, numeric storage, extension source, random behavior, worker count, preprocessing, and external model artifacts.

### Related documentation

- [Distances](../reference/Distances.md)
- [Configuration Reference](../reference/Configuration_Reference.md)
- [CLI Reference](../reference/CLI_Reference.md)
- [Dataset Representations](../data/Dataset_Representations.md)
- [Readers](../data/Readers.md)
- [Missing Values](../reference/Missing_Values.md)
- [Model Persistence](../reference/Model_Persistence.md)
- [Parallelism and Reproducibility](../guides/Parallelism_and_Reproducibility.md)
