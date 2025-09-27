# UserDistances Maven Project

This project contains three example distance functions:
- EuclideanDistance
- NaiveDTWDistance
- SpectralDistance (uses Apache Commons Math)

Each class implements the DistanceFunction interface and can be used with PFGAP.

## Project Structure

Place all Java source files under `src/main/java/com/example/`:
- `DistanceFunction.java`
- `EuclideanDistance.java`
- `NaiveDTWDistance.java`
- `SpectralDistance.java`

The `pom.xml` file is located at the root of the project.

## How to Build

Make sure you have Maven installed. From the root of the project directory, run:

    mvn package

This will produce a fat JAR file in the `target/` directory named `userdistances-1.0.jar`.

## How to Use with PFGAP

To use these distances with your `PFGAP.jar` file via the Python wrapper, specify each distance separately in the `distances` argument like this:

    distances = [
        "javadistance:/path/to/userdistances-1.0.jar:com.example.EuclideanDistance",
        "javadistance:/path/to/userdistances-1.0.jar:com.example.NaiveDTWDistance",
        "javadistance:/path/to/userdistances-1.0.jar:com.example.SpectralDistance"
    ]

Make sure the path to the JAR file is correct and accessible from your Python environment.

## Using External Libraries

If your custom distance depends on a library that is not available from Maven Central, you can include it using one of the following methods:

### Method 1: Install the JAR into the Local Maven Repository

1. Place the `.jar` file somewhere accessible.
2. Run the following command to install it into your local Maven repository:

    mvn install:install-file -Dfile=/path/to/library.jar \
        -DgroupId=com.custom \
        -DartifactId=customlib \
        -Dversion=1.0 \
        -Dpackaging=jar

3. Add the dependency to your `pom.xml`:

    <dependency>
        <groupId>com.custom</groupId>
        <artifactId>customlib</artifactId>
        <version>1.0</version>
    </dependency>

This method makes the library available just like any Maven dependency.

### Method 2: Include the JAR in a `lib/` Folder and Reference It

1. Create a `lib/` folder in your project root.
2. Place the `.jar` file inside it.
3. Modify your `pom.xml` to include the library using `systemPath`:

    <dependency>
        <groupId>com.custom</groupId>
        <artifactId>customlib</artifactId>
        <version>1.0</version>
        <scope>system</scope>
        <systemPath>${project.basedir}/lib/library.jar</systemPath>
    </dependency>

Note: `systemPath` is discouraged in modern Maven usage but works for simple cases.

## Extending the Project

You can add more distance classes by implementing the `DistanceFunction` interface and placing them in the same package.
Use Maven to build a new fat JAR and reference the new classes in the Python wrapper.
