# Installation

This guide explains how to install and verify PFGAP for normal use, Python-assisted use, and source development. PFGAP runs on Java 21; Python and other external runtimes are optional and are required only for the interfaces or extensions that use them.

> **Version status:** PFGAP is under active development toward version 1.0. Use the `Application/PFGAP.jar` and `Application/PF_wrapper.py` files from the same revision. Mixing files from different revisions can expose incompatible options or behavior.

## Requirements

### Required

- A 64-bit operating system supported by a Java 21 runtime
- Java Development Kit (JDK) 21
- Sufficient memory and storage for the selected datasets, outputs, and execution mode

A Java Runtime Environment alone may be sufficient to launch the packaged application, but a full JDK is recommended and is required when building PFGAP or compiling Java extensions.

### Optional

Install these components only when the corresponding workflow requires them:

- Python 3 for `Application/PF_wrapper.py`, notebooks, Python-driven experiments, or Python-backed custom distances
- Python packages required by the user's own analysis, notebooks, or custom-distance implementation
- Maple for Maple-backed custom distances
- A Java IDE, such as IntelliJ IDEA, for source development
- Build and packaging tools required by the current source project when rebuilding `PFGAP.jar`

The core Java command-line application does not require Python, a notebook environment, or Maple.

## Install Java 21

Install a Java 21 JDK using the package manager or JDK distribution appropriate for the operating system. After installation, open a new terminal and verify the active runtime:

```bash
java -version
```

Verify the compiler as well if PFGAP will be built from source or extended with Java code:

```bash
javac -version
```

Both commands should report major version `21`. If they report another version, update `JAVA_HOME` and the system `PATH`, or invoke the intended Java installation explicitly.

### Verify `JAVA_HOME`

On macOS or Linux:

```bash
printf '%s\n' "$JAVA_HOME"
```

On Windows PowerShell:

```powershell
$env:JAVA_HOME
```

`JAVA_HOME` should identify the JDK installation directory, not its `bin` subdirectory. The `java` and `javac` executables selected by `PATH` should belong to that same installation.

## Obtain PFGAP

### Clone the repository

Cloning the repository is the recommended installation method for development, documentation, tests, examples, custom extensions, or reproducible work tied to a specific revision:

```bash
git clone https://github.com/KevinMoonLab/PF-GAP.git
cd PF-GAP
```

The files used for normal packaged execution are located in `Application/`:

```text
Application/
├── PFGAP.jar
└── PF_wrapper.py
```

Keep these files together unless paths are supplied explicitly. The Python helper launches the Java application; it is not an independent implementation of PFGAP.

### Download only the packaged application

For direct command-line use, `Application/PFGAP.jar` is the essential packaged artifact. For Python-assisted use, also obtain `Application/PF_wrapper.py` from the same repository revision.

When downloading files individually, preserve their exact names and place them in a common project directory, for example:

```text
my-pfgap-project/
├── Application/
│   ├── PFGAP.jar
│   └── PF_wrapper.py
├── data/
└── output/
```

A full repository clone is preferable when custom Java classes, documentation, tests, or source-level debugging are needed.

## Verify the packaged application

From the repository root, first confirm that Java can inspect the packaged JAR:

```bash
jar --describe-module --file Application/PFGAP.jar
```

A non-modular JAR may be reported as an automatic module. The important installation check is that the JAR can be opened without a version, corruption, or missing-file error.

PFGAP is launched with the following general command form:

```bash
java -jar Application/PFGAP.jar -name=value
```

A complete execution requires task and data options, so the absence of those options may produce usage information or a configuration error. That still confirms that Java found and started the application. Use the [Quick Start](Quick_Start.md) for the first complete run and the [CLI Reference](../reference/CLI_Reference.md) for the full option catalog.

## Install the optional Python interface

Python is required only when calling PFGAP through `PF_wrapper.py` or using a Python-backed extension.

### Verify Python

On systems where the command is `python3`:

```bash
python3 --version
```

On systems where Python 3 is available as `python`:

```bash
python --version
```

Use a currently supported Python 3 release. The exact third-party package requirements depend on the workflow. The wrapper's primary role is to construct and launch a Java command, while notebooks, data preparation, plotting, and custom Python distances may require additional packages.

### Create an isolated environment

Creating a virtual environment is recommended for Python-based workflows.

On macOS or Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

On Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

Upgrade the packaging tools after activation:

```bash
python -m pip install --upgrade pip setuptools wheel
```

Install only the packages required by the planned workflow. Typical analysis or demonstration environments may use NumPy, pandas, Matplotlib, scikit-learn, or aeon, but those packages are not all required by every PFGAP execution:

```bash
python -m pip install numpy pandas matplotlib scikit-learn aeon
```

Do not treat this broad demonstration set as a mandatory dependency list for the Java application.

### Make `PF_wrapper.py` importable

The simplest approach is to run Python from `Application/`:

```bash
cd Application
python
```

Then import the helper:

```python
import PF_wrapper as PF
```

Alternatively, add `Application/` to the Python module search path from the repository root:

On macOS or Linux:

```bash
export PYTHONPATH="$PWD/Application${PYTHONPATH:+:$PYTHONPATH}"
```

On Windows PowerShell:

```powershell
$env:PYTHONPATH = "$PWD\Application;$env:PYTHONPATH"
```

The helper must also be able to locate `PFGAP.jar`. Keeping the helper and JAR together avoids unnecessary path configuration.

## Running from another working directory

Relative paths are resolved from the process working directory unless a particular option states otherwise. When launching PFGAP outside the repository root, use absolute paths or paths that are valid from the current directory.

Direct Java invocation:

```bash
java -jar /path/to/PF-GAP/Application/PFGAP.jar -name=value
```

Windows PowerShell:

```powershell
java -jar C:\path\to\PF-GAP\Application\PFGAP.jar -name=value
```

The same principle applies to dataset locations, model paths, output directories, extension JARs, and other file-valued options.

## Build from source

Building from source is intended for contributors, developers changing the Java implementation, and users who need a JAR from a particular commit that is not already packaged in `Application/`.

Before building:

1. Install JDK 21.
2. Clone the complete repository.
3. Check out the desired branch, tag, or commit.
4. Open the Java project under `PFGAP/` using the build configuration committed with that revision.
5. Compile and run the repository's tests.
6. Package the application JAR according to the current project configuration.
7. Place or copy the resulting executable JAR to `Application/PFGAP.jar` only after successful testing.

Do not combine compiled classes or dependencies from an older Java version or another PFGAP revision. Perform a clean rebuild after switching revisions or changing major dependencies.

The repository's active project files are authoritative for the exact build command. This documentation does not prescribe a Maven or Gradle command unless that build system is present in the checked-out revision.

## Install optional custom-distance runtimes

PFGAP can load custom distance implementations through supported extension mechanisms. No additional runtime is needed for built-in distances.

### Java custom distances

Java extensions require JDK 21 for compilation. Compile against the PFGAP APIs and dependency versions from the same revision as the application. Package shared implementations in a JAR where practical rather than managing loose class files.

See [Custom Distances](../extensions/Custom_Distances.md) for implementation, packaging, registration, and invocation.

### Python custom distances

Python-backed distances require a working Python interpreter plus every package imported by the custom implementation. Use the same virtual environment for installation and execution, and verify that the Java-launched process can resolve the intended Python executable.

### Maple custom distances

Maple-backed distances require a locally installed and licensed Maple environment compatible with the configured integration. Maple is not required for built-in distances, Java extensions, the Python helper, or ordinary Java execution.

## Platform notes

### Windows

- Use PowerShell or Command Prompt syntax consistently.
- Quote paths containing spaces.
- Use semicolons in `PATH` and `PYTHONPATH`.
- If script execution is restricted, PowerShell may require an execution-policy adjustment before activating a virtual environment.

### macOS

- Confirm that the terminal is using the intended JDK rather than an older system or IDE-selected Java installation.
- On Apple silicon, prefer an ARM64 JDK unless a required native dependency demands x86-64 execution.

### Linux

- Distribution packages may install multiple Java versions at once. Verify both `java` and `javac` after changing the selected alternative.
- Ensure that the user running PFGAP can read input files and create files in the output directory.

## Memory and temporary storage

Dataset size, representation, eager or lazy reading, forest size, proximity output, and parallel worker count can materially affect memory use. For memory-intensive runs, set the Java heap explicitly:

```bash
java -Xms2g -Xmx8g -jar Application/PFGAP.jar -name=value
```

Choose values appropriate for the machine and workload. A larger heap does not correct an invalid configuration, and allocating nearly all physical memory can make the operating system unstable or cause severe paging.

Lazy readers can reduce full-dataset materialization, but they may increase file access and retain caches or open resources according to the selected reader. See [Eager and Lazy Data](../guides/Eager_and_Lazy_Data.md) before choosing an execution strategy for large collections.

## Updating PFGAP

For a Git clone:

```bash
git pull
```

After updating:

1. Review the Git history, pull request, or tagged release information for the revision being installed.
2. Keep `PFGAP.jar` and `PF_wrapper.py` on the same revision.
3. Rebuild custom Java extensions when public APIs or dependencies change.
4. Re-run a representative workflow before replacing an established environment.
5. Preserve old models and outputs until compatibility has been verified.

Because the project is pre-v1, serialized models, configuration names, and extension interfaces may change between revisions. See [Model Persistence](../reference/Model_Persistence.md) for saved-model contents, external dependencies, and compatibility boundaries.

## Troubleshooting

### `java` or `javac` is not recognized

The JDK is not installed, or its `bin` directory is not on `PATH`. Install JDK 21, update `JAVA_HOME` and `PATH`, and open a new terminal.

### Java reports the wrong version

Another Java installation appears earlier on `PATH`. Inspect the executable selected by the shell and update the system or session configuration.

On macOS or Linux:

```bash
command -v java
command -v javac
```

On Windows PowerShell:

```powershell
Get-Command java
Get-Command javac
```

### `Unable to access jarfile`

The path to `PFGAP.jar` is incorrect relative to the current working directory. Confirm the path and file name, or use an absolute path.

### `UnsupportedClassVersionError`

The active Java runtime is older than the Java version used to build PFGAP or a loaded extension. Run PFGAP with Java 21 and rebuild custom Java code with a compatible toolchain.

### `NoClassDefFoundError` or `ClassNotFoundException`

A required dependency or custom extension is missing from the runtime class path, or incompatible artifacts have been mixed. Use the packaged application and extensions from compatible revisions, then perform a clean rebuild if necessary.

### Python cannot import `PF_wrapper`

Run Python from `Application/`, add `Application/` to `PYTHONPATH`, or import the module from its explicit file location. Confirm that the file is named exactly `PF_wrapper.py`.

### The wrapper cannot find Java or `PFGAP.jar`

Verify that `java -version` succeeds in the same environment that starts Python. Keep `PF_wrapper.py` and `PFGAP.jar` together, or configure the paths accepted by the current wrapper interface.

### Output files cannot be created

Confirm that the parent output directory exists where required and that the current user has write permission. Also confirm that relative output paths are being resolved from the expected working directory.

## Next steps

- Follow the [Quick Start](Quick_Start.md) for a complete first run.
- Read [Configuration](Configuration.md) for configuration sources and precedence.
- Use the [CLI Reference](../reference/CLI_Reference.md) for direct Java options.
- Use the [Configuration Reference](../reference/Configuration_Reference.md) for the complete public configuration catalog.
- Review [Data Formats](../data/Data_Formats.md), [Dataset Representations](../data/Dataset_Representations.md), and [Readers](../data/Readers.md) before preparing nontrivial input data.
