package distance.interop;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.io.Serial;
import java.io.Serializable;
import java.util.Arrays;

/**
 * User-defined distance backed by a persistent Python interpreter.
 *
 * <p>The configured Python function continues to receive two ordinary Python
 * lists and must return one numeric distance. Matching {@code double[]} and
 * {@code float[]} inputs are supported. Calls remain serialized because all
 * instances share one interpreter process and its input/output streams.</p>
 */
public final class PythonDistance implements Serializable {

    @Serial
    private static final long serialVersionUID = 1L;

    private static final Object PYTHON_LOCK = new Object();
    private static final String RESULT_MARKER = "RESULT:";

    private static Process pythonProcess;
    private static BufferedWriter pythonInput;
    private static BufferedReader pythonOutput;
    private static boolean initialized;

    private final String scriptPath;
    private final String functionName;

    public PythonDistance(String descriptor) throws IOException {
        String[] parts = parseDescriptor(descriptor);
        scriptPath = parts[1].trim();
        functionName = parts.length >= 3 && !parts[2].isBlank()
                ? parts[2].trim()
                : "Distance";

        synchronized (PYTHON_LOCK) {
            initialize();
        }
    }

    public double distance(Object first, Object second) throws IOException {
        String firstVector = toPythonVector(first);
        String secondVector = toPythonVector(second);

        synchronized (PYTHON_LOCK) {
            reinitializeIfNeeded();
            pythonInput.write(
                    "print("
                            + pythonStringLiteral(RESULT_MARKER)
                            + ", "
                            + functionName
                            + "("
                            + firstVector
                            + ", "
                            + secondVector
                            + ")); sys.stdout.flush()\n"
            );
            pythonInput.flush();

            String line;
            while ((line = pythonOutput.readLine()) != null) {
                int markerIndex = line.indexOf(RESULT_MARKER);
                if (markerIndex < 0) {
                    continue;
                }
                String result = line.substring(
                        markerIndex + RESULT_MARKER.length()
                ).trim();
                return Double.parseDouble(result);
            }

            initialized = false;
            throw new IOException("Failed to get result from Python.");
        }
    }

    public static void close() throws IOException {
        synchronized (PYTHON_LOCK) {
            IOException failure = null;
            if (pythonInput != null) {
                try {
                    pythonInput.write("exit()\n");
                    pythonInput.flush();
                } catch (IOException exception) {
                    failure = exception;
                }
            }
            closeProcessResources();
            initialized = false;
            if (failure != null) {
                throw failure;
            }
        }
    }

    private void initialize() throws IOException {
        if (initialized
                && pythonProcess != null
                && pythonProcess.isAlive()
                && pythonInput != null
                && pythonOutput != null) {
            return;
        }

        ProcessBuilder processBuilder = new ProcessBuilder("python3", "-i");
        pythonProcess = processBuilder.redirectErrorStream(true).start();
        pythonInput = new BufferedWriter(
                new OutputStreamWriter(pythonProcess.getOutputStream())
        );
        pythonOutput = new BufferedReader(
                new InputStreamReader(pythonProcess.getInputStream())
        );

        pythonInput.write("import sys; sys.ps1=''; sys.ps2=''\n");
        pythonInput.write(
                "exec(compile(open("
                        + pythonStringLiteral(scriptPath)
                        + ").read(), "
                        + pythonStringLiteral(scriptPath)
                        + ", 'exec'))\n"
        );
        pythonInput.flush();
        initialized = true;
    }

    private void reinitializeIfNeeded() throws IOException {
        if (!initialized
                || pythonProcess == null
                || !pythonProcess.isAlive()
                || pythonInput == null
                || pythonOutput == null) {
            closeProcessResources();
            initialized = false;
            initialize();
        }
    }

    private static String toPythonVector(Object input) {
        if (input instanceof double[] values) {
            return Arrays.toString(values);
        }
        if (input instanceof float[] values) {
            StringBuilder builder = new StringBuilder(
                    Math.max(2, values.length * 8)
            );
            builder.append('[');
            for (int index = 0; index < values.length; index++) {
                if (index > 0) {
                    builder.append(',').append(' ');
                }
                builder.append(Float.toString(values[index]));
            }
            return builder.append(']').toString();
        }
        throw new IllegalArgumentException(
                "PythonDistance requires double[] or float[] input. Received "
                        + typeName(input) + "."
        );
    }

    private static String[] parseDescriptor(String descriptor) {
        if (descriptor == null) {
            throw new IllegalArgumentException(
                    "Python distance descriptor cannot be null."
            );
        }
        String[] parts = descriptor.split(":", 3);
        if (parts.length < 2 || parts[1].isBlank()) {
            throw new IllegalArgumentException(
                    "Use python:path/to/file[:FunctionName]"
            );
        }
        return parts;
    }

    private static String pythonStringLiteral(String value) {
        return "'" + value
                .replace("\\", "\\\\")
                .replace("'", "\\'")
                .replace("\n", "\\n")
                .replace("\r", "\\r")
                + "'";
    }

    private static void closeProcessResources() {
        try {
            if (pythonInput != null) {
                pythonInput.close();
            }
        } catch (IOException ignored) {
            // The process is being discarded.
        }
        try {
            if (pythonOutput != null) {
                pythonOutput.close();
            }
        } catch (IOException ignored) {
            // The process is being discarded.
        }
        if (pythonProcess != null) {
            pythonProcess.destroyForcibly();
        }
        pythonInput = null;
        pythonOutput = null;
        pythonProcess = null;
    }

    private static String typeName(Object value) {
        return value == null ? "null" : value.getClass().getTypeName();
    }
}
