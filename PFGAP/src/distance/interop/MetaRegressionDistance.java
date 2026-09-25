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
 * Regression meta-distance backed by a persistent Python interpreter.
 *
 * <p>The distance is unchanged: the absolute difference between the two scalar
 * predictions returned by the configured Python function.</p>
 *
 * <p>Primitive {@code double[]} and {@code float[]} observations are accepted.
 * Float input is serialized directly without allocating a converted double
 * array. Python access remains protected by a process-wide lock because all
 * requests and responses share one interpreter stream.</p>
 */
public final class MetaRegressionDistance implements Serializable {

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

    public MetaRegressionDistance(String descriptor) throws IOException {
        String[] parts = parseDescriptor(descriptor);
        scriptPath = parts[1].trim();
        functionName = parts.length >= 3 && !parts[2].isBlank()
                ? parts[2].trim()
                : "predict";

        synchronized (PYTHON_LOCK) {
            initialize();
        }
    }

    public double distance(Object first, Object second) throws IOException {
        synchronized (PYTHON_LOCK) {
            double firstPrediction = getPrediction(first);
            double secondPrediction = getPrediction(second);
            return Math.abs(firstPrediction - secondPrediction);
        }
    }

    public void reinitializeIfNeeded() throws IOException {
        synchronized (PYTHON_LOCK) {
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

    private double getPrediction(Object input) throws IOException {
        reinitializeIfNeeded();
        String pythonVector = toPythonVector(input);

        pythonInput.write(
                "print("
                        + pythonStringLiteral(RESULT_MARKER)
                        + ", "
                        + functionName
                        + "("
                        + pythonVector
                        + ")); sys.stdout.flush()\n"
        );
        pythonInput.flush();

        String line;
        while ((line = pythonOutput.readLine()) != null) {
            int markerPosition = line.indexOf(RESULT_MARKER);
            if (markerPosition < 0) {
                continue;
            }

            String result = line.substring(
                    markerPosition + RESULT_MARKER.length()
            ).trim();
            return Double.parseDouble(result);
        }

        initialized = false;
        throw new IOException("Failed to get prediction from Python.");
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
                "MetaRegressionDistance requires double[] or float[] input. "
                        + "Received "
                        + (input == null
                        ? "null"
                        : input.getClass().getTypeName())
                        + "."
        );
    }

    private static String[] parseDescriptor(String descriptor) {
        if (descriptor == null) {
            throw new IllegalArgumentException(
                    "Meta-regression descriptor cannot be null."
            );
        }
        String[] parts = descriptor.split(":", 3);
        if (parts.length < 2 || parts[1].isBlank()) {
            throw new IllegalArgumentException(
                    "Use metaregress:path/to/model.py[:FunctionName]"
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
            // Reinitialization replaces the stale stream.
        }
        try {
            if (pythonOutput != null) {
                pythonOutput.close();
            }
        } catch (IOException ignored) {
            // Reinitialization replaces the stale stream.
        }
        if (pythonProcess != null) {
            pythonProcess.destroyForcibly();
        }
        pythonInput = null;
        pythonOutput = null;
        pythonProcess = null;
    }
}
