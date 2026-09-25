package distance.interop;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.io.Serial;
import java.io.Serializable;
import java.util.Arrays;
import java.util.Locale;

/**
 * Classification meta-distance backed by a persistent Python interpreter.
 *
 * <p>The behavioral contract is unchanged:</p>
 * <ul>
 *     <li>{@code class}: distance is zero for equal predicted classes and one
 *         otherwise;</li>
 *     <li>{@code prob}: distance is cosine distance between predicted
 *         probability vectors.</li>
 * </ul>
 *
 * <p>Matching primitive {@code double[]} and {@code float[]} observations are
 * accepted. Float input is serialized directly without allocating a converted
 * double array. Python access remains protected by the existing process-wide
 * lock because requests and responses share one interpreter stream.</p>
 */
public final class MetaClassMatchDistance implements Serializable {

    @Serial
    private static final long serialVersionUID = 1L;

    private static final String CLASS_METHOD = "class";
    private static final String PROBABILITY_METHOD = "prob";
    private static final String RESULT_MARKER = "RESULT:";

    private static final Object PYTHON_LOCK = new Object();

    private static Process pythonProcess;
    private static BufferedWriter pythonInput;
    private static BufferedReader pythonOutput;
    private static boolean initialized;

    private final String scriptPath;
    private final String functionName;
    private final String method;

    public MetaClassMatchDistance(String descriptor) throws IOException {
        String[] parts = parseDescriptor(descriptor);
        scriptPath = parts[1].trim();
        functionName = parts.length >= 3 && !parts[2].isBlank()
                ? parts[2].trim()
                : "predict";
        method = parts.length >= 4 && !parts[3].isBlank()
                ? parts[3].trim().toLowerCase(Locale.ROOT)
                : CLASS_METHOD;
        validateMethod(method);

        synchronized (PYTHON_LOCK) {
            initialize();
        }
    }

    public double distance(Object first, Object second) throws IOException {
        synchronized (PYTHON_LOCK) {
            Object firstPrediction = getPrediction(first);
            Object secondPrediction = getPrediction(second);

            if (CLASS_METHOD.equals(method)) {
                return firstPrediction.equals(secondPrediction) ? 0.0 : 1.0;
            }

            return cosine(
                    (double[]) firstPrediction,
                    (double[]) secondPrediction
            );
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

    private Object getPrediction(Object input) throws IOException {
        reinitializeIfNeeded();
        String pythonInputVector = toPythonVector(input);

        pythonInput.write(
                "print("
                        + pythonStringLiteral(RESULT_MARKER)
                        + ", "
                        + functionName
                        + "("
                        + pythonInputVector
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
            return CLASS_METHOD.equals(method)
                    ? result
                    : parseProbabilityVector(result);
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
                "MetaClassMatchDistance requires double[] or float[] input. "
                        + "Received "
                        + (input == null
                        ? "null"
                        : input.getClass().getTypeName())
                        + "."
        );
    }

    private static double[] parseProbabilityVector(String result) {
        String cleaned = result.trim();
        if (cleaned.startsWith("[") && cleaned.endsWith("]")) {
            cleaned = cleaned.substring(1, cleaned.length() - 1).trim();
        }
        if (cleaned.isEmpty()) {
            return new double[0];
        }

        String[] tokens = cleaned.split(",");
        double[] probabilities = new double[tokens.length];
        for (int index = 0; index < tokens.length; index++) {
            probabilities[index] = Double.parseDouble(tokens[index].trim());
        }
        return probabilities;
    }

    private static double cosine(double[] first, double[] second) {
        if (first.length != second.length) {
            throw new IllegalArgumentException(
                    "Probability vectors must have equal lengths. Received "
                            + first.length + " and " + second.length + "."
            );
        }

        double dotProduct = 0.0;
        double firstNorm = 0.0;
        double secondNorm = 0.0;
        for (int index = 0; index < first.length; index++) {
            double firstValue = first[index];
            double secondValue = second[index];
            dotProduct += firstValue * secondValue;
            firstNorm += firstValue * firstValue;
            secondNorm += secondValue * secondValue;
        }
        return 1.0 - dotProduct
                / (Math.sqrt(firstNorm) * Math.sqrt(secondNorm));
    }

    private static String[] parseDescriptor(String descriptor) {
        if (descriptor == null) {
            throw new IllegalArgumentException(
                    "Meta-class descriptor cannot be null."
            );
        }
        String[] parts = descriptor.split(":", 4);
        if (parts.length < 2 || parts[1].isBlank()) {
            throw new IllegalArgumentException(
                    "Use meta_classmatch:path/to/model.py"
                            + "[:FunctionName][:Method]"
            );
        }
        return parts;
    }

    private static void validateMethod(String method) {
        if (!CLASS_METHOD.equals(method)
                && !PROBABILITY_METHOD.equals(method)) {
            throw new IllegalArgumentException(
                    "Unknown meta-class method: " + method + "."
            );
        }
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
            // Reinitialization will replace the stale stream.
        }
        try {
            if (pythonOutput != null) {
                pythonOutput.close();
            }
        } catch (IOException ignored) {
            // Reinitialization will replace the stale stream.
        }
        if (pythonProcess != null) {
            pythonProcess.destroyForcibly();
        }
        pythonInput = null;
        pythonOutput = null;
        pythonProcess = null;
    }
}
