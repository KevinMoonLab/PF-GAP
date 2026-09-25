package distance.interop;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.io.Serial;
import java.io.Serializable;

/**
 * User-defined distance backed by a persistent Maple process.
 *
 * <p>The configured Maple procedure continues to receive two Maple lists and
 * must return one numeric distance. Matching {@code double[]} and
 * {@code float[]} inputs are supported. Calls remain serialized because all
 * instances share one Maple process and its input/output streams.</p>
 */
public final class MapleDistance implements Serializable {

    @Serial
    private static final long serialVersionUID = 1L;

    private static final Object MAPLE_LOCK = new Object();
    private static final String READY_MARKER = "READY";
    private static final String LOADED_MARKER = "LOADED";
    private static final String RESULT_MARKER = "RESULT:";

    private static Process mapleProcess;
    private static BufferedWriter mapleInput;
    private static BufferedReader mapleOutput;
    private static boolean initialized;

    private final String scriptPath;
    private final String functionName;

    public MapleDistance(String descriptor) throws IOException {
        String[] parts = parseDescriptor(descriptor);
        scriptPath = parts[1].trim();
        functionName = parts.length >= 3 && !parts[2].isBlank()
                ? parts[2].trim()
                : "Distance";

        synchronized (MAPLE_LOCK) {
            initialize();
        }
    }

    public double distance(Object first, Object second) throws IOException {
        String firstList = toMapleList(first);
        String secondList = toMapleList(second);

        synchronized (MAPLE_LOCK) {
            reinitializeIfNeeded();
            mapleInput.write("t1 := " + firstList + ":\n");
            mapleInput.write("t2 := " + secondList + ":\n");
            mapleInput.write(
                    "res := " + functionName + "(t1, t2):\n"
            );
            mapleInput.write(
                    "printf(\"RESULT: %.15f\\n\", res):\n"
            );
            mapleInput.flush();

            String line;
            while ((line = mapleOutput.readLine()) != null) {
                String trimmed = line.trim();
                if (trimmed.startsWith(RESULT_MARKER)) {
                    return Double.parseDouble(
                            trimmed.substring(RESULT_MARKER.length()).trim()
                    );
                }
            }

            initialized = false;
            throw new IOException("Failed to read distance result from Maple.");
        }
    }

    public static void close() throws IOException {
        synchronized (MAPLE_LOCK) {
            IOException failure = null;
            if (mapleInput != null) {
                try {
                    mapleInput.write("quit:\n");
                    mapleInput.flush();
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
                && mapleProcess != null
                && mapleProcess.isAlive()
                && mapleInput != null
                && mapleOutput != null) {
            return;
        }

        ProcessBuilder processBuilder = new ProcessBuilder("maple", "-q");
        mapleProcess = processBuilder.redirectErrorStream(true).start();
        mapleInput = new BufferedWriter(
                new OutputStreamWriter(mapleProcess.getOutputStream())
        );
        mapleOutput = new BufferedReader(
                new InputStreamReader(mapleProcess.getInputStream())
        );

        mapleInput.write("printf(\"READY\\n\"):\n");
        mapleInput.flush();
        waitForMarker(READY_MARKER);

        mapleInput.write(
                "read(\"" + mapleStringContent(scriptPath) + "\"):\n"
        );
        mapleInput.write("printf(\"LOADED\\n\"):\n");
        mapleInput.flush();
        waitForMarker(LOADED_MARKER);
        initialized = true;
    }

    private void reinitializeIfNeeded() throws IOException {
        if (!initialized
                || mapleProcess == null
                || !mapleProcess.isAlive()
                || mapleInput == null
                || mapleOutput == null) {
            closeProcessResources();
            initialized = false;
            initialize();
        }
    }

    private static void waitForMarker(String marker) throws IOException {
        String line;
        while ((line = mapleOutput.readLine()) != null) {
            if (line.contains(marker)) {
                return;
            }
        }
        throw new IOException(
                "Did not receive expected marker from Maple: " + marker
        );
    }

    private static String toMapleList(Object input) {
        if (input instanceof double[] values) {
            StringBuilder builder = new StringBuilder(
                    Math.max(2, values.length * 10)
            );
            builder.append('[');
            for (int index = 0; index < values.length; index++) {
                if (index > 0) {
                    builder.append(',');
                }
                builder.append(Double.toString(values[index]));
            }
            return builder.append(']').toString();
        }
        if (input instanceof float[] values) {
            StringBuilder builder = new StringBuilder(
                    Math.max(2, values.length * 8)
            );
            builder.append('[');
            for (int index = 0; index < values.length; index++) {
                if (index > 0) {
                    builder.append(',');
                }
                builder.append(Float.toString(values[index]));
            }
            return builder.append(']').toString();
        }
        throw new IllegalArgumentException(
                "MapleDistance requires double[] or float[] input. Received "
                        + typeName(input) + "."
        );
    }

    private static String[] parseDescriptor(String descriptor) {
        if (descriptor == null) {
            throw new IllegalArgumentException(
                    "Maple distance descriptor cannot be null."
            );
        }
        String[] parts = descriptor.split(":", 3);
        if (parts.length < 2 || parts[1].isBlank()) {
            throw new IllegalArgumentException(
                    "Use maple:path/to/file.mpl[:FunctionName]"
            );
        }
        return parts;
    }

    private static String mapleStringContent(String value) {
        return value
                .replace("\\", "\\\\")
                .replace("\"", "\\\"")
                .replace("\n", "\\n")
                .replace("\r", "\\r");
    }

    private static void closeProcessResources() {
        try {
            if (mapleInput != null) {
                mapleInput.close();
            }
        } catch (IOException ignored) {
            // The process is being discarded.
        }
        try {
            if (mapleOutput != null) {
                mapleOutput.close();
            }
        } catch (IOException ignored) {
            // The process is being discarded.
        }
        if (mapleProcess != null) {
            mapleProcess.destroyForcibly();
        }
        mapleInput = null;
        mapleOutput = null;
        mapleProcess = null;
    }

    private static String typeName(Object value) {
        return value == null ? "null" : value.getClass().getTypeName();
    }
}
