package distance.interop;

import java.io.*;

public class MapleDistance implements Serializable {
    private static Process mapleProcess;
    private static BufferedWriter mapleInput;
    private static BufferedReader mapleOutput;
    private static final Object mapleLock = new Object();
    private static boolean initialized = false;

    private final String scriptPath;
    private final String functionName;

    public MapleDistance(String descriptor) throws IOException {
        String[] parts = descriptor.split(":");
        if (parts.length < 2) {
            throw new IllegalArgumentException("Invalid maple descriptor format. Use maple:path/to/file.mpl[:FunctionName]");
        }

        this.scriptPath = parts[1].trim();
        this.functionName = (parts.length >= 3) ? parts[2].trim() : "Distance";

        initialize();
    }

    private void initialize() throws IOException {
        if (initialized) return;

        ProcessBuilder pb = new ProcessBuilder("maple", "-q");
        mapleProcess = pb.redirectErrorStream(true).start();
        mapleInput = new BufferedWriter(new OutputStreamWriter(mapleProcess.getOutputStream()));
        mapleOutput = new BufferedReader(new InputStreamReader(mapleProcess.getInputStream()));

        mapleInput.write("printf(\"READY\\n\"):\n");
        mapleInput.flush();
        waitForMarker("READY");

        mapleInput.write("read(\"" + scriptPath + "\"):\n");
        mapleInput.flush();

        mapleInput.write("printf(\"LOADED\\n\"):\n");
        mapleInput.flush();
        waitForMarker("LOADED");

        initialized = true;
    }

    private void waitForMarker(String marker) throws IOException {
        String line;
        while ((line = mapleOutput.readLine()) != null) {
            if (line.contains(marker)) {
                return;
            }
        }
        throw new IOException("Did not receive expected marker from Maple: " + marker);
    }

    public double distance(Object T1, Object T2) throws IOException {
        double[] t1 = (double[]) T1;
        double[] t2 = (double[]) T2;

        synchronized (mapleLock) {
            if (!initialized) {
                initialize();
            }

            mapleInput.write("t1 := " + arrayToMapleList(t1) + ":\n");
            mapleInput.write("t2 := " + arrayToMapleList(t2) + ":\n");
            mapleInput.write("res := " + functionName + "(t1, t2):\n");
            mapleInput.write("printf(\"RESULT: %.15f\\n\", res):\n");
            mapleInput.flush();

            String line;
            while ((line = mapleOutput.readLine()) != null) {
                line = line.trim();
                if (line.startsWith("RESULT:")) {
                    String result = line.substring("RESULT:".length()).trim();
                    return Double.parseDouble(result);
                }
            }

            throw new IOException("Failed to read distance result from Maple.");
        }
    }

    public static void close() throws IOException {
        if (!initialized) return;

        mapleInput.write("quit:\n");
        mapleInput.flush();
        mapleProcess.destroy();
        initialized = false;
    }

    private static String arrayToMapleList(double[] arr) {
        StringBuilder sb = new StringBuilder("[");
        for (int i = 0; i < arr.length; i++) {
            sb.append(arr[i]);
            if (i < arr.length - 1) {
                sb.append(",");
            }
        }
        sb.append("]");
        return sb.toString();
    }
}

/*package distance.interop;

import java.io.*;

public class MapleDistance implements Serializable {
    private static Process mapleProcess;
    private static BufferedWriter mapleInput;
    private static BufferedReader mapleOutput;
    private static boolean initialized = false;

    static {
        try {
            initialize();
        } catch (IOException e) {
            throw new RuntimeException("Failed to start Maple process", e);
        }
    }

    private static void initialize() throws IOException {
        if (initialized) return;

        ProcessBuilder pb = new ProcessBuilder("maple", "-q");
        mapleProcess = pb.redirectErrorStream(true).start();
        mapleInput = new BufferedWriter(new OutputStreamWriter(mapleProcess.getOutputStream()));
        mapleOutput = new BufferedReader(new InputStreamReader(mapleProcess.getInputStream()));

        // Wait for initial READY marker
        mapleInput.write("printf(\"READY\\n\"):\n");
        mapleInput.flush();
        waitForMarker("READY");

        // ✅ Load the distance function script
        mapleInput.write("read(\"MapleDistance.mpl\"):\n"); // Adjust path if needed
        mapleInput.flush();

        // Wait for confirmation after loading the script
        mapleInput.write("printf(\"LOADED\\n\"):\n");
        mapleInput.flush();
        waitForMarker("LOADED");

        initialized = true;
    }

    private static void waitForMarker(String marker) throws IOException {
        String line;
        while ((line = mapleOutput.readLine()) != null) {
            if (line.contains(marker)) {
                return;
            }
        }
        throw new IOException("Did not receive expected marker from Maple: " + marker);
    }

    //public static double distance(double[] t1, double[] t2) throws IOException {
    public static double distance(Object T1, Object T2) throws IOException {

        double[] t1 = (double[]) T1;
        double[] t2 = (double[]) T2;

        if (!initialized) {
            initialize();
        }

        mapleInput.write("t1 := " + arrayToMapleList(t1) + ":\n");
        mapleInput.write("t2 := " + arrayToMapleList(t2) + ":\n");
        //mapleInput.write("t1 := Vector(" + arrayToMapleList(t1) + "):\n");
        //mapleInput.write("t2 := Vector(" + arrayToMapleList(t2) + "):\n");
        mapleInput.write("res := Distance(t1, t2):\n");
        mapleInput.write("printf(\"RESULT: %.15f\\n\", res):\n");
        mapleInput.flush();

        // Wait for the result
        String line;
        while ((line = mapleOutput.readLine()) != null) {
            if (line.startsWith("RESULT:")) {
                return Double.parseDouble(line.substring(7).trim());
            }
        }

        throw new IOException("Failed to read distance result from Maple.");
    }

    public static void close() throws IOException { // This may not actually be needed.
        if (!initialized) return;

        mapleInput.write("quit:\n");
        mapleInput.flush();
        mapleProcess.destroy();
        initialized = false;
    }

    private static String arrayToMapleList(double[] arr) {
        StringBuilder sb = new StringBuilder("[");
        for (int i = 0; i < arr.length; i++) {
            sb.append(arr[i]);
            if (i < arr.length - 1) {
                sb.append(",");
            }
        }
        sb.append("]");
        return sb.toString();
    }

}*/
