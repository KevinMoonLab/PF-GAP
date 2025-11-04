package distance.interop;

import java.io.*;
import java.util.Arrays;

public class PythonDistance implements Serializable {
    private static Process pythonProcess;
    private static BufferedWriter pythonInput;
    private static BufferedReader pythonOutput;
    private static final Object pythonLock = new Object();
    private static boolean initialized = false;

    private final String scriptPath;
    private final String functionName;


    public PythonDistance(String descriptor) throws IOException {
        String[] parts = descriptor.split(":");
        if (parts.length < 2) {
            throw new IllegalArgumentException("Invalid python descriptor format. Use python:path/to/file[:FunctionName]");
        }

        this.scriptPath = parts[1].trim();
        this.functionName = (parts.length >= 3) ? parts[2].trim() : "Distance";

        initialize();
    }

    private void initialize() throws IOException {
        if (initialized) return;

        ProcessBuilder pb = new ProcessBuilder("python3", "-i");
        pythonProcess = pb.redirectErrorStream(true).start();
        pythonInput = new BufferedWriter(new OutputStreamWriter(pythonProcess.getOutputStream()));
        pythonOutput = new BufferedReader(new InputStreamReader(pythonProcess.getInputStream()));

        pythonInput.write("import sys; sys.ps1=''; sys.ps2=''\n");
        pythonInput.flush();

        pythonInput.write("exec(open('" + scriptPath + "').read())\n");
        pythonInput.flush();

        initialized = true;
    }

    public double distance(Object T1, Object T2) throws IOException {
        double[] t1 = (double[]) T1;
        double[] t2 = (double[]) T2;

        synchronized (pythonLock) {
            if (!initialized) {
                initialize();
            }

            String pyT1 = Arrays.toString(t1);
            String pyT2 = Arrays.toString(t2);
            String marker = "RESULT:";

            pythonInput.write("print('" + marker + "', " + functionName + "(" + pyT1 + ", " + pyT2 + ")); sys.stdout.flush()\n");
            pythonInput.flush();

            String line;
            while ((line = pythonOutput.readLine()) != null) {
                line = line.trim();
                if (line.isEmpty() || line.equals(">>>") || line.equals("...")) continue;
                if (line.contains(marker)) {
                    //String result = line.substring(marker.length()).trim();
                    //return Double.parseDouble(result);
                    int markerIndex = line.indexOf(marker);
                    if (markerIndex >= 0) {
                        String result = line.substring(markerIndex + marker.length()).trim();
                        return Double.parseDouble(result);
                    }
                }
            }

            throw new IOException("Failed to get result from Python.");
        }
    }

    public static void close() throws IOException {
        if (pythonInput != null) {
            pythonInput.write("exit()\n");
            pythonInput.flush();
        }
        if (pythonProcess != null) {
            pythonProcess.destroy();
        }
    }
}

/*
public class PythonDistance implements Serializable {
    private static Process pythonProcess;
    private static BufferedWriter pythonInput;
    private static BufferedReader pythonOutput;
    private static final Object pythonLock = new Object();
    private static boolean initialized = false;

    public PythonDistance() throws IOException {
        initialize();
    }

    private static void initialize() throws IOException {
        if (initialized) return;

        // Start a persistent Python interpreter
        //ProcessBuilder pb = new ProcessBuilder("python3", "-i", "-q"); // -i = interactive, -q = quiet
        ProcessBuilder pb = new ProcessBuilder("python3", "-i"); // unbuffered, non-interactive
        pythonProcess = pb.redirectErrorStream(true).start();
        pythonInput = new BufferedWriter(new OutputStreamWriter(pythonProcess.getOutputStream()));
        pythonOutput = new BufferedReader(new InputStreamReader(pythonProcess.getInputStream()));

        // Disable interactive prompts
        pythonInput.write("import sys; sys.ps1=''; sys.ps2=''\n");
        pythonInput.flush();

        // Load the distance function
        pythonInput.write("exec(open('PythonDistance.py').read())\n");
        pythonInput.flush();

        // Drain the initial prompt/output
        initialized = true;
    }

    //public static double distance(double[] t1, double[] t2) throws IOException {
    public static double distance(Object T1, Object T2) throws IOException {

        double[] t1 = (double[]) T1;
        double[] t2 = (double[]) T2;

        synchronized (pythonLock) {
            if (!initialized) {
                initialize();
            }

            String pyT1 = Arrays.toString(t1);
            String pyT2 = Arrays.toString(t2);
            String marker = "RESULT:";

            // Send Python command
            pythonInput.write("print('" + marker + "', Distance(" + pyT1 + ", " + pyT2 + ")); sys.stdout.flush()\n");
            pythonInput.flush();

            // Read lines until result is found
            String line;
            long startTime = System.currentTimeMillis();
            while ((line = pythonOutput.readLine()) != null) {
                line = line.trim();

                // Are there any more of these ">>>"?
                if (line.isEmpty() || line.equals(">>>") || line.equals("...")) continue;

                // Any More at all??
                if (line.contains(marker)) {
                    line = line.replace(">>>", "").trim(); // remove stray prompt
                    String result = line.substring(marker.length()).trim();
                    return Double.parseDouble(result);
                }

                // Safety: timeout if Python hangs.
                //if (System.currentTimeMillis() - startTime > 5000) {
                //    throw new IOException("Timeout waiting for Python result");
                //}
            }

            throw new IOException("Failed to get result from Python.");
        }
    }


    public static void close() throws IOException { // Is this needed?
        if (pythonInput != null) {
            pythonInput.write("exit()\n");
            pythonInput.flush();
        }
        if (pythonProcess != null) {
            pythonProcess.destroy();
        }
    }
}*/