package distance.interop;

import java.io.*;
import java.util.Arrays;

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

    public static double distance(double[] t1, double[] t2) throws IOException {
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
}