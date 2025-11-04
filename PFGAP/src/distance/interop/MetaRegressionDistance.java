package distance.interop;

import java.io.*;
import java.util.Arrays;

public class MetaRegressionDistance implements Serializable {
    private static Process pythonProcess;
    private static BufferedWriter pythonInput;
    private static BufferedReader pythonOutput;
    private static final Object pythonLock = new Object();
    private static boolean initialized = false;

    private final String scriptPath;
    private final String functionName;

    public MetaRegressionDistance(String descriptor) throws IOException {
        String[] parts = descriptor.split(":");
        if (parts.length < 2) throw new IllegalArgumentException("Use metaregress:path/to/model.py[:FunctionName]");

        this.scriptPath = parts[1].trim();
        this.functionName = (parts.length >= 3) ? parts[2].trim() : "predict";

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
        synchronized (pythonLock) {
            double pred1 = getPrediction(T1);
            double pred2 = getPrediction(T2);
            return Math.abs(pred1 - pred2);
        }
    }

    private double getPrediction(Object input) throws IOException {
        double[] x = (double[]) input;
        String pyX = Arrays.toString(x);
        String marker = "RESULT:";

        pythonInput.write("print('" + marker + "', " + functionName + "(" + pyX + ")); sys.stdout.flush()\n");
        pythonInput.flush();

        String line;
        while ((line = pythonOutput.readLine()) != null) {
            line = line.trim();
            if (line.contains(marker)) {
                String result = line.substring(line.indexOf(marker) + marker.length()).trim();
                return Double.parseDouble(result);
            }
        }

        throw new IOException("Failed to get prediction from Python.");
    }
}
