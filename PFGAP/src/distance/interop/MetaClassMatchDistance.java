package distance.interop;

import java.io.*;
import java.util.Arrays;

public class MetaClassMatchDistance implements Serializable {
    private static Process pythonProcess;
    private static BufferedWriter pythonInput;
    private static BufferedReader pythonOutput;
    private static final Object pythonLock = new Object();
    private static boolean initialized = false;

    private final String scriptPath;
    private final String functionName;
    private final String method;

    public MetaClassMatchDistance(String descriptor) throws IOException {
        String[] parts = descriptor.split(":");
        if (parts.length < 2) throw new IllegalArgumentException("Use meta_classmatch:path/to/model.py[:FunctionName][:Method]");

        this.scriptPath = parts[1].trim();
        this.functionName = (parts.length >= 3) ? parts[2].trim() : "predict";
        this.method = (parts.length >= 4) ? parts[3].trim().toLowerCase() : "class";

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

    public void reinitializeIfNeeded() throws IOException {
        if (!initialized || pythonInput == null || pythonOutput == null || pythonProcess == null) {
            initialize();
        }
    }

    //@Override
    public double distance(Object T1, Object T2) throws IOException {
        synchronized (pythonLock) {
            Object pred1 = getPrediction(T1);
            Object pred2 = getPrediction(T2);

            if (method.equals("class")) {
                return pred1.equals(pred2) ? 0.0 : 1.0;
            } else if (method.equals("prob")) {
                double[] p1 = (double[]) pred1;
                double[] p2 = (double[]) pred2;
                //return euclidean(p1, p2);
                return cosine(p1, p2);
            } else {
                throw new IllegalArgumentException("Unknown method: " + method);
            }
        }
    }

    private Object getPrediction(Object input) throws IOException {
        double[] x = (double[]) input;
        String pyX = Arrays.toString(x);
        String marker = "RESULT:";

        reinitializeIfNeeded();
        pythonInput.write("print('" + marker + "', " + functionName + "(" + pyX + ")); sys.stdout.flush()\n");
        pythonInput.flush();

        String line;
        while ((line = pythonOutput.readLine()) != null) {
            line = line.trim();
            if (line.contains(marker)) {
                String result = line.substring(line.indexOf(marker) + marker.length()).trim();
                if (method.equals("class")) {
                    return result;
                } else {
                    // Parse as probability vector
                    result = result.replace("[", "").replace("]", "");
                    String[] tokens = result.split(",");
                    double[] probs = new double[tokens.length];
                    for (int i = 0; i < tokens.length; i++) {
                        probs[i] = Double.parseDouble(tokens[i].trim());
                    }
                    return probs;
                }
            }
        }

        throw new IOException("Failed to get prediction from Python.");
    }

    private double euclidean(double[] a, double[] b) {
        double sum = 0.0;
        for (int i = 0; i < a.length; i++) {
            double diff = a[i] - b[i];
            sum += diff * diff;
        }
        return Math.sqrt(sum);
    }


    private double cosine(double[] a, double[] b) {
        double dot = 0.0, normA = 0.0, normB = 0.0;
        for (int i = 0; i < a.length; i++) {
            dot += a[i] * b[i];
            normA += a[i] * a[i];
            normB += b[i] * b[i];
        }
        return 1.0 - (dot / (Math.sqrt(normA) * Math.sqrt(normB)));
    }

}
