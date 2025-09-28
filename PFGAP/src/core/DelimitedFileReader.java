package core;

import datasets.ListObjectDataset;
import org.apache.commons.lang3.time.DurationFormatUtils;
import util.PrintUtilities;

import java.io.*;
import java.util.*;
import java.util.stream.Collectors;

public class DelimitedFileReader {

    public static ListObjectDataset readToListObjectDataset(
            String dataFileName,
            String labelFileName,
            String entry_separator,
            String array_separator,
            boolean hasHeader,
            boolean is2D,
            boolean isNumeric,
            boolean hasMissingValues,
            boolean targetColumnIsFirst,
            boolean isTest,
            boolean isRegression
    ) {
        ListObjectDataset dataset = new ListObjectDataset();
        File f = new File(dataFileName);
        int i = 0;
        long start = System.nanoTime();

        try {
            //List<Integer> labels = readLabels(labelFileName, hasHeader);
            List<Object> labels = new ArrayList<>();
            if (labelFileName!=null) {
                labels = readGenericLabels(labelFileName, hasHeader, isRegression);
            }
            BufferedReader br = new BufferedReader(new FileReader(dataFileName));
            if (hasHeader) br.readLine(); // skip header

            String line = "";
            while ((line = br.readLine()) != null) {

                if (isNumeric){ //either double or Double
                    if (hasMissingValues){ //must be Double since there's missing data
                        if (is2D){
                            // numeric, has missing values, 2D: Double[][]
                            Double[][] data = RowParser.parseBoxedDoubleMatrix(line, array_separator, entry_separator);
                            //Integer label = labels.get(i);
                            Object label;// = labels.get(i);
                            if (labelFileName!=null) {
                                label = labels.get(i);
                            } else {
                                label = null;
                            }
                            dataset.add(label, data, i);
                            //dataset.setLength(data[0].length); // just the first one: risky.
                            AppContext.length = data[0].length;
                        } else {
                            // numeric, missing values, 1D: Double[]
                            Double[] data = RowParser.parseBoxedDoubleArray(line, entry_separator);
                            //Integer label = labels.get(i);
                            Object label;// = labels.get(i);
                            if (labelFileName!=null) {
                                label = labels.get(i);
                            } else {
                                label = null;
                            }
                            dataset.add(label, data, i);
                            AppContext.length = data.length;
                        }
                    } else {
                        // Should be double since there's no missing data.
                        if (is2D){
                            // numeric, no missing values, 2D: double[][]
                            double[][] data = RowParser.parseDoubleMatrix(line, array_separator, entry_separator);
                            //Integer label = labels.get(i);
                            Object label;// = labels.get(i);
                            if (labelFileName!=null) {
                                label = labels.get(i);
                            } else {
                                label = null;
                            }
                            dataset.add(label, data, i);
                            //dataset.setLength(data[0].length); // just the first one: risky.
                            AppContext.length = data[0].length;
                        } else {
                            // numeric, no missing values, 1D: double[].
                            if (labelFileName == null && !isTest){
                                // this is the original case: the labels can't be missing.
                                String[] lineArray = line.split(entry_separator);
                                ParsedDoubleRow parsed = RowParser.parseDoubleRow(lineArray, targetColumnIsFirst, isRegression);
                                dataset.add(parsed.label, parsed.features, i);
                                //dataset.setLength(parsed.features.length);
                                AppContext.length = parsed.features.length;
                            } else if (labelFileName == null && isTest) {
                                // we need to know if we're looking for testlabels.
                                if (AppContext.exists_testlabels) {
                                    // if there are test labels and no test label file was given, they must be in the test file.
                                    String[] lineArray = line.split(entry_separator);
                                    ParsedDoubleRow parsed = RowParser.parseDoubleRow(lineArray, targetColumnIsFirst, isRegression);
                                    dataset.add(parsed.label, parsed.features, i);
                                    //dataset.setLength(parsed.features.length);
                                    AppContext.length = parsed.features.length;
                                } else {
                                    // if there are no test labels, we're setting them to null.
                                    double[] data = RowParser.parseDoubleArray(line, entry_separator);
                                    //Integer label = labels.get(i);
                                    Object label = null; //labels.get(i);
                                    dataset.add(label, data, i);
                                    //dataset.setLength(data.length);
                                    AppContext.length = data.length;
                                }
                            } else {
                                // this is like the original case, except when labels are provided separately.
                                double[] data = RowParser.parseDoubleArray(line, entry_separator);
                                //Integer label = labels.get(i);
                                Object label = labels.get(i);
                                dataset.add(label, data, i);
                                //dataset.setLength(data.length);
                                AppContext.length = data.length;
                            }

                        }
                    }
                } else{
                    // it is either Object[] or Object[][] since it's not all numeric.
                    if (is2D) {
                        Object[][] data = RowParser.parse2DRow(line, array_separator, entry_separator);
                        //Integer label = labels.get(i);
                        Object label;// = labels.get(i);
                        if (labelFileName!=null) {
                            label = labels.get(i);
                        } else {
                            label = null;
                        }
                        dataset.add(label, data, i);
                        //dataset.setLength(data[0].length); //just the first one: risky.
                        AppContext.length = data[0].length;
                    } else {
                        Object[] data = RowParser.parse1DRow(line, entry_separator);
                        //Integer label = labels.get(i);
                        Object label;// = labels.get(i);
                        if (labelFileName!=null) {
                            label = labels.get(i);
                        } else {
                            label = null;
                        }
                        dataset.add(label, data, i);
                        //dataset.setLength(data.length);
                        AppContext.length = data.length;
                    }
                }

                ProgressLogger.logProgress(i);
                i++;
            }

            long end = System.nanoTime();
            ProgressLogger.logDuration(start, end);

        } catch (IOException e) {
            PrintUtilities.abort(e);
        }

        return dataset;
    }



    public class ProgressLogger {

        public static void logProgress(int i) {
            if (i % 1000 == 0) {
                if (i % 100000 == 0) {
                    System.out.print("\n");
                    if (i % 1000000 == 0) {
                        long usedMem = AppContext.runtime.totalMemory() - AppContext.runtime.freeMemory();
                        System.out.print(i + ":" + usedMem / 1024 / 1024 + "mb\n");
                    }
                } else {
                    System.out.print(".");
                }
            }
        }

        public static void logDuration(long start, long end) {
            long elapsed = end - start;
            String timeDuration = DurationFormatUtils.formatDuration((long) (elapsed / 1e6), "H:m:s.SSS");
            System.out.println("finished in " + timeDuration);
        }
    }


    public class FileInfoExtractor {

        public static int[] getFileInformation(String fileName, boolean hasHeader, String separator) throws IOException {
            String line;
            String[] lineArray = null;
            int[] fileInfo = new int[2];

            try (FileReader input = new FileReader(fileName);
                 LineNumberReader lineNumberReader = new LineNumberReader(input)) {

                boolean lengthCheck = true;

                while ((line = lineNumberReader.readLine()) != null) {
                    if (lengthCheck) {
                        lengthCheck = false;
                        lineArray = line.split(separator);
                    }
                }

                fileInfo[0] = hasHeader ? lineNumberReader.getLineNumber() - 1 : lineNumberReader.getLineNumber();
                fileInfo[1] = lineArray.length;
            }

            return fileInfo;
        }
    }


    public static List<Integer> readLabels(String labelFileName, boolean hasHeader) throws IOException {
        List<Integer> labels = new ArrayList<>();
        if (Objects.equals(labelFileName, null)){ //this is for when labels exist in the training/testing files
            return labels;
        }
        try (BufferedReader br = new BufferedReader(new FileReader(labelFileName))) {
            if (hasHeader) br.readLine(); // skip header
            String line;
            while ((line = br.readLine()) != null) {
                labels.add(Integer.parseInt(line.trim()));
            }
        }
        return labels;
    }


    public static List<Object> readGenericLabels(String labelFileName, boolean hasHeader, boolean isRegression) throws IOException {
        List<Object> labels = new ArrayList<>();
        if (Objects.equals(labelFileName, null)) { //this is for when labels exist in the training/testing files.
            return labels;
        }
        try (BufferedReader br = new BufferedReader(new FileReader(labelFileName))) {
            if (hasHeader) br.readLine();

            String line;
            while ((line = br.readLine()) != null) {
                if (isRegression) {
                    labels.add(Double.parseDouble(line.trim()));
                } else {
                    try {
                        labels.add(Integer.parseInt(line.trim()));
                    } catch (NumberFormatException e) {
                        labels.add(line.trim()); // fallback to string label
                    }
                }
            }
        }
        return labels;
    }



    public static class ParsedDoubleRow {
        //public final int label;
        public final Object label;
        public final double[] features;

        public ParsedDoubleRow(Object label, double[] features) {
            this.label = label;
            this.features = features;
        }
    }


    public class RowParser {

        //usage: RowParser.setMissingIndicators(Set.of("", "NA", "N/A", "null", "?"));

        private static Set<String> missingIndicators = Set.of("", "NA", "N/A", "null");

        public static void setMissingIndicators(Set<String> indicators) {
            missingIndicators = indicators.stream()
                    .map(String::toUpperCase)
                    .collect(Collectors.toSet());
        }

        public static Set<String> getMissingIndicators() {
            return missingIndicators;
        }


        public static ParsedDoubleRow parseDoubleRow(String[] lineArray, boolean targetColumnIsFirst, boolean isRegression) {
            int dataLength = lineArray.length - 1;
            double[] features = new double[dataLength];
            //int label;
            Object label;

            /*if (targetColumnIsFirst) {
                label = Integer.parseInt(lineArray[0]);
                for (int j = 1; j <= dataLength; j++) {
                    features[j - 1] = Double.parseDouble(lineArray[j]);
                }
            } else {
                label = Integer.parseInt(lineArray[dataLength]);
                for (int j = 0; j < dataLength; j++) {
                    features[j] = Double.parseDouble(lineArray[j]);
                }
            }*/


            if (targetColumnIsFirst) {
                label = isRegression
                        ? Double.parseDouble(lineArray[0])
                        : tryParseLabel(lineArray[0]);
                for (int j = 1; j <= dataLength; j++) {
                    features[j - 1] = Double.parseDouble(lineArray[j]);
                }
            } else {
                label = isRegression
                        ? Double.parseDouble(lineArray[dataLength])
                        : tryParseLabel(lineArray[dataLength]);
                for (int j = 0; j < dataLength; j++) {
                    features[j] = Double.parseDouble(lineArray[j]);
                }
            }


            return new ParsedDoubleRow(label, features);
        }


        private static Object tryParseLabel(String token) {
            try {
                return Integer.parseInt(token.trim());
            } catch (NumberFormatException e) {
                return token.trim(); // fallback to string
            }
        }


        public static double[] parseDoubleArray(String row, String separator) { // for no labels
            String[] tokens = row.split(separator);
            double[] parsed = new double[tokens.length];
            for (int i = 0; i < tokens.length; i++) {
                parsed[i] = Double.parseDouble(tokens[i]);
            }
            return parsed;
        }

        public static double[][] parseDoubleMatrix(String row, String array_separator, String entry_separator) {
            String[] rowStrings = row.split(array_separator);
            double[][] matrix = new double[rowStrings.length][];
            for (int i = 0; i < rowStrings.length; i++) {
                matrix[i] = parseDoubleArray(rowStrings[i], entry_separator);
            }
            return matrix;
        }

        public static Double[] parseBoxedDoubleArray(String row, String separator) {
            String[] tokens = row.split(separator);
            Double[] parsed = new Double[tokens.length];
            for (int i = 0; i < tokens.length; i++) {
                //parsed[i] = Double.valueOf(tokens[i]);
                String token = tokens[i].trim();
                parsed[i] = missingIndicators.contains(token.toUpperCase()) ? null : Double.valueOf(token);
            }
            return parsed;
        }

        public static Double[][] parseBoxedDoubleMatrix(String row, String array_separator, String entry_separator) {
            String[] rowStrings = row.split(array_separator);
            Double[][] matrix = new Double[rowStrings.length][];
            for (int i = 0; i < rowStrings.length; i++) {
                matrix[i] = parseBoxedDoubleArray(rowStrings[i], entry_separator);
            }
            return matrix;
        }

        // for Object[]
        public static Object[] parse1DRow(String row, String separator) {
            String[] tokens = row.split(separator);
            Object[] parsed = new Object[tokens.length];
            for (int i = 0; i < tokens.length; i++) {
                parsed[i] = parseValue(tokens[i]);
            }
            return parsed;
        }

        /**
         * Parses a semicolon-separated string of rows into an Object[][].
         * Each row is parsed using the same logic as parse1DRow.
         */
        // for Object[][]
        public static Object[][] parse2DRow(String row, String array_separator, String entry_separator) {
            String[] rowStrings = row.split(array_separator);
            Object[][] matrix = new Object[rowStrings.length][];
            for (int i = 0; i < rowStrings.length; i++) {
                matrix[i] = parse1DRow(rowStrings[i], entry_separator);
            }
            return matrix;
        }

        /**
         * Attempts to parse a string into a Double, Boolean, or leaves it as a String.
         */
        public static Object parseValue(String token) {

            if (token == null) return null;

            String trimmed = token.trim();
            if (missingIndicators.contains(trimmed.toUpperCase())) return null;

            try {
                return Double.parseDouble(trimmed);
            } catch (NumberFormatException e1) {
                if (trimmed.equalsIgnoreCase("true") || trimmed.equalsIgnoreCase("false")) {
                    return Boolean.parseBoolean(trimmed);
                }
                return trimmed; // fallback to String
            }

        }
    }




}
