package core;

//import datasets.ListDataset;
import datasets.ListObjectDataset;
import imputation.MissingIndicesBuilder;
import org.apache.commons.lang3.time.DurationFormatUtils;
import util.PrintUtilities;

import java.io.*;
import java.util.*;
import java.util.stream.Collectors;

public class DelimitedFileReader {

    /*public static ListDataset readToListDataset(String fileName, boolean hasHeader, boolean targetColumnIsFirst, String separator) {
        File f = new File(fileName);
        ListDataset dataset = null;
        int i = 0;
        long start = System.nanoTime();

        try (BufferedReader br = new BufferedReader(new FileReader(fileName))) {
            System.out.print("reading file [" + f.getName() + "]:");

            int[] fileInfo = FileInfoExtractor.getFileInformation(fileName, hasHeader, separator);
            int expectedSize = fileInfo[0];
            int dataLength = fileInfo[1] - 1;

            dataset = new ListDataset(expectedSize, dataLength);

            if (hasHeader) br.readLine(); // skip header

            String line;
            while ((line = br.readLine()) != null) {
                String[] lineArray = line.split(separator);
                ParsedDoubleRow parsed = RowParser.parseDoubleRow(lineArray, targetColumnIsFirst);
                dataset.add(parsed.label, parsed.features, i);

                ProgressLogger.logProgress(i);
                i++;
            }

            long end = System.nanoTime();
            ProgressLogger.logDuration(start, end);

        } catch (IOException e) {
            PrintUtilities.abort(e);
        }

        return dataset;
    }*/


    // this is the more generic reader.
    public static ListObjectDataset readToListObjectDataset(
            String dataFileName,
            String labelFileName,
            String entry_separator,
            String array_separator,
            boolean hasHeader,
            boolean is2D,
            boolean isNumeric,
            boolean hasMissingValues,
            boolean targetColumnIsFirst
    ) {
        ListObjectDataset dataset = new ListObjectDataset();
        File f = new File(dataFileName);
        int i = 0;
        long start = System.nanoTime();

        try {
            List<Integer> labels = readLabels(labelFileName, hasHeader);
            BufferedReader br = new BufferedReader(new FileReader(dataFileName));
            if (hasHeader) br.readLine(); // skip header

            String line = "";
            while ((line = br.readLine()) != null) {
                //Object data;

                if (isNumeric){ //either double or Double
                    if (hasMissingValues){ //must be Double since there's missing data
                        if (is2D){
                            // numeric, has missing values, 2D: Double[][]
                            Double[][] data = RowParser.parseBoxedDoubleMatrix(line, array_separator, entry_separator);
                            Integer label = labels.get(i);
                            dataset.add(label, data, i);
                            //dataset.setLength(data[0].length); // just the first one: risky.
                            AppContext.length = data[0].length;
                            //AppContext.missing_train_indices.add(MissingIndicesBuilder.buildFrom(data, isTrain));
                            //MissingIndicesBuilder.buildFrom(data, isTrain);
                        } else {
                            // numeric, missing values, 1D: Double[]
                            //AppContext.missing_train_indices.add(MissingIndicesBuilder.buildFrom(data, isTrain));
                            //MissingIndicesBuilder.buildFrom(data, isTrain);
                            Double[] data = RowParser.parseBoxedDoubleArray(line, entry_separator);
                            Integer label = labels.get(i);
                            dataset.add(label, data, i);
                            AppContext.length = data.length;
                        }
                    } else {
                        // Should be double since there's no missing data.
                        if (is2D){
                            // numeric, no missing values, 2D: double[][]
                            double[][] data = RowParser.parseDoubleMatrix(line, array_separator, entry_separator);
                            Integer label = labels.get(i);
                            dataset.add(label, data, i);
                            //dataset.setLength(data[0].length); // just the first one: risky.
                            AppContext.length = data[0].length;
                            // there should be no missing data: do we have to construct this?
                            //AppContext.missing_train_indices.add(MissingIndicesBuilder.buildFrom(data, isTrain));
                        } else {
                            // numeric, no missing values, 1D: double[].
                            if (labelFileName == null){
                                // this is the original case
                                //System.out.println(line);
                                //String[] lineArray = line.split("\t"); //line.split(rowSeparator);
                                String[] lineArray = line.split(entry_separator);
                                //System.out.println(Arrays.toString(lineArray));
                                ParsedDoubleRow parsed = RowParser.parseDoubleRow(lineArray, targetColumnIsFirst);
                                dataset.add(parsed.label, parsed.features, i);
                                //dataset.setLength(parsed.features.length);
                                AppContext.length = parsed.features.length;
                                //System.out.println(parsed.features.length);
                                //System.out.println(dataset.length());
                                // there should be no missing data: do we have to construct this?
                                //AppContext.missing_train_indices.add(MissingIndicesBuilder.buildFrom(data, isTrain));
                            } else {
                                // this is like the original case, except when labels are provided separately.
                                double[] data = RowParser.parseDoubleArray(line, entry_separator);
                                Integer label = labels.get(i);
                                dataset.add(label, data, i);
                                //dataset.setLength(data.length);
                                AppContext.length = data.length;
                                // there should be no missing data: do we have to construct this?
                                //AppContext.missing_train_indices.add(MissingIndicesBuilder.buildFrom(data, isTrain));
                            }

                        }
                    }
                } else{
                    // it is either Object[] or Object[][] since it's not all numeric.
                    if (is2D) {
                        Object[][] data = RowParser.parse2DRow(line, array_separator, entry_separator);
                        Integer label = labels.get(i);
                        dataset.add(label, data, i);
                        //dataset.setLength(data[0].length); //just the first one: risky.
                        AppContext.length = data[0].length;
                        //AppContext.missing_train_indices.add(MissingIndicesBuilder.buildFrom(data, isTrain));
                        //MissingIndicesBuilder.buildFrom(data, isTrain);
                    } else {
                        Object[] data = RowParser.parse1DRow(line, entry_separator);
                        Integer label = labels.get(i);
                        dataset.add(label, data, i);
                        //dataset.setLength(data.length);
                        AppContext.length = data.length;
                        //AppContext.missing_train_indices.add(MissingIndicesBuilder.buildFrom(data, isTrain));
                        //MissingIndicesBuilder.buildFrom(data, isTrain);
                    }
                }

                //Integer label = labels.get(i);
                //dataset.add(label, data, i);

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



    public static class ParsedDoubleRow {
        public final int label;
        public final double[] features;

        public ParsedDoubleRow(int label, double[] features) {
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


        public static ParsedDoubleRow parseDoubleRow(String[] lineArray, boolean targetColumnIsFirst) {
            int dataLength = lineArray.length - 1;
            double[] features = new double[dataLength];
            int label;

            if (targetColumnIsFirst) {
                //System.out.println(lineArray[0]);
                label = Integer.parseInt(lineArray[0]);
                for (int j = 1; j <= dataLength; j++) {
                    features[j - 1] = Double.parseDouble(lineArray[j]);
                }
            } else {
                label = Integer.parseInt(lineArray[dataLength]);
                for (int j = 0; j < dataLength; j++) {
                    features[j] = Double.parseDouble(lineArray[j]);
                }
            }

            return new ParsedDoubleRow(label, features);
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


            /*if (token == null || token.trim().isEmpty() || token.equalsIgnoreCase("null")) return null;
            try {
                return Double.parseDouble(token);
            } catch (NumberFormatException e1) {
                if (token.equalsIgnoreCase("true") || token.equalsIgnoreCase("false")) {
                    return Boolean.parseBoolean(token);
                }
                return token; // fallback to String
            }*/
        }
    }




}
