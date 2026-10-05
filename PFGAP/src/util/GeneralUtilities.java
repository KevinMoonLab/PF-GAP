package util;

import org.apache.commons.lang3.time.DurationFormatUtils;

import java.io.BufferedWriter;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.List;
import java.util.Objects;

/**
 * Legacy general-purpose utilities retained for compatibility with existing
 * experiment and output code.
 *
 * <p>Dataset writing will migrate to the dedicated {@code datasets.writers}
 * package. The delimited writer in this class deliberately preserves its
 * existing two-separator layout while adding complete float-array support so
 * current imputed-data output remains usable during that migration.</p>
 */
public final class GeneralUtilities {

    private GeneralUtilities() {
        // Utility class.
    }

    public static String getCurrentTimeStamp(
            String format
    ) {
        Objects.requireNonNull(format, "Timestamp format cannot be null.");
        return LocalDateTime.now().format(
                DateTimeFormatter.ofPattern(format)
        );
    }

    public static String formatTime(
            long duration,
            String format
    ) {
        // Preserve the legacy output format and public signature. The supplied
        // format parameter was historically ignored by this implementation.
        return DurationFormatUtils.formatDuration(
                duration,
                "H:m:s.SSS"
        );
    }

    public static void warmUpJavaRuntime() {
        // TODO: Replace with a measured warm-up routine or remove this legacy
        // utility after all call sites have been audited.
        System.out.println(
                "TODO doing some extra work to warm up jvm..."
                        + "this helps to measure time more accurately for "
                        + "short experiments"
        );
    }

    /**
     * Writes materialized dataset instances using the legacy two-separator
     * delimited representation.
     *
     * <p>Each top-level list element is written on one physical line. For a
     * one-dimensional array, {@code secondSeparator} separates numeric or
     * object entries. For a two-dimensional array,
     * {@code secondSeparator} separates entries within a row and
     * {@code firstSeparator} separates rows within the top-level instance.</p>
     *
     * <p>This method preserves the historical trailing
     * {@code firstSeparator} after the last row of a two-dimensional instance.
     * The future FastCSV-backed writer will define a clearer layout contract;
     * this compatibility method should not silently change existing output
     * files during the transition.</p>
     *
     * <p>Supported representations are {@code Object[]}, {@code Object[][]},
     * {@code double[]}, {@code double[][]}, {@code float[]}, and
     * {@code float[][]}. Boxed {@code Double[]} values are covered by the
     * {@code Object[]} branch.</p>
     *
     * @param data materialized instances to write
     * @param filePath output file path
     * @param firstSeparator separator between rows of a two-dimensional
     *        instance
     * @param secondSeparator separator between entries within one row or
     *        one-dimensional instance
     * @throws IOException if directories cannot be created or output fails
     */
    public static void writeDelimitedData(
            List<Object> data,
            String filePath,
            String firstSeparator,
            String secondSeparator
    ) throws IOException {
        Objects.requireNonNull(data, "Data cannot be null.");
        Objects.requireNonNull(filePath, "Output file path cannot be null.");
        Objects.requireNonNull(
                firstSeparator,
                "First separator cannot be null."
        );
        Objects.requireNonNull(
                secondSeparator,
                "Second separator cannot be null."
        );

        Path outputPath = Paths.get(filePath);
        Path parent = outputPath.toAbsolutePath().getParent();
        if (parent != null) {
            Files.createDirectories(parent);
        }

        try (BufferedWriter writer = Files.newBufferedWriter(
                outputPath,
                StandardCharsets.UTF_8
        )) {
            for (int instanceIndex = 0;
                 instanceIndex < data.size();
                 instanceIndex++) {
                Object value = data.get(instanceIndex);
                if (value == null) {
                    throw new IllegalArgumentException(
                            "Cannot write null dataset instance at index "
                                    + instanceIndex
                                    + "."
                    );
                }

                writeInstance(
                        writer,
                        value,
                        firstSeparator,
                        secondSeparator,
                        instanceIndex
                );
                writer.newLine();
            }
        }
    }

    private static void writeInstance(
            BufferedWriter writer,
            Object value,
            String firstSeparator,
            String secondSeparator,
            int instanceIndex
    ) throws IOException {
        if (value instanceof double[][] matrix) {
            writeMatrix(
                    writer,
                    matrix,
                    firstSeparator,
                    secondSeparator,
                    instanceIndex
            );
            return;
        }
        if (value instanceof float[][] matrix) {
            writeMatrix(
                    writer,
                    matrix,
                    firstSeparator,
                    secondSeparator,
                    instanceIndex
            );
            return;
        }
        if (value instanceof Object[][] matrix) {
            writeMatrix(
                    writer,
                    matrix,
                    firstSeparator,
                    secondSeparator,
                    instanceIndex
            );
            return;
        }
        if (value instanceof double[] array) {
            writer.write(join(array, secondSeparator));
            return;
        }
        if (value instanceof float[] array) {
            writer.write(join(array, secondSeparator));
            return;
        }
        if (value instanceof Object[] array) {
            writer.write(join(array, secondSeparator));
            return;
        }

        writer.write(value.toString());
    }

    private static void writeMatrix(
            BufferedWriter writer,
            double[][] matrix,
            String rowSeparator,
            String entrySeparator,
            int instanceIndex
    ) throws IOException {
        for (int rowIndex = 0; rowIndex < matrix.length; rowIndex++) {
            double[] row = requireRow(
                    matrix[rowIndex],
                    instanceIndex,
                    rowIndex
            );
            writer.write(join(row, entrySeparator));
            writer.write(rowSeparator);
        }
    }

    private static void writeMatrix(
            BufferedWriter writer,
            float[][] matrix,
            String rowSeparator,
            String entrySeparator,
            int instanceIndex
    ) throws IOException {
        for (int rowIndex = 0; rowIndex < matrix.length; rowIndex++) {
            float[] row = requireRow(
                    matrix[rowIndex],
                    instanceIndex,
                    rowIndex
            );
            writer.write(join(row, entrySeparator));
            writer.write(rowSeparator);
        }
    }

    private static void writeMatrix(
            BufferedWriter writer,
            Object[][] matrix,
            String rowSeparator,
            String entrySeparator,
            int instanceIndex
    ) throws IOException {
        for (int rowIndex = 0; rowIndex < matrix.length; rowIndex++) {
            Object[] row = requireRow(
                    matrix[rowIndex],
                    instanceIndex,
                    rowIndex
            );
            writer.write(join(row, entrySeparator));
            writer.write(rowSeparator);
        }
    }

    private static String join(
            double[] array,
            String separator
    ) {
        StringBuilder builder = new StringBuilder(
                Math.max(16, array.length * 12)
        );
        for (int index = 0; index < array.length; index++) {
            if (index > 0) {
                builder.append(separator);
            }
            builder.append(array[index]);
        }
        return builder.toString();
    }

    private static String join(
            float[] array,
            String separator
    ) {
        StringBuilder builder = new StringBuilder(
                Math.max(16, array.length * 8)
        );
        for (int index = 0; index < array.length; index++) {
            if (index > 0) {
                builder.append(separator);
            }
            builder.append(array[index]);
        }
        return builder.toString();
    }

    private static String join(
            Object[] array,
            String separator
    ) {
        StringBuilder builder = new StringBuilder(
                Math.max(16, array.length * 8)
        );
        for (int index = 0; index < array.length; index++) {
            if (index > 0) {
                builder.append(separator);
            }
            builder.append(array[index]);
        }
        return builder.toString();
    }

    private static double[] requireRow(
            double[] row,
            int instanceIndex,
            int rowIndex
    ) {
        return Objects.requireNonNull(
                row,
                "Null double row at dataset instance "
                        + instanceIndex
                        + ", row "
                        + rowIndex
                        + "."
        );
    }

    private static float[] requireRow(
            float[] row,
            int instanceIndex,
            int rowIndex
    ) {
        return Objects.requireNonNull(
                row,
                "Null float row at dataset instance "
                        + instanceIndex
                        + ", row "
                        + rowIndex
                        + "."
        );
    }

    private static Object[] requireRow(
            Object[] row,
            int instanceIndex,
            int rowIndex
    ) {
        return Objects.requireNonNull(
                row,
                "Null object row at dataset instance "
                        + instanceIndex
                        + ", row "
                        + rowIndex
                        + "."
        );
    }
}
