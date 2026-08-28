package output;

import java.io.BufferedWriter;
import java.io.IOException;
import java.io.Writer;
import java.nio.charset.Charset;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.List;
import java.util.Objects;

/**
 * Shared utilities for writing standards-compatible CSV output.
 *
 * This class is intended for row-oriented PFGAP artifacts such as:
 *
 *     predictions
 *     outlier scores
 *     dense proximity matrices
 *     repeat-level result summaries
 *
 * CSV escaping follows the conventional rules:
 *
 *     1. Null values are written as empty fields.
 *
 *     2. Fields containing commas, quotation marks, carriage returns,
 *        or line feeds are enclosed in double quotation marks.
 *
 *     3. A quotation mark inside a quoted field is escaped by writing
 *        it twice.
 *
 * For example:
 *
 *     plain
 *
 * remains:
 *
 *     plain
 *
 * while:
 *
 *     value,with,commas
 *
 * becomes:
 *
 *     "value,with,commas"
 *
 * and:
 *
 *     value "with quotes"
 *
 * becomes:
 *
 *     "value ""with quotes"""
 *
 * This utility writes UTF-8 by default and uses a comma as the delimiter.
 *
 * The class is stateless and thread-safe. A single Writer instance should
 * not be used concurrently unless callers provide their own synchronization.
 */
public final class CsvUtils {

    public static final char DEFAULT_DELIMITER =
            ',';

    public static final Charset DEFAULT_CHARSET =
            StandardCharsets.UTF_8;

    /**
     * CRLF is the traditional CSV record separator and is broadly supported
     * by spreadsheet, database, and scientific-data tools.
     */
    public static final String RECORD_SEPARATOR =
            "\r\n";

    private CsvUtils() {
    }

    /**
     * Opens a UTF-8 CSV writer, creating the parent directory when needed.
     *
     * An existing file is replaced.
     *
     * @param path output CSV path
     * @return buffered UTF-8 writer
     * @throws IOException if the directory or file cannot be opened
     */
    public static BufferedWriter newWriter(
            Path path
    ) throws IOException {
        return newWriter(
                path,
                DEFAULT_CHARSET,
                false
        );
    }

    /**
     * Opens a CSV writer with a configurable charset and append behavior.
     *
     * When append is false, any existing file is truncated. When append is
     * true, new records are added to the end of the file.
     *
     * This method does not determine whether an appended file already
     * contains a header. Callers using append mode must decide whether a
     * header should be written.
     *
     * @param path output CSV path
     * @param charset output character encoding
     * @param append whether to append rather than replace
     * @return buffered writer
     * @throws IOException if the directory or file cannot be opened
     */
    public static BufferedWriter newWriter(
            Path path,
            Charset charset,
            boolean append
    ) throws IOException {
        Objects.requireNonNull(
                path,
                "CSV output path cannot be null."
        );

        Objects.requireNonNull(
                charset,
                "CSV output charset cannot be null."
        );

        Path normalizedPath =
                path.toAbsolutePath()
                        .normalize();

        Path parent =
                normalizedPath.getParent();

        if (parent != null) {
            Files.createDirectories(parent);
        }

        if (append) {
            return Files.newBufferedWriter(
                    normalizedPath,
                    charset,
                    StandardOpenOption.CREATE,
                    StandardOpenOption.WRITE,
                    StandardOpenOption.APPEND
            );
        }

        return Files.newBufferedWriter(
                normalizedPath,
                charset,
                StandardOpenOption.CREATE,
                StandardOpenOption.WRITE,
                StandardOpenOption.TRUNCATE_EXISTING
        );
    }

    /**
     * Escapes a value for use as one CSV field.
     *
     * Null is represented as an empty field.
     *
     * @param value field value
     * @return escaped CSV field text
     */
    public static String escape(
            Object value
    ) {
        return escape(
                value,
                DEFAULT_DELIMITER
        );
    }

    /**
     * Escapes a value for a delimited file using a specified delimiter.
     *
     * The delimiter must not be a quotation mark, carriage return, or line
     * feed.
     *
     * @param value field value
     * @param delimiter field delimiter
     * @return escaped field text
     */
    public static String escape(
            Object value,
            char delimiter
    ) {
        validateDelimiter(delimiter);

        if (value == null) {
            return "";
        }

        String text =
                String.valueOf(value);

        boolean requiresQuoting =
                text.indexOf(delimiter) >= 0
                        || text.indexOf('"') >= 0
                        || text.indexOf('\r') >= 0
                        || text.indexOf('\n') >= 0;

        if (!requiresQuoting) {
            return text;
        }

        String escapedQuotes =
                text.replace(
                        "\"",
                        "\"\""
                );

        return '"'
                + escapedQuotes
                + '"';
    }

    /**
     * Writes one CSV row using the default comma delimiter.
     *
     * A record separator is written after the row.
     *
     * @param writer destination writer
     * @param values row values
     * @throws IOException if writing fails
     */
    public static void writeRow(
            Writer writer,
            Object... values
    ) throws IOException {
        writeRow(
                writer,
                DEFAULT_DELIMITER,
                values
        );
    }

    /**
     * Writes one delimited row using a specified delimiter.
     *
     * A record separator is written after the row.
     *
     * @param writer destination writer
     * @param delimiter field delimiter
     * @param values row values
     * @throws IOException if writing fails
     */
    public static void writeRow(
            Writer writer,
            char delimiter,
            Object... values
    ) throws IOException {
        Objects.requireNonNull(
                writer,
                "CSV Writer cannot be null."
        );

        validateDelimiter(delimiter);

        if (values == null) {
            writer.write(
                    RECORD_SEPARATOR
            );

            return;
        }

        for (int index = 0;
             index < values.length;
             index++) {

            if (index > 0) {
                writer.write(delimiter);
            }

            writer.write(
                    escape(
                            values[index],
                            delimiter
                    )
            );
        }

        writer.write(
                RECORD_SEPARATOR
        );
    }

    /**
     * Writes one CSV row from a list using the default comma delimiter.
     *
     * @param writer destination writer
     * @param values row values
     * @throws IOException if writing fails
     */
    public static void writeRow(
            Writer writer,
            List<?> values
    ) throws IOException {
        writeRow(
                writer,
                DEFAULT_DELIMITER,
                values
        );
    }

    /**
     * Writes one delimited row from a list.
     *
     * @param writer destination writer
     * @param delimiter field delimiter
     * @param values row values
     * @throws IOException if writing fails
     */
    public static void writeRow(
            Writer writer,
            char delimiter,
            List<?> values
    ) throws IOException {
        Objects.requireNonNull(
                writer,
                "CSV Writer cannot be null."
        );

        validateDelimiter(delimiter);

        if (values == null) {
            writer.write(
                    RECORD_SEPARATOR
            );

            return;
        }

        for (int index = 0;
             index < values.size();
             index++) {

            if (index > 0) {
                writer.write(delimiter);
            }

            writer.write(
                    escape(
                            values.get(index),
                            delimiter
                    )
            );
        }

        writer.write(
                RECORD_SEPARATOR
        );
    }

    /**
     * Writes a header row using the default comma delimiter.
     *
     * Header names must be non-null and nonblank.
     *
     * @param writer destination writer
     * @param columnNames ordered column names
     * @throws IOException if writing fails
     */
    public static void writeHeader(
            Writer writer,
            String... columnNames
    ) throws IOException {
        writeHeader(
                writer,
                DEFAULT_DELIMITER,
                columnNames
        );
    }

    /**
     * Writes a header row using a specified delimiter.
     *
     * Header names must be non-null and nonblank.
     *
     * @param writer destination writer
     * @param delimiter field delimiter
     * @param columnNames ordered column names
     * @throws IOException if writing fails
     */
    public static void writeHeader(
            Writer writer,
            char delimiter,
            String... columnNames
    ) throws IOException {
        validateColumnNames(
                columnNames
        );

        writeRow(
                writer,
                delimiter,
                (Object[]) columnNames
        );
    }

    /**
     * Writes a header row from a list using the default comma delimiter.
     *
     * @param writer destination writer
     * @param columnNames ordered column names
     * @throws IOException if writing fails
     */
    public static void writeHeader(
            Writer writer,
            List<String> columnNames
    ) throws IOException {
        writeHeader(
                writer,
                DEFAULT_DELIMITER,
                columnNames
        );
    }

    /**
     * Writes a header row from a list using a specified delimiter.
     *
     * @param writer destination writer
     * @param delimiter field delimiter
     * @param columnNames ordered column names
     * @throws IOException if writing fails
     */
    public static void writeHeader(
            Writer writer,
            char delimiter,
            List<String> columnNames
    ) throws IOException {
        Objects.requireNonNull(
                columnNames,
                "CSV column names cannot be null."
        );

        validateColumnNames(
                columnNames.toArray(
                        new String[0]
                )
        );

        writeRow(
                writer,
                delimiter,
                columnNames
        );
    }

    /**
     * Writes a complete CSV file containing a header and rows.
     *
     * This convenience method is appropriate for small row-oriented
     * outputs. Large outputs should use newWriter(...) and stream rows
     * directly rather than constructing a complete iterable in memory.
     *
     * @param path output CSV path
     * @param header ordered column names
     * @param rows row collection
     * @throws IOException if writing fails
     */
    public static void writeFile(
            Path path,
            List<String> header,
            Iterable<? extends List<?>> rows
    ) throws IOException {
        Objects.requireNonNull(
                rows,
                "CSV rows cannot be null."
        );

        try (BufferedWriter writer =
                     newWriter(path)) {

            if (header != null
                    && !header.isEmpty()) {

                writeHeader(
                        writer,
                        header
                );
            }

            for (List<?> row : rows) {
                writeRow(
                        writer,
                        row
                );
            }
        }
    }

    /**
     * Returns whether a file exists and contains at least one byte.
     *
     * This can be used by append-mode writers to decide whether they need
     * to emit a header.
     *
     * @param path path to inspect
     * @return true when the path is a nonempty regular file
     * @throws IOException if file metadata cannot be inspected
     */
    public static boolean isNonemptyFile(
            Path path
    ) throws IOException {
        Objects.requireNonNull(
                path,
                "CSV path cannot be null."
        );

        return Files.isRegularFile(path)
                && Files.size(path) > 0L;
    }

    /**
     * Converts a finite or non-finite double into a stable CSV field value.
     *
     * Finite numbers are returned as Double values. NaN and infinities are
     * rendered as explicit strings so the CSV remains readable and does not
     * silently convert them into blank missing values.
     *
     * @param value numeric value
     * @return Double for finite values, otherwise a descriptive String
     */
    public static Object numericField(
            double value
    ) {
        if (Double.isNaN(value)) {
            return "NaN";
        }

        if (value == Double.POSITIVE_INFINITY) {
            return "Infinity";
        }

        if (value == Double.NEGATIVE_INFINITY) {
            return "-Infinity";
        }

        return value;
    }

    /**
     * Converts a boxed numeric value into a stable CSV field value.
     *
     * Null remains null and is written as an empty CSV field.
     *
     * @param value boxed numeric value
     * @return CSV-ready value
     */
    public static Object numericField(
            Double value
    ) {
        if (value == null) {
            return null;
        }

        return numericField(
                value.doubleValue()
        );
    }

    private static void validateDelimiter(
            char delimiter
    ) {
        if (delimiter == '"'
                || delimiter == '\r'
                || delimiter == '\n') {

            throw new IllegalArgumentException(
                    "CSV delimiter cannot be a quotation mark, "
                            + "carriage return, or line feed."
            );
        }
    }

    private static void validateColumnNames(
            String[] columnNames
    ) {
        Objects.requireNonNull(
                columnNames,
                "CSV column names cannot be null."
        );

        if (columnNames.length == 0) {
            throw new IllegalArgumentException(
                    "CSV header must contain at least one column."
            );
        }

        for (int index = 0;
             index < columnNames.length;
             index++) {

            String columnName =
                    columnNames[index];

            if (columnName == null
                    || columnName.isBlank()) {

                throw new IllegalArgumentException(
                        "CSV column name at index "
                                + index
                                + " cannot be null or blank."
                );
            }
        }
    }
}