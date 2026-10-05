package datasets.readers.lazy;

import core.AppContext;
import datasets.ListObjectDataset;
import datasets.NumericStorageType;
import datasets.readers.DatasetReader;
import datasets.readers.DelimitedFileReader;
import datasets.readers.NumericPerFileParquetSeriesReader;
import datasets.readers.ReaderOptions;
import datasets.readers.ReaderType;
import preprocessing.standardization.StandardizationStats;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Lazy reader for one shared C-order NPY feature array shaped [N,T] or
 * [N,D,T]. Each dataset item references the same file and selects its
 * observation through {@link LazySeriesRef#getIndex()}.
 */
public final class LazyNpyReader implements DatasetReader {
    private final Path dataPath;
    private final Path labelPath;
    private final boolean labelHeader;
    private final boolean regression;
    private final String readerKey;
    private final NumericStorageType numericStorageType;
    private final StandardizationStats standardizationStats;

    public LazyNpyReader(ReaderOptions options) {
        this(requiredPath(requireOptions(options).getDataPath(), "dataPath"),
                optionalPath(options.getLabelPath()), options.hasHeader(),
                options.isRegression(), options.isTest() ? "test" : "train",
                options.getNumericStorageType(),
                options.getStandardizationStats());
        if (!options.isNumeric()) {
            throw new IllegalArgumentException(
                    "Lazy NPY feature reading is numeric-only.");
        }
    }

    public LazyNpyReader(
            Path dataPath,
            Path labelPath,
            boolean labelHeader,
            boolean regression,
            String readerKey,
            NumericStorageType numericStorageType,
            StandardizationStats standardizationStats
    ) {
        this.dataPath = Objects.requireNonNull(dataPath,
                "dataPath cannot be null.").toAbsolutePath().normalize();
        this.labelPath = labelPath == null ? null
                : labelPath.toAbsolutePath().normalize();
        this.labelHeader = labelHeader;
        this.regression = regression;
        if (readerKey == null || readerKey.isBlank()) {
            throw new IllegalArgumentException("readerKey cannot be blank.");
        }
        this.readerKey = readerKey.trim();
        this.numericStorageType = Objects.requireNonNull(
                numericStorageType, "numericStorageType cannot be null.");
        this.standardizationStats = standardizationStats;
    }

    @Override
    public ListObjectDataset read() throws IOException {
        NpyLazySeriesReader.Inspection inspection =
                NpyLazySeriesReader.inspect(dataPath);
        if (inspection.fortranOrder()) {
            throw new IOException(
                    "Lazy NPY requires C-order features for contiguous "
                            + "per-instance memory mapping.");
        }
        List<Object> labels = readLabels();
        if (!labels.isEmpty() && labels.size() != inspection.instances()) {
            throw new IOException(
                    "Label count " + labels.size()
                            + " does not match NPY instance count "
                            + inspection.instances() + ".");
        }

        LazySeriesReaderSpec spec = new LazySeriesReaderSpec(
                readerKey,
                ReaderType.LAZY_NPY,
                null,
                List.of(),
                true,
                true,
                null,
                false,
                standardizationStats,
                LazySeriesReaderSpec.DEFAULT_INITIAL_TIME_CAPACITY,
                numericStorageType,
                NumericPerFileParquetSeriesReader.TimeOrderPolicy.FILE_ORDER
        );
        AppContext.registerLazySeriesReader(spec);

        ListObjectDataset dataset =
                new ListObjectDataset(inspection.instances());
        for (int index = 0; index < inspection.instances(); index++) {
            Object label = labels.isEmpty() ? null : labels.get(index);
            dataset.add(label,
                    new LazySeriesRef(readerKey, index, dataPath), index);
        }
        dataset.setLength(inspection.positions());
        AppContext.length = inspection.positions();
        return dataset;
    }

    private List<Object> readLabels() throws IOException {
        if (labelPath == null) {
            return List.of();
        }
        if (!labelPath.getFileName().toString()
                .toLowerCase(java.util.Locale.ROOT).endsWith(".npy")) {
            return DelimitedFileReader.readGenericLabels(
                    labelPath.toString(), labelHeader, regression);
        }
        return NpyLabels.read(labelPath, regression);
    }

    private static ReaderOptions requireOptions(ReaderOptions options) {
        if (options == null) {
            throw new IllegalArgumentException(
                    "LazyNpyReader requires ReaderOptions.");
        }
        return options;
    }

    private static Path requiredPath(String value, String role) {
        if (value == null || value.isBlank()) {
            throw new IllegalArgumentException(role + " cannot be blank.");
        }
        return Path.of(value);
    }

    private static Path optionalPath(String value) {
        return value == null || value.isBlank()
                || value.equalsIgnoreCase("None") ? null : Path.of(value);
    }

    private static final class NpyLabels {
        private static final byte[] MAGIC = {
                (byte) 0x93, 'N', 'U', 'M', 'P', 'Y'
        };
        private static final Pattern DESCR = Pattern.compile(
                "['\"]descr['\"]\\s*:\\s*['\"]([^'\"]+)['\"]");
        private static final Pattern SHAPE = Pattern.compile(
                "['\"]shape['\"]\\s*:\\s*\\(([^)]*)\\)");
        private static final Pattern TYPE = Pattern.compile(
                "([<>=|])([iufb?])(1|2|4|8)");

        private static List<Object> read(Path file, boolean regression)
                throws IOException {
            if (!Files.isRegularFile(file) || !Files.isReadable(file)) {
                throw new IOException("NPY label file is not readable: " + file);
            }
            try (FileChannel channel = FileChannel.open(
                    file, StandardOpenOption.READ)) {
                LabelHeader header = header(channel);
                ByteBuffer values = ByteBuffer.allocate(
                                Math.toIntExact(Math.multiplyExact(
                                        header.count(), header.bytes())))
                        .order(header.order());
                readFully(channel, values);
                values.flip();
                List<Object> result = new ArrayList<>(header.count());
                for (int i = 0; i < header.count(); i++) {
                    result.add(readValue(values, header, regression));
                }
                return result;
            }
        }

        private static Object readValue(
                ByteBuffer buffer, LabelHeader h, boolean regression
        ) {
            double value = switch (h.kind()) {
                case 'f' -> h.bytes() == 4
                        ? buffer.getFloat() : buffer.getDouble();
                case 'b', '?' -> buffer.get() == 0 ? 0.0 : 1.0;
                case 'i' -> switch (h.bytes()) {
                    case 1 -> buffer.get();
                    case 2 -> buffer.getShort();
                    case 4 -> buffer.getInt();
                    case 8 -> buffer.getLong();
                    default -> throw new AssertionError();
                };
                case 'u' -> switch (h.bytes()) {
                    case 1 -> Byte.toUnsignedInt(buffer.get());
                    case 2 -> Short.toUnsignedInt(buffer.getShort());
                    case 4 -> Integer.toUnsignedLong(buffer.getInt());
                    case 8 -> {
                        long v = buffer.getLong();
                        if (v < 0) {
                            throw new IllegalArgumentException(
                                    "uint64 label exceeds Java long range.");
                        }
                        yield v;
                    }
                    default -> throw new AssertionError();
                };
                default -> throw new AssertionError();
            };
            if (regression || h.kind() == 'f') {
                return value;
            }
            long integer = (long) value;
            return integer >= Integer.MIN_VALUE && integer <= Integer.MAX_VALUE
                    ? (int) integer : integer;
        }

        private static LabelHeader header(FileChannel channel)
                throws IOException {
            ByteBuffer prefix = ByteBuffer.allocate(8);
            readFully(channel, prefix);
            prefix.flip();
            for (byte b : MAGIC) {
                if (prefix.get() != b) {
                    throw new IOException("Invalid NPY label magic bytes.");
                }
            }
            int major = Byte.toUnsignedInt(prefix.get());
            prefix.get();
            int lengthBytes = major == 1 ? 2 : 4;
            ByteBuffer length = ByteBuffer.allocate(lengthBytes)
                    .order(ByteOrder.LITTLE_ENDIAN);
            readFully(channel, length);
            length.flip();
            int n = lengthBytes == 2
                    ? Short.toUnsignedInt(length.getShort()) : length.getInt();
            ByteBuffer textBytes = ByteBuffer.allocate(n);
            readFully(channel, textBytes);
            String text = new String(textBytes.array(),
                    major >= 3 ? StandardCharsets.UTF_8
                            : StandardCharsets.ISO_8859_1);
            Matcher descr = DESCR.matcher(text);
            Matcher shape = SHAPE.matcher(text);
            if (!descr.find() || !shape.find()) {
                throw new IOException("Malformed NPY label header.");
            }
            String[] dimensions = shape.group(1).split(",");
            int count = 0;
            int rank = 0;
            for (String dimension : dimensions) {
                if (!dimension.trim().isEmpty()) {
                    count = Integer.parseInt(dimension.trim());
                    rank++;
                }
            }
            if (rank != 1 || count < 1) {
                throw new IOException("NPY labels must have shape [N].");
            }
            Matcher type = TYPE.matcher(descr.group(1));
            if (!type.matches()) {
                throw new IOException("Unsupported NPY label dtype: "
                        + descr.group(1));
            }
            ByteOrder order = type.group(1).equals(">")
                    ? ByteOrder.BIG_ENDIAN : ByteOrder.LITTLE_ENDIAN;
            return new LabelHeader(count, type.group(2).charAt(0),
                    Integer.parseInt(type.group(3)), order);
        }

        private static void readFully(
                FileChannel channel, ByteBuffer buffer
        ) throws IOException {
            while (buffer.hasRemaining()) {
                if (channel.read(buffer) < 0) {
                    throw new IOException("Unexpected end of NPY label file.");
                }
            }
        }

        private record LabelHeader(
                int count, char kind, int bytes, ByteOrder order
        ) {
        }
    }
}
