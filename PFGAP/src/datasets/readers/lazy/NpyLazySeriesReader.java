package datasets.readers.lazy;

import datasets.NumericStorageType;
import preprocessing.standardization.StandardizationStats;
import preprocessing.standardization.Standardizer;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.Arrays;
import java.util.Objects;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Lazily materializes one axis-0 observation from a shared C-order NPY file.
 *
 * <p>The NPY header is parsed once. Each read memory maps exactly one
 * contiguous observation and copies directly into a dimension-major primitive
 * matrix. The reader supports float32 and float64 arrays shaped [N,T] or
 * [N,D,T].</p>
 */
public final class NpyLazySeriesReader
        implements LazySeriesReader, AutoCloseable {
    private static final byte[] MAGIC = {
            (byte) 0x93, 'N', 'U', 'M', 'P', 'Y'
    };
    private static final int MAX_HEADER_SIZE = 16 * 1024 * 1024;
    private static final Pattern DESCR = Pattern.compile(
            "['\"]descr['\"]\\s*:\\s*['\"]([^'\"]+)['\"]");
    private static final Pattern FORTRAN = Pattern.compile(
            "['\"]fortran_order['\"]\\s*:\\s*(True|False)");
    private static final Pattern SHAPE = Pattern.compile(
            "['\"]shape['\"]\\s*:\\s*\\(([^)]*)\\)");
    private static final Pattern DTYPE = Pattern.compile("([<>=|])([fF])(4|8)");

    private final NumericStorageType requestedType;
    private final StandardizationStats standardizationStats;
    private final Object initializationLock = new Object();

    private volatile State state;
    private volatile boolean closed;

    public NpyLazySeriesReader(
            NumericStorageType requestedType,
            StandardizationStats standardizationStats
    ) {
        this.requestedType = Objects.requireNonNull(
                requestedType, "numericStorageType cannot be null.");
        this.standardizationStats = standardizationStats;
    }

    @Override
    public Object read(LazySeriesRef reference) {
        if (reference == null) {
            throw new IllegalArgumentException("Cannot read null LazySeriesRef.");
        }
        State active;
        try {
            active = stateFor(reference.getFile());
            int index = reference.getIndex();
            if (index < 0 || index >= active.layout.instances()) {
                throw new IndexOutOfBoundsException(
                        "NPY instance index " + index + " is outside [0, "
                                + active.layout.instances() + ").");
            }
            Object result = active.outputType == NumericStorageType.FLOAT32
                    ? readFloat(active, index)
                    : readDouble(active, index);
            if (standardizationStats != null) {
                Standardizer.transformInstanceInPlace(
                        result, standardizationStats);
            }
            return result;
        } catch (IOException e) {
            throw new IllegalStateException(
                    "Failed to materialize NPY instance "
                            + reference.getIndex() + " from "
                            + reference.getFile(), e);
        }
    }

    public static Inspection inspect(Path file) throws IOException {
        validateFile(file);
        try (FileChannel channel = FileChannel.open(
                file, StandardOpenOption.READ)) {
            Header header = readHeader(channel);
            Layout layout = Layout.of(header.shape());
            validatePayload(channel, header, layout.totalElements());
            return new Inspection(layout.instances(), layout.dimensions(),
                    layout.positions(), header.dtype().bytes(),
                    header.fortranOrder());
        }
    }

    private State stateFor(Path path) throws IOException {
        if (closed) {
            throw new IllegalStateException("NPY lazy reader is closed.");
        }
        Path normalized = path.toAbsolutePath().normalize();
        State active = state;
        if (active != null) {
            if (!active.path.equals(normalized)) {
                throw new IllegalArgumentException(
                        "One NPY lazy reader key cannot reference multiple "
                                + "feature files: " + active.path + " and "
                                + normalized + ".");
            }
            return active;
        }
        synchronized (initializationLock) {
            active = state;
            if (active == null) {
                validateFile(normalized);
                FileChannel channel = FileChannel.open(
                        normalized, StandardOpenOption.READ);
                try {
                    Header header = readHeader(channel);
                    Layout layout = Layout.of(header.shape());
                    if (header.fortranOrder()) {
                        throw new IOException(
                                "Lazy NPY currently requires C-order data. "
                                        + "Fortran-order observations are not "
                                        + "contiguous on disk.");
                    }
                    validatePayload(channel, header,
                            layout.totalElements());
                    NumericStorageType output = requestedType
                            == NumericStorageType.AUTO
                            ? (header.dtype().bytes() == Float.BYTES
                            ? NumericStorageType.FLOAT32
                            : NumericStorageType.FLOAT64)
                            : requestedType;
                    active = new State(normalized, channel, header, layout,
                            output);
                    state = active;
                } catch (IOException | RuntimeException | Error failure) {
                    channel.close();
                    throw failure;
                }
            }
            return active;
        }
    }

    private static float[][] readFloat(State state, int index)
            throws IOException {
        Layout layout = state.layout;
        float[][] result = new float[layout.dimensions()][layout.positions()];
        MappedByteBuffer mapped = mapObservation(state, index);
        if (state.header.dtype().bytes() == Float.BYTES) {
            for (int d = 0; d < result.length; d++) {
                for (int t = 0; t < result[d].length; t++) {
                    result[d][t] = mapped.getFloat();
                }
            }
        } else {
            for (int d = 0; d < result.length; d++) {
                for (int t = 0; t < result[d].length; t++) {
                    result[d][t] = (float) mapped.getDouble();
                }
            }
        }
        return result;
    }

    private static double[][] readDouble(State state, int index)
            throws IOException {
        Layout layout = state.layout;
        double[][] result = new double[layout.dimensions()][layout.positions()];
        MappedByteBuffer mapped = mapObservation(state, index);
        if (state.header.dtype().bytes() == Double.BYTES) {
            for (int d = 0; d < result.length; d++) {
                for (int t = 0; t < result[d].length; t++) {
                    result[d][t] = mapped.getDouble();
                }
            }
        } else {
            for (int d = 0; d < result.length; d++) {
                for (int t = 0; t < result[d].length; t++) {
                    result[d][t] = mapped.getFloat();
                }
            }
        }
        return result;
    }

    private static MappedByteBuffer mapObservation(State state, int index)
            throws IOException {
        long bytes = Math.multiplyExact(
                state.layout.elementsPerInstance(),
                state.header.dtype().bytes());
        if (bytes > Integer.MAX_VALUE) {
            throw new IOException(
                    "One NPY observation exceeds Java's mapped-buffer limit: "
                            + bytes + " bytes.");
        }
        long offset = Math.addExact(state.header.dataOffset(),
                Math.multiplyExact((long) index, bytes));
        MappedByteBuffer mapped = state.channel.map(
                FileChannel.MapMode.READ_ONLY, offset, bytes);
        mapped.order(state.header.dtype().order());
        return mapped;
    }

    @Override
    public void close() throws IOException {
        synchronized (initializationLock) {
            closed = true;
            State active = state;
            state = null;
            if (active != null) {
                active.channel.close();
            }
        }
    }

    private static Header readHeader(FileChannel channel) throws IOException {
        channel.position(0L);
        ByteBuffer prefix = ByteBuffer.allocate(12);
        prefix.limit(8);
        readFully(channel, prefix);
        prefix.flip();
        for (byte expected : MAGIC) {
            if (prefix.get() != expected) {
                throw new IOException("Invalid NPY magic bytes.");
            }
        }
        int major = Byte.toUnsignedInt(prefix.get());
        int minor = Byte.toUnsignedInt(prefix.get());
        int lengthBytes = major == 1 ? 2 : 4;
        if (major < 1 || major > 3) {
            throw new IOException(
                    "Unsupported NPY version: " + major + "." + minor);
        }
        ByteBuffer length = ByteBuffer.allocate(lengthBytes)
                .order(ByteOrder.LITTLE_ENDIAN);
        readFully(channel, length);
        length.flip();
        long headerLength = lengthBytes == 2
                ? Short.toUnsignedLong(length.getShort())
                : Integer.toUnsignedLong(length.getInt());
        if (headerLength <= 0 || headerLength > MAX_HEADER_SIZE) {
            throw new IOException("Invalid NPY header length: " + headerLength);
        }
        ByteBuffer headerBytes = ByteBuffer.allocate((int) headerLength);
        readFully(channel, headerBytes);
        String text = new String(headerBytes.array(),
                major >= 3 ? StandardCharsets.UTF_8
                        : StandardCharsets.ISO_8859_1);
        Matcher descr = DESCR.matcher(text);
        Matcher fortran = FORTRAN.matcher(text);
        Matcher shape = SHAPE.matcher(text);
        if (!descr.find() || !fortran.find() || !shape.find()) {
            throw new IOException("Malformed NPY header dictionary.");
        }
        return new Header(parseDType(descr.group(1)),
                Boolean.parseBoolean(fortran.group(1).toLowerCase()),
                parseShape(shape.group(1)), channel.position());
    }

    private static DType parseDType(String text) throws IOException {
        Matcher matcher = DTYPE.matcher(text.trim());
        if (!matcher.matches()) {
            throw new IOException(
                    "Lazy NPY supports only float32 and float64, received: "
                            + text);
        }
        int bytes = Integer.parseInt(matcher.group(3));
        String marker = matcher.group(1);
        ByteOrder order = marker.equals(">")
                ? ByteOrder.BIG_ENDIAN : ByteOrder.LITTLE_ENDIAN;
        return new DType(bytes, order);
    }

    private static long[] parseShape(String text) throws IOException {
        String[] tokens = text.split(",");
        long[] temporary = new long[tokens.length];
        int count = 0;
        for (String token : tokens) {
            String trimmed = token.trim();
            if (!trimmed.isEmpty()) {
                try {
                    long value = Long.parseLong(trimmed);
                    if (value <= 0) {
                        throw new IOException(
                                "NPY dimensions must be positive: " + text);
                    }
                    temporary[count++] = value;
                } catch (NumberFormatException e) {
                    throw new IOException("Invalid NPY shape: " + text, e);
                }
            }
        }
        return Arrays.copyOf(temporary, count);
    }

    private static void validatePayload(
            FileChannel channel, Header header, long elements
    ) throws IOException {
        long expected = Math.addExact(header.dataOffset(),
                Math.multiplyExact(elements, header.dtype().bytes()));
        if (channel.size() < expected) {
            throw new IOException(
                    "NPY payload is truncated. Expected at least " + expected
                            + " bytes, found " + channel.size() + ".");
        }
    }

    private static void validateFile(Path file) throws IOException {
        if (file == null || !Files.isRegularFile(file)
                || !Files.isReadable(file)) {
            throw new IOException("NPY file is not readable: " + file);
        }
    }

    private static void readFully(FileChannel channel, ByteBuffer buffer)
            throws IOException {
        while (buffer.hasRemaining()) {
            if (channel.read(buffer) < 0) {
                throw new IOException("Unexpected end of NPY file.");
            }
        }
    }

    public record Inspection(
            int instances,
            int dimensions,
            int positions,
            int sourceBytes,
            boolean fortranOrder
    ) {
    }

    private record Header(
            DType dtype,
            boolean fortranOrder,
            long[] shape,
            long dataOffset
    ) {
    }

    private record DType(int bytes, ByteOrder order) {
    }

    private record Layout(
            int instances,
            int dimensions,
            int positions,
            long elementsPerInstance,
            long totalElements
    ) {
        private static Layout of(long[] shape) throws IOException {
            if (shape.length != 2 && shape.length != 3) {
                throw new IOException(
                        "Lazy NPY features require shape [N,T] or [N,D,T], "
                                + "received rank " + shape.length + ".");
            }
            int instances = toInt(shape[0], "instance count");
            int dimensions = shape.length == 2
                    ? 1 : toInt(shape[1], "dimension count");
            int positions = toInt(shape[shape.length - 1], "position count");
            long perInstance = Math.multiplyExact(
                    (long) dimensions, positions);
            return new Layout(instances, dimensions, positions, perInstance,
                    Math.multiplyExact((long) instances, perInstance));
        }

        private static int toInt(long value, String role) throws IOException {
            if (value <= 0 || value > Integer.MAX_VALUE) {
                throw new IOException(role + " is outside Java int range: "
                        + value);
            }
            return (int) value;
        }
    }

    private record State(
            Path path,
            FileChannel channel,
            Header header,
            Layout layout,
            NumericStorageType outputType
    ) {
    }
}
