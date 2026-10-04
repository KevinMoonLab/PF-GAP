package datasets.readers;

import core.AppContext;
import datasets.ListObjectDataset;
import datasets.NumericStorageType;

import java.io.EOFException;
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
import java.util.Locale;
import java.util.Objects;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Eager NumPy NPY reader for primitive numeric feature arrays.
 *
 * <p>Supported feature layouts are:</p>
 * <ul>
 *     <li>Rank 1: one one-dimensional observation, {@code [feature]}</li>
 *     <li>Rank 2: multiple one-dimensional observations,
 *         {@code [instance][feature]}</li>
 *     <li>Rank 3: multiple two-dimensional observations,
 *         {@code [instance][dimension][position]}</li>
 * </ul>
 *
 * <p>A one-dimensional observation may represent a tabular feature vector, a
 * univariate time series, or another ordered numeric vector. The Java array
 * type does not impose time-series semantics.</p>
 *
 * <p>Feature dtypes must be float32 or float64. AUTO preserves the source
 * precision. Explicit FLOAT32 or FLOAT64 converts into the requested primitive
 * representation. C-order and Fortran-order NPY payloads are supported.</p>
 *
 * <p>IEEE NaN feature values are decoded and preserved regardless of the
 * {@code hasMissingValues} configuration flag. This is intentional: NPY is a
 * typed binary format, and decoding preserves its source values without an
 * additional validation branch in the hot loop. Dataset missingness discovery
 * can therefore record the original NaN positions, and the ordinary PFGAP
 * imputation pipeline can impute them.</p>
 *
 * <p>Labels may be supplied by a rank-1 NPY array or by a separate delimited
 * label file. Rank-2 multi-label and multi-target NPY label arrays are deferred
 * to the pre-v1 multi-target implementation.</p>
 */
public final class NpyReader implements DatasetReader {
    private static final byte[] MAGIC = {
            (byte) 0x93, 'N', 'U', 'M', 'P', 'Y'
    };
    private static final int MAX_HEADER_SIZE = 16 * 1024 * 1024;
    private static final int BUFFER_SIZE = 8 * 1024 * 1024;

    private static final Pattern DESCR_PATTERN = Pattern.compile(
            "['\"]descr['\"]\\s*:\\s*['\"]([^'\"]+)['\"]");
    private static final Pattern FORTRAN_PATTERN = Pattern.compile(
            "['\"]fortran_order['\"]\\s*:\\s*(True|False)");
    private static final Pattern SHAPE_PATTERN = Pattern.compile(
            "['\"]shape['\"]\\s*:\\s*\\(([^)]*)\\)");
    private static final Pattern DTYPE_PATTERN = Pattern.compile(
            "([<>=|])?([fiu])(1|2|4|8)");

    private final Path dataPath;
    private final Path labelPath;
    private final boolean labelHeader;
    private final boolean regression;
    private final NumericStorageType requestedType;

    public NpyReader(ReaderOptions options) {
        Objects.requireNonNull(options, "ReaderOptions cannot be null.");
        this.dataPath = requiredPath(options.getDataPath(), "dataPath");
        this.labelPath = optionalPath(options.getLabelPath());
        this.labelHeader = options.hasHeader();
        this.regression = options.isRegression();
        this.requestedType = Objects.requireNonNull(
                options.getNumericStorageType(),
                "numericStorageType cannot be null.");
        if (!options.isNumeric()) {
            throw new IllegalArgumentException(
                    "NPY feature reading is numeric-only.");
        }
    }

    @Override
    public ListObjectDataset read() throws IOException {
        validateFile(dataPath);
        List<Object> labels = readLabels();

        try (FileChannel channel = FileChannel.open(
                dataPath, StandardOpenOption.READ)) {
            Header header = readHeader(channel);
            validateFeatureDType(header.dtype());
            Layout layout = Layout.of(header.shape());
            validatePayload(channel, header, layout.elements());
            validateLabelCount(labels, layout.instances());

            NumericStorageType outputType = resolveOutputType(
                    requestedType, header.dtype());
            Object[] observations = allocateObservations(layout, outputType);
            decodeFeatures(
                    channel, header, layout, outputType, observations);

            ListObjectDataset dataset =
                    new ListObjectDataset(layout.instances());
            for (int instance = 0;
                 instance < layout.instances();
                 instance++) {
                Object label = labels.isEmpty()
                        ? null
                        : labels.get(instance);
                dataset.add(label, observations[instance], instance);
            }

            dataset.setLength(layout.positionCount());
            AppContext.length = layout.positionCount();
            return dataset;
        }
    }

    private List<Object> readLabels() throws IOException {
        if (labelPath == null) {
            return List.of();
        }
        if (!isNpyFile(labelPath)) {
            return DelimitedFileReader.readGenericLabels(
                    labelPath.toString(), labelHeader, regression);
        }

        validateFile(labelPath);
        try (FileChannel channel = FileChannel.open(
                labelPath, StandardOpenOption.READ)) {
            Header header = readHeader(channel);
            if (header.shape().length != 1) {
                throw new IOException(
                        "NPY labels must currently have shape [N]. "
                                + "Rank-2 multi-label and multi-target arrays "
                                + "are not yet supported.");
            }

            DType dtype = header.dtype();
            validateLabelDType(dtype);
            int labelCount = checkedPositiveInt(
                    header.shape()[0], "label count");
            validatePayload(channel, header, labelCount);

            ArrayList<Object> labels = new ArrayList<>(labelCount);
            channel.position(header.dataOffset());
            ByteBuffer buffer = createBuffer(dtype);
            int remaining = labelCount;
            while (remaining > 0) {
                int batch = Math.min(
                        remaining, buffer.capacity() / dtype.bytes());
                buffer.clear();
                buffer.limit(batch * dtype.bytes());
                readFully(channel, buffer);
                buffer.flip();
                for (int index = 0; index < batch; index++) {
                    labels.add(readLabel(buffer, dtype));
                }
                remaining -= batch;
            }
            return labels;
        }
    }

    private Object readLabel(ByteBuffer buffer, DType dtype)
            throws IOException {
        if (dtype.kind() == 'f') {
            double value = dtype.bytes() == Float.BYTES
                    ? buffer.getFloat()
                    : buffer.getDouble();
            if (!Double.isFinite(value)) {
                throw new IOException(
                        "NPY labels cannot contain NaN or infinity.");
            }
            if (regression) {
                return value;
            }
            if (value != Math.rint(value)
                    || value < Long.MIN_VALUE
                    || value > Long.MAX_VALUE) {
                throw new IOException(
                        "Classification labels stored as floating-point "
                                + "values must be exact integers: " + value);
            }
            return narrowInteger((long) value);
        }

        long value = switch (dtype.bytes()) {
            case Byte.BYTES -> dtype.kind() == 'u'
                    ? Byte.toUnsignedLong(buffer.get())
                    : buffer.get();
            case Short.BYTES -> dtype.kind() == 'u'
                    ? Short.toUnsignedLong(buffer.getShort())
                    : buffer.getShort();
            case Integer.BYTES -> dtype.kind() == 'u'
                    ? Integer.toUnsignedLong(buffer.getInt())
                    : buffer.getInt();
            case Long.BYTES -> {
                long decoded = buffer.getLong();
                if (dtype.kind() == 'u' && decoded < 0) {
                    throw new IOException(
                            "uint64 label exceeds the Java long range.");
                }
                yield decoded;
            }
            default -> throw new IOException(
                    "Unsupported integer label width: " + dtype.bytes());
        };
        return regression ? (double) value : narrowInteger(value);
    }

    private static void decodeFeatures(
            FileChannel channel,
            Header header,
            Layout layout,
            NumericStorageType outputType,
            Object[] observations
    ) throws IOException {
        channel.position(header.dataOffset());
        ByteBuffer buffer = createBuffer(header.dtype());
        long ordinal = 0L;

        while (ordinal < layout.elements()) {
            int batch = (int) Math.min(
                    layout.elements() - ordinal,
                    buffer.capacity() / header.dtype().bytes());
            buffer.clear();
            buffer.limit(batch * header.dtype().bytes());
            readFully(channel, buffer);
            buffer.flip();

            if (header.dtype().bytes() == Float.BYTES) {
                if (outputType == NumericStorageType.FLOAT32) {
                    ordinal = decodeFloatToFloat(
                            buffer, batch, ordinal, header.fortranOrder(),
                            layout, observations);
                } else {
                    ordinal = decodeFloatToDouble(
                            buffer, batch, ordinal, header.fortranOrder(),
                            layout, observations);
                }
            } else if (outputType == NumericStorageType.FLOAT32) {
                ordinal = decodeDoubleToFloat(
                        buffer, batch, ordinal, header.fortranOrder(),
                        layout, observations);
            } else {
                ordinal = decodeDoubleToDouble(
                        buffer, batch, ordinal, header.fortranOrder(),
                        layout, observations);
            }
        }
    }

    private static long decodeFloatToFloat(
            ByteBuffer buffer,
            int batch,
            long ordinal,
            boolean fortranOrder,
            Layout layout,
            Object[] observations
    ) {
        for (int index = 0; index < batch; index++) {
            float value = buffer.getFloat();
            Coordinate coordinate = layout.coordinate(
                    ordinal++, fortranOrder);
            storeFloat(observations, layout, coordinate, value);
        }
        return ordinal;
    }

    private static long decodeFloatToDouble(
            ByteBuffer buffer,
            int batch,
            long ordinal,
            boolean fortranOrder,
            Layout layout,
            Object[] observations
    ) {
        for (int index = 0; index < batch; index++) {
            double value = buffer.getFloat();
            Coordinate coordinate = layout.coordinate(
                    ordinal++, fortranOrder);
            storeDouble(observations, layout, coordinate, value);
        }
        return ordinal;
    }

    private static long decodeDoubleToFloat(
            ByteBuffer buffer,
            int batch,
            long ordinal,
            boolean fortranOrder,
            Layout layout,
            Object[] observations
    ) {
        for (int index = 0; index < batch; index++) {
            float value = (float) buffer.getDouble();
            Coordinate coordinate = layout.coordinate(
                    ordinal++, fortranOrder);
            storeFloat(observations, layout, coordinate, value);
        }
        return ordinal;
    }

    private static long decodeDoubleToDouble(
            ByteBuffer buffer,
            int batch,
            long ordinal,
            boolean fortranOrder,
            Layout layout,
            Object[] observations
    ) {
        for (int index = 0; index < batch; index++) {
            double value = buffer.getDouble();
            Coordinate coordinate = layout.coordinate(
                    ordinal++, fortranOrder);
            storeDouble(observations, layout, coordinate, value);
        }
        return ordinal;
    }

    private static void storeFloat(
            Object[] observations,
            Layout layout,
            Coordinate coordinate,
            float value
    ) {
        if (layout.rank() == 3) {
            ((float[][]) observations[coordinate.instance()])
                    [coordinate.dimension()][coordinate.position()] = value;
        } else {
            ((float[]) observations[coordinate.instance()])
                    [coordinate.position()] = value;
        }
    }

    private static void storeDouble(
            Object[] observations,
            Layout layout,
            Coordinate coordinate,
            double value
    ) {
        if (layout.rank() == 3) {
            ((double[][]) observations[coordinate.instance()])
                    [coordinate.dimension()][coordinate.position()] = value;
        } else {
            ((double[]) observations[coordinate.instance()])
                    [coordinate.position()] = value;
        }
    }

    private static Object[] allocateObservations(
            Layout layout,
            NumericStorageType outputType
    ) {
        Object[] observations = new Object[layout.instances()];
        for (int instance = 0;
             instance < observations.length;
             instance++) {
            if (layout.rank() == 3) {
                observations[instance] =
                        outputType == NumericStorageType.FLOAT32
                                ? new float[layout.dimensions()]
                                [layout.positionCount()]
                                : new double[layout.dimensions()]
                                [layout.positionCount()];
            } else {
                observations[instance] =
                        outputType == NumericStorageType.FLOAT32
                                ? new float[layout.positionCount()]
                                : new double[layout.positionCount()];
            }
        }
        return observations;
    }

    private static NumericStorageType resolveOutputType(
            NumericStorageType requested,
            DType sourceType
    ) {
        if (requested != NumericStorageType.AUTO) {
            return requested;
        }
        return sourceType.bytes() == Float.BYTES
                ? NumericStorageType.FLOAT32
                : NumericStorageType.FLOAT64;
    }

    private static Header readHeader(FileChannel channel) throws IOException {
        ByteBuffer prefix = ByteBuffer.allocate(8);
        readFully(channel, prefix);
        prefix.flip();
        for (byte expected : MAGIC) {
            if (prefix.get() != expected) {
                throw new IOException("Invalid NPY magic prefix.");
            }
        }

        int major = Byte.toUnsignedInt(prefix.get());
        int minor = Byte.toUnsignedInt(prefix.get());
        if (minor != 0 || major < 1 || major > 3) {
            throw new IOException(
                    "Unsupported NPY version " + major + "." + minor + ".");
        }

        int lengthBytes = major == 1 ? Short.BYTES : Integer.BYTES;
        ByteBuffer lengthBuffer = ByteBuffer.allocate(lengthBytes)
                .order(ByteOrder.LITTLE_ENDIAN);
        readFully(channel, lengthBuffer);
        lengthBuffer.flip();
        long headerLength = major == 1
                ? Short.toUnsignedLong(lengthBuffer.getShort())
                : Integer.toUnsignedLong(lengthBuffer.getInt());
        if (headerLength <= 0 || headerLength > MAX_HEADER_SIZE) {
            throw new IOException(
                    "Invalid NPY header length: " + headerLength);
        }

        ByteBuffer headerBytes = ByteBuffer.allocate((int) headerLength);
        readFully(channel, headerBytes);
        headerBytes.flip();
        String headerText = (major == 3
                ? StandardCharsets.UTF_8
                : StandardCharsets.ISO_8859_1)
                .decode(headerBytes)
                .toString();

        String descriptor = requiredGroup(
                DESCR_PATTERN, headerText, "descr");
        boolean fortranOrder = Boolean.parseBoolean(
                requiredGroup(FORTRAN_PATTERN, headerText, "fortran_order")
                        .toLowerCase(Locale.ROOT));
        long[] shape = parseShape(
                requiredGroup(SHAPE_PATTERN, headerText, "shape"));
        return new Header(
                DType.parse(descriptor),
                fortranOrder,
                shape,
                channel.position());
    }

    private static long[] parseShape(String shapeText) throws IOException {
        String[] tokens = shapeText.trim().split(",");
        ArrayList<Long> dimensions = new ArrayList<>();
        for (String token : tokens) {
            String value = token.trim();
            if (value.isEmpty()) {
                continue;
            }
            try {
                long dimension = Long.parseLong(value);
                if (dimension <= 0) {
                    throw new IOException(
                            "NPY dimensions must be positive: " + dimension);
                }
                dimensions.add(dimension);
            } catch (NumberFormatException e) {
                throw new IOException(
                        "Invalid NPY shape component: " + value, e);
            }
        }
        if (dimensions.isEmpty() || dimensions.size() > 3) {
            throw new IOException(
                    "NPY feature reader supports array ranks 1 through 3.");
        }
        long[] shape = new long[dimensions.size()];
        for (int index = 0; index < shape.length; index++) {
            shape[index] = dimensions.get(index);
        }
        return shape;
    }

    private static String requiredGroup(
            Pattern pattern,
            String text,
            String name
    ) throws IOException {
        Matcher matcher = pattern.matcher(text);
        if (!matcher.find()) {
            throw new IOException("NPY header lacks " + name + ".");
        }
        return matcher.group(1);
    }

    private static void validateFeatureDType(DType dtype)
            throws IOException {
        if (dtype.kind() != 'f'
                || (dtype.bytes() != Float.BYTES
                && dtype.bytes() != Double.BYTES)) {
            throw new IOException(
                    "NPY feature dtype must be float32 or float64: "
                            + dtype.source());
        }
    }

    private static void validateLabelDType(DType dtype)
            throws IOException {
        if (dtype.kind() != 'f'
                && dtype.kind() != 'i'
                && dtype.kind() != 'u') {
            throw new IOException(
                    "Unsupported NPY label dtype: " + dtype.source());
        }
    }

    private static void validateLabelCount(
            List<Object> labels,
            int instanceCount
    ) throws IOException {
        if (!labels.isEmpty() && labels.size() != instanceCount) {
            throw new IOException(
                    "NPY label count " + labels.size()
                            + " does not match instance count "
                            + instanceCount + ".");
        }
    }

    private static void validatePayload(
            FileChannel channel,
            Header header,
            long elementCount
    ) throws IOException {
        final long expectedBytes;
        try {
            expectedBytes = Math.multiplyExact(
                    elementCount, header.dtype().bytes());
        } catch (ArithmeticException e) {
            throw new IOException("NPY payload size overflows long.", e);
        }
        long actualBytes = channel.size() - header.dataOffset();
        if (expectedBytes != actualBytes) {
            throw new IOException(
                    "NPY payload mismatch: expected " + expectedBytes
                            + " bytes, found " + actualBytes + ".");
        }
    }

    private static ByteBuffer createBuffer(DType dtype) {
        int capacity = Math.max(
                dtype.bytes(),
                BUFFER_SIZE - (BUFFER_SIZE % dtype.bytes()));
        return ByteBuffer.allocateDirect(capacity).order(dtype.order());
    }

    private static Object narrowInteger(long value) {
        if (value >= Integer.MIN_VALUE && value <= Integer.MAX_VALUE) {
            return (int) value;
        }
        return value;
    }

    private static boolean isNpyFile(Path path) {
        return path.getFileName().toString()
                .toLowerCase(Locale.ROOT)
                .endsWith(".npy");
    }

    private static void validateFile(Path path) throws IOException {
        if (!Files.isRegularFile(path) || !Files.isReadable(path)) {
            throw new IOException(
                    "NPY file is missing, unreadable, or not regular: "
                            + path);
        }
    }

    private static Path requiredPath(String value, String name) {
        if (value == null || value.isBlank()) {
            throw new IllegalArgumentException(name + " cannot be blank.");
        }
        return Path.of(value.trim());
    }

    private static Path optionalPath(String value) {
        if (value == null || value.isBlank()
                || value.trim().equalsIgnoreCase("None")) {
            return null;
        }
        return Path.of(value.trim());
    }

    private static int checkedPositiveInt(long value, String name)
            throws IOException {
        if (value <= 0 || value > Integer.MAX_VALUE) {
            throw new IOException(
                    "NPY " + name + " exceeds Java array limits: " + value);
        }
        return (int) value;
    }

    private static void readFully(
            FileChannel channel,
            ByteBuffer buffer
    ) throws IOException {
        while (buffer.hasRemaining()) {
            if (channel.read(buffer) < 0) {
                throw new EOFException("Unexpected end of NPY file.");
            }
        }
    }

    private record Header(
            DType dtype,
            boolean fortranOrder,
            long[] shape,
            long dataOffset
    ) {
    }

    private record DType(
            char kind,
            int bytes,
            ByteOrder order,
            String source
    ) {
        private static DType parse(String source) throws IOException {
            Matcher matcher = DTYPE_PATTERN.matcher(source.trim());
            if (!matcher.matches()) {
                throw new IOException("Unsupported NPY dtype: " + source);
            }
            char kind = matcher.group(2).charAt(0);
            int bytes = Integer.parseInt(matcher.group(3));
            if (kind == 'f'
                    && bytes != Float.BYTES
                    && bytes != Double.BYTES) {
                throw new IOException(
                        "Unsupported floating-point dtype: " + source);
            }
            String marker = matcher.group(1) == null
                    ? "="
                    : matcher.group(1);
            ByteOrder order = switch (marker) {
                case "<" -> ByteOrder.LITTLE_ENDIAN;
                case ">" -> ByteOrder.BIG_ENDIAN;
                case "=", "|" -> ByteOrder.nativeOrder();
                default -> throw new IOException(
                        "Unsupported byte-order marker: " + marker);
            };
            return new DType(kind, bytes, order, source);
        }
    }

    private record Coordinate(
            int instance,
            int dimension,
            int position
    ) {
    }

    private record Layout(
            int rank,
            int instances,
            int dimensions,
            int positionCount,
            long elements
    ) {
        private static Layout of(long[] shape) throws IOException {
            long instanceCount = shape.length == 1 ? 1 : shape[0];
            long dimensionCount = shape.length == 3 ? shape[1] : 1;
            long positionCount = shape[shape.length - 1];
            long elementCount = 1L;
            try {
                for (long dimension : shape) {
                    elementCount = Math.multiplyExact(
                            elementCount, dimension);
                }
            } catch (ArithmeticException e) {
                throw new IOException(
                        "NPY shape element count overflows long.", e);
            }
            return new Layout(
                    shape.length,
                    checkedPositiveInt(instanceCount, "instance count"),
                    checkedPositiveInt(dimensionCount, "dimension count"),
                    checkedPositiveInt(positionCount, "position count"),
                    elementCount);
        }

        private Coordinate coordinate(
                long ordinal,
                boolean fortranOrder
        ) {
            if (rank == 1) {
                return new Coordinate(0, 0, (int) ordinal);
            }
            if (rank == 2) {
                return fortranOrder
                        ? new Coordinate(
                        (int) (ordinal % instances),
                        0,
                        (int) (ordinal / instances))
                        : new Coordinate(
                        (int) (ordinal / positionCount),
                        0,
                        (int) (ordinal % positionCount));
            }
            if (fortranOrder) {
                int instance = (int) (ordinal % instances);
                long remaining = ordinal / instances;
                return new Coordinate(
                        instance,
                        (int) (remaining % dimensions),
                        (int) (remaining / dimensions));
            }
            long valuesPerInstance =
                    (long) dimensions * positionCount;
            int instance = (int) (ordinal / valuesPerInstance);
            long remaining = ordinal % valuesPerInstance;
            return new Coordinate(
                    instance,
                    (int) (remaining / positionCount),
                    (int) (remaining % positionCount));
        }
    }
}
