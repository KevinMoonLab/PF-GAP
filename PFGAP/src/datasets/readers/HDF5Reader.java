package datasets.readers;

import ch.randelshofer.fastdoubleparser.JavaDoubleParser;
import core.AppContext;
import datasets.ListObjectDataset;
import io.jhdf.HdfFile;
import io.jhdf.api.Dataset;
import org.apache.commons.lang3.time.DurationFormatUtils;

import java.io.IOException;
import java.lang.reflect.Array;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Locale;
import java.util.Set;

/**
 * Eager reader for fixed-size HDF5 datasets.
 *
 * <p>The reader expects one primary data dataset, typically {@code /X}, and
 * optionally one label dataset, typically {@code /y}.</p>
 *
 * <p>Supported data layouts:</p>
 *
 * <pre>
 * Rank 2:
 *     NT       [instance][time-or-feature]
 *
 * Rank 3:
 *     NDT      [instance][dimension][time]
 *     NTD      [instance][time][dimension]
 *
 * Rank 4:
 *     NHWC     [instance][height][width][channel]
 *     NCHW     [instance][channel][height][width]
 * </pre>
 *
 * <p>Rank-3 output is always dimension-major:</p>
 *
 * <pre>
 * data[dimension][time]
 * </pre>
 *
 * <p>Rank-4 output is always:</p>
 *
 * <pre>
 * data[pixel][channel]
 * </pre>
 *
 * <p>The optimized implementation provides direct primitive-array paths for
 * datasets decoded by jHDF as:</p>
 *
 * <ul>
 *     <li>{@code double[][]}, {@code double[][][]},
 *         {@code double[][][][]}</li>
 *     <li>{@code float[][]}, {@code float[][][]},
 *         {@code float[][][][]}</li>
 *     <li>{@code int[][]}, {@code int[][][]},
 *         {@code int[][][][]}</li>
 *     <li>{@code long[][]}, {@code long[][][]},
 *         {@code long[][][][]}</li>
 * </ul>
 *
 * <p>In the common FLOAT64 NDT case, the reader can retain the decoded
 * instance arrays directly without scalar-level reflection, boxing, or
 * copying. Other primitive numeric types are converted to PFGAP doubles using
 * typed primitive loops.</p>
 *
 * <p>Reflection is retained only as a generic fallback for uncommon numeric
 * types, strings, Booleans, byte arrays, and object datasets.</p>
 *
 * <p>Missing-value behavior:</p>
 *
 * <ul>
 *     <li>
 *         Primitive HDF5 nulls are generally unavailable. HDF5 numerical
 *         missingness is commonly represented through NaN, fill values, or a
 *         separate mask dataset.
 *     </li>
 *     <li>
 *         With {@code hasMissingValues=true}, boxed arrays are returned.
 *         Literal NaN remains {@code Double.NaN}; it is not automatically
 *         converted to null.
 *     </li>
 *     <li>
 *         Generic null or configured textual missing values become null.
 *     </li>
 * </ul>
 */
public class HDF5Reader
        implements DatasetReader {

    public enum Hdf5Layout {
        AUTO,
        NT,
        NDT,
        NTD,
        NHWC,
        NCHW
    }

    private static final Hdf5Layout DEFAULT_LAYOUT =
            Hdf5Layout.AUTO;

    private static final boolean
            DEFAULT_COLLAPSE_SINGLETON_DIMENSION =
            false;

    private final String dataFileName;
    private final String hdf5DatasetPath;
    private final String hdf5LabelDatasetPath;

    private final boolean isNumeric;
    private final boolean hasMissingValues;
    private final boolean isRegression;

    private final Hdf5Layout hdf5Layout;
    private final boolean collapseSingletonDimension;
    private final Set<String> missingIndicators;

    public HDF5Reader(
            ReaderOptions options
    ) {
        this(
                requireOptions(options).getDataPath(),
                options.getHdf5DatasetPath(),
                options.getHdf5LabelDatasetPath(),
                options.isNumeric(),
                options.hasMissingValues(),
                options.isRegression(),
                DEFAULT_LAYOUT,
                DEFAULT_COLLAPSE_SINGLETON_DIMENSION
        );
    }

    public HDF5Reader(
            String dataFileName,
            String hdf5DatasetPath,
            String hdf5LabelDatasetPath,
            boolean isNumeric,
            boolean hasMissingValues,
            boolean isRegression
    ) {
        this(
                dataFileName,
                hdf5DatasetPath,
                hdf5LabelDatasetPath,
                isNumeric,
                hasMissingValues,
                isRegression,
                DEFAULT_LAYOUT,
                DEFAULT_COLLAPSE_SINGLETON_DIMENSION
        );
    }

    /**
     * Backward-compatible constructor accepting a layout string.
     */
    public HDF5Reader(
            String dataFileName,
            String hdf5DatasetPath,
            String hdf5LabelDatasetPath,
            boolean isNumeric,
            boolean hasMissingValues,
            boolean isRegression,
            String hdf5Layout,
            boolean collapseSingletonDimension
    ) {
        this(
                dataFileName,
                hdf5DatasetPath,
                hdf5LabelDatasetPath,
                isNumeric,
                hasMissingValues,
                isRegression,
                parseLayout(hdf5Layout),
                collapseSingletonDimension
        );
    }

    public HDF5Reader(
            String dataFileName,
            String hdf5DatasetPath,
            String hdf5LabelDatasetPath,
            boolean isNumeric,
            boolean hasMissingValues,
            boolean isRegression,
            Hdf5Layout hdf5Layout,
            boolean collapseSingletonDimension
    ) {
        this.dataFileName =
                requireNonblank(
                        dataFileName,
                        "dataFileName"
                );

        this.hdf5DatasetPath =
                requireNonblank(
                        hdf5DatasetPath,
                        "hdf5DatasetPath"
                );

        this.hdf5LabelDatasetPath =
                normalizeNullableString(
                        hdf5LabelDatasetPath
                );

        this.isNumeric =
                isNumeric;

        this.hasMissingValues =
                hasMissingValues;

        this.isRegression =
                isRegression;

        this.hdf5Layout =
                hdf5Layout == null
                        ? DEFAULT_LAYOUT
                        : hdf5Layout;

        this.collapseSingletonDimension =
                collapseSingletonDimension;

        this.missingIndicators =
                snapshotMissingIndicators();
    }

    @Override
    public ListObjectDataset read()
            throws IOException {

        Path file =
                validateFile();

        long start =
                System.nanoTime();

        try (HdfFile hdfFile =
                     new HdfFile(
                             file
                     )) {

            Dataset xDataset =
                    requireDataset(
                            hdfFile,
                            hdf5DatasetPath,
                            "data"
                    );

            int[] dataShape =
                    toIntShape(
                            xDataset.getDimensions()
                    );

            validateDataShape(
                    dataShape
            );

            Hdf5Layout effectiveLayout =
                    resolveEffectiveLayout(
                            dataShape.length
                    );

            validateLayoutForRank(
                    effectiveLayout,
                    dataShape.length
            );

            int instanceCount =
                    dataShape[0];

            if (instanceCount == 0) {
                throw new IllegalArgumentException(
                        "HDF5 data dataset contains no instances: "
                                + hdf5DatasetPath
                );
            }

            Object rawData =
                    xDataset.getData();

            if (rawData == null) {
                throw new IllegalArgumentException(
                        "jHDF returned null data for dataset: "
                                + hdf5DatasetPath
                );
            }

            Object rawLabels =
                    null;

            int[] labelShape =
                    null;

            if (hdf5LabelDatasetPath != null) {
                Dataset yDataset =
                        requireDataset(
                                hdfFile,
                                hdf5LabelDatasetPath,
                                "label"
                        );

                labelShape =
                        toIntShape(
                                yDataset.getDimensions()
                        );

                validateLabelShape(
                        labelShape,
                        instanceCount
                );

                rawLabels =
                        yDataset.getData();
            }

            ListObjectDataset dataset =
                    new ListObjectDataset(
                            instanceCount
                    );

            for (int instanceIndex = 0;
                 instanceIndex < instanceCount;
                 instanceIndex++) {

                Object data =
                        buildInstance(
                                rawData,
                                dataShape,
                                instanceIndex,
                                effectiveLayout
                        );

                Object label =
                        buildLabel(
                                rawLabels,
                                labelShape,
                                instanceIndex
                        );

                dataset.add(
                        label,
                        data,
                        instanceIndex
                );

                if (instanceIndex > 0) {
                    ProgressLogger.logProgress(
                            instanceIndex
                    );
                }
            }

            int commonLength =
                    calculateCommonLength(
                            dataShape,
                            effectiveLayout
                    );

            dataset.setLength(
                    commonLength
            );

            AppContext.length =
                    commonLength;

            long end =
                    System.nanoTime();

            ProgressLogger.logDuration(
                    start,
                    end
            );

            return dataset;
        } catch (IllegalArgumentException e) {
            throw e;
        } catch (RuntimeException e) {
            throw new IOException(
                    "Failed while reading HDF5 data file: "
                            + file,
                    e
            );
        }
    }

    private Object buildInstance(
            Object rawData,
            int[] shape,
            int instanceIndex,
            Hdf5Layout layout
    ) {
        /*
         * FLOAT64 is the most direct and fastest PFGAP representation.
         */
        if (rawData instanceof double[][] array) {
            return buildRank2DoubleInstance(
                    array,
                    instanceIndex
            );
        }

        if (rawData instanceof double[][][] array) {
            return buildRank3DoubleInstance(
                    array,
                    instanceIndex,
                    layout
            );
        }

        if (rawData instanceof double[][][][] array) {
            return buildRank4DoubleInstance(
                    array,
                    instanceIndex,
                    layout
            );
        }

        /*
         * FLOAT32 conversion paths.
         */
        if (rawData instanceof float[][] array) {
            return buildRank2FloatInstance(
                    array,
                    instanceIndex
            );
        }

        if (rawData instanceof float[][][] array) {
            return buildRank3FloatInstance(
                    array,
                    instanceIndex,
                    layout
            );
        }

        if (rawData instanceof float[][][][] array) {
            return buildRank4FloatInstance(
                    array,
                    instanceIndex,
                    layout
            );
        }

        /*
         * Integer conversion paths.
         */
        if (rawData instanceof int[][] array) {
            return buildRank2IntInstance(
                    array,
                    instanceIndex
            );
        }

        if (rawData instanceof int[][][] array) {
            return buildRank3IntInstance(
                    array,
                    instanceIndex,
                    layout
            );
        }

        if (rawData instanceof int[][][][] array) {
            return buildRank4IntInstance(
                    array,
                    instanceIndex,
                    layout
            );
        }

        if (rawData instanceof long[][] array) {
            return buildRank2LongInstance(
                    array,
                    instanceIndex
            );
        }

        if (rawData instanceof long[][][] array) {
            return buildRank3LongInstance(
                    array,
                    instanceIndex,
                    layout
            );
        }

        if (rawData instanceof long[][][][] array) {
            return buildRank4LongInstance(
                    array,
                    instanceIndex,
                    layout
            );
        }

        /*
         * Generic compatibility fallback.
         */
        return buildGenericInstance(
                rawData,
                shape,
                instanceIndex,
                layout
        );
    }

    /*
     * ---------------------------------------------------------------------
     * FLOAT64 fast paths
     * ---------------------------------------------------------------------
     */

    private Object buildRank2DoubleInstance(
            double[][] rawData,
            int instanceIndex
    ) {
        double[] source =
                rawData[instanceIndex];

        if (!isNumeric) {
            return boxAsObjects(
                    source
            );
        }

        if (hasMissingValues) {
            return boxDoubles(
                    source
            );
        }

        /*
         * jHDF has already produced exactly the PFGAP representation.
         * Retaining the row avoids an unnecessary complete data copy.
         */
        return source;
    }

    private Object buildRank3DoubleInstance(
            double[][][] rawData,
            int instanceIndex,
            Hdf5Layout layout
    ) {
        double[][] source =
                rawData[instanceIndex];

        if (layout == Hdf5Layout.NTD) {
            double[][] transposed =
                    transposeDouble(
                            source
                    );

            return convertRank2DoubleResult(
                    transposed
            );
        }

        if (collapseSingletonDimension
                && source.length == 1) {

            return convertRank1DoubleResult(
                    source[0]
            );
        }

        return convertRank2DoubleResult(
                source
        );
    }

    private Object buildRank4DoubleInstance(
            double[][][][] rawData,
            int instanceIndex,
            Hdf5Layout layout
    ) {
        double[][][] source =
                rawData[instanceIndex];

        double[][] flattened =
                layout == Hdf5Layout.NCHW
                        ? flattenNchwDouble(source)
                        : flattenNhwcDouble(source);

        return convertRank2DoubleResult(
                flattened
        );
    }

    private Object convertRank1DoubleResult(
            double[] source
    ) {
        if (!isNumeric) {
            return boxAsObjects(
                    source
            );
        }

        if (hasMissingValues) {
            return boxDoubles(
                    source
            );
        }

        return source;
    }

    private Object convertRank2DoubleResult(
            double[][] source
    ) {
        if (!isNumeric) {
            Object[][] result =
                    new Object[source.length][];

            for (int index = 0;
                 index < source.length;
                 index++) {

                result[index] =
                        boxAsObjects(
                                source[index]
                        );
            }

            return result;
        }

        if (hasMissingValues) {
            Double[][] result =
                    new Double[source.length][];

            for (int index = 0;
                 index < source.length;
                 index++) {

                result[index] =
                        boxDoubles(
                                source[index]
                        );
            }

            return result;
        }

        return source;
    }

    /*
     * ---------------------------------------------------------------------
     * FLOAT32 paths
     * ---------------------------------------------------------------------
     */

    private Object buildRank2FloatInstance(
            float[][] rawData,
            int instanceIndex
    ) {
        return convertRank1DoubleResult(
                toDoubleArray(
                        rawData[instanceIndex]
                )
        );
    }

    private Object buildRank3FloatInstance(
            float[][][] rawData,
            int instanceIndex,
            Hdf5Layout layout
    ) {
        float[][] source =
                rawData[instanceIndex];

        double[][] converted =
                layout == Hdf5Layout.NTD
                        ? transposeFloatToDouble(source)
                        : toDoubleMatrix(source);

        if (collapseSingletonDimension
                && converted.length == 1) {

            return convertRank1DoubleResult(
                    converted[0]
            );
        }

        return convertRank2DoubleResult(
                converted
        );
    }

    private Object buildRank4FloatInstance(
            float[][][][] rawData,
            int instanceIndex,
            Hdf5Layout layout
    ) {
        float[][][] source =
                rawData[instanceIndex];

        double[][] flattened =
                layout == Hdf5Layout.NCHW
                        ? flattenNchwFloat(source)
                        : flattenNhwcFloat(source);

        return convertRank2DoubleResult(
                flattened
        );
    }

    /*
     * ---------------------------------------------------------------------
     * INT32 paths
     * ---------------------------------------------------------------------
     */

    private Object buildRank2IntInstance(
            int[][] rawData,
            int instanceIndex
    ) {
        return convertRank1DoubleResult(
                toDoubleArray(
                        rawData[instanceIndex]
                )
        );
    }

    private Object buildRank3IntInstance(
            int[][][] rawData,
            int instanceIndex,
            Hdf5Layout layout
    ) {
        int[][] source =
                rawData[instanceIndex];

        double[][] converted =
                layout == Hdf5Layout.NTD
                        ? transposeIntToDouble(source)
                        : toDoubleMatrix(source);

        if (collapseSingletonDimension
                && converted.length == 1) {

            return convertRank1DoubleResult(
                    converted[0]
            );
        }

        return convertRank2DoubleResult(
                converted
        );
    }

    private Object buildRank4IntInstance(
            int[][][][] rawData,
            int instanceIndex,
            Hdf5Layout layout
    ) {
        int[][][] source =
                rawData[instanceIndex];

        double[][] flattened =
                layout == Hdf5Layout.NCHW
                        ? flattenNchwInt(source)
                        : flattenNhwcInt(source);

        return convertRank2DoubleResult(
                flattened
        );
    }

    /*
     * ---------------------------------------------------------------------
     * INT64 paths
     * ---------------------------------------------------------------------
     */

    private Object buildRank2LongInstance(
            long[][] rawData,
            int instanceIndex
    ) {
        return convertRank1DoubleResult(
                toDoubleArray(
                        rawData[instanceIndex]
                )
        );
    }

    private Object buildRank3LongInstance(
            long[][][] rawData,
            int instanceIndex,
            Hdf5Layout layout
    ) {
        long[][] source =
                rawData[instanceIndex];

        double[][] converted =
                layout == Hdf5Layout.NTD
                        ? transposeLongToDouble(source)
                        : toDoubleMatrix(source);

        if (collapseSingletonDimension
                && converted.length == 1) {

            return convertRank1DoubleResult(
                    converted[0]
            );
        }

        return convertRank2DoubleResult(
                converted
        );
    }

    private Object buildRank4LongInstance(
            long[][][][] rawData,
            int instanceIndex,
            Hdf5Layout layout
    ) {
        long[][][] source =
                rawData[instanceIndex];

        double[][] flattened =
                layout == Hdf5Layout.NCHW
                        ? flattenNchwLong(source)
                        : flattenNhwcLong(source);

        return convertRank2DoubleResult(
                flattened
        );
    }

    /*
     * ---------------------------------------------------------------------
     * Generic reflective fallback
     * ---------------------------------------------------------------------
     */

    private Object buildGenericInstance(
            Object rawData,
            int[] shape,
            int instanceIndex,
            Hdf5Layout layout
    ) {
        return switch (shape.length) {
            case 2 ->
                    buildGenericRank2(
                            rawData,
                            shape,
                            instanceIndex
                    );

            case 3 ->
                    buildGenericRank3(
                            rawData,
                            shape,
                            instanceIndex,
                            layout
                    );

            case 4 ->
                    buildGenericRank4(
                            rawData,
                            shape,
                            instanceIndex,
                            layout
                    );

            default ->
                    throw new IllegalArgumentException(
                            "Unsupported HDF5 data rank: "
                                    + shape.length
                    );
        };
    }

    private Object buildGenericRank2(
            Object rawData,
            int[] shape,
            int instanceIndex
    ) {
        int length =
                shape[1];

        if (isNumeric) {
            if (hasMissingValues) {
                Double[] result =
                        new Double[length];

                for (int index = 0;
                     index < length;
                     index++) {

                    result[index] =
                            toBoxedDouble(
                                    getValue(
                                            rawData,
                                            instanceIndex,
                                            index
                                    )
                            );
                }

                return result;
            }

            double[] result =
                    new double[length];

            for (int index = 0;
                 index < length;
                 index++) {

                result[index] =
                        toPrimitiveDouble(
                                getValue(
                                        rawData,
                                        instanceIndex,
                                        index
                                )
                        );
            }

            return result;
        }

        Object[] result =
                new Object[length];

        for (int index = 0;
             index < length;
             index++) {

            result[index] =
                    parseGenericValue(
                            getValue(
                                    rawData,
                                    instanceIndex,
                                    index
                            )
                    );
        }

        return result;
    }

    private Object buildGenericRank3(
            Object rawData,
            int[] shape,
            int instanceIndex,
            Hdf5Layout layout
    ) {
        boolean ntd =
                layout == Hdf5Layout.NTD;

        int dimensionCount =
                ntd
                        ? shape[2]
                        : shape[1];

        int timeLength =
                ntd
                        ? shape[1]
                        : shape[2];

        if (collapseSingletonDimension
                && dimensionCount == 1) {

            return buildGenericCollapsedRank3(
                    rawData,
                    instanceIndex,
                    ntd,
                    timeLength
            );
        }

        if (isNumeric) {
            if (hasMissingValues) {
                Double[][] result =
                        new Double[dimensionCount][timeLength];

                for (int dimension = 0;
                     dimension < dimensionCount;
                     dimension++) {

                    for (int time = 0;
                         time < timeLength;
                         time++) {

                        result[dimension][time] =
                                toBoxedDouble(
                                        ntd
                                                ? getValue(
                                                rawData,
                                                instanceIndex,
                                                time,
                                                dimension
                                        )
                                                : getValue(
                                                rawData,
                                                instanceIndex,
                                                dimension,
                                                time
                                        )
                                );
                    }
                }

                return result;
            }

            double[][] result =
                    new double[dimensionCount][timeLength];

            for (int dimension = 0;
                 dimension < dimensionCount;
                 dimension++) {

                for (int time = 0;
                     time < timeLength;
                     time++) {

                    result[dimension][time] =
                            toPrimitiveDouble(
                                    ntd
                                            ? getValue(
                                            rawData,
                                            instanceIndex,
                                            time,
                                            dimension
                                    )
                                            : getValue(
                                            rawData,
                                            instanceIndex,
                                            dimension,
                                            time
                                    )
                            );
                }
            }

            return result;
        }

        Object[][] result =
                new Object[dimensionCount][timeLength];

        for (int dimension = 0;
             dimension < dimensionCount;
             dimension++) {

            for (int time = 0;
                 time < timeLength;
                 time++) {

                result[dimension][time] =
                        parseGenericValue(
                                ntd
                                        ? getValue(
                                        rawData,
                                        instanceIndex,
                                        time,
                                        dimension
                                )
                                        : getValue(
                                        rawData,
                                        instanceIndex,
                                        dimension,
                                        time
                                )
                        );
            }
        }

        return result;
    }

    private Object buildGenericCollapsedRank3(
            Object rawData,
            int instanceIndex,
            boolean ntd,
            int timeLength
    ) {
        if (isNumeric) {
            if (hasMissingValues) {
                Double[] result =
                        new Double[timeLength];

                for (int time = 0;
                     time < timeLength;
                     time++) {

                    result[time] =
                            toBoxedDouble(
                                    ntd
                                            ? getValue(
                                            rawData,
                                            instanceIndex,
                                            time,
                                            0
                                    )
                                            : getValue(
                                            rawData,
                                            instanceIndex,
                                            0,
                                            time
                                    )
                            );
                }

                return result;
            }

            double[] result =
                    new double[timeLength];

            for (int time = 0;
                 time < timeLength;
                 time++) {

                result[time] =
                        toPrimitiveDouble(
                                ntd
                                        ? getValue(
                                        rawData,
                                        instanceIndex,
                                        time,
                                        0
                                )
                                        : getValue(
                                        rawData,
                                        instanceIndex,
                                        0,
                                        time
                                )
                        );
            }

            return result;
        }

        Object[] result =
                new Object[timeLength];

        for (int time = 0;
             time < timeLength;
             time++) {

            result[time] =
                    parseGenericValue(
                            ntd
                                    ? getValue(
                                    rawData,
                                    instanceIndex,
                                    time,
                                    0
                            )
                                    : getValue(
                                    rawData,
                                    instanceIndex,
                                    0,
                                    time
                            )
                    );
        }

        return result;
    }

    private Object buildGenericRank4(
            Object rawData,
            int[] shape,
            int instanceIndex,
            Hdf5Layout layout
    ) {
        boolean nchw =
                layout == Hdf5Layout.NCHW;

        int channels =
                nchw
                        ? shape[1]
                        : shape[3];

        int height =
                nchw
                        ? shape[2]
                        : shape[1];

        int width =
                nchw
                        ? shape[3]
                        : shape[2];

        int pixels =
                Math.multiplyExact(
                        height,
                        width
                );

        if (isNumeric) {
            if (hasMissingValues) {
                Double[][] result =
                        new Double[pixels][channels];

                for (int heightIndex = 0;
                     heightIndex < height;
                     heightIndex++) {

                    for (int widthIndex = 0;
                         widthIndex < width;
                         widthIndex++) {

                        int pixel =
                                heightIndex * width
                                        + widthIndex;

                        for (int channel = 0;
                             channel < channels;
                             channel++) {

                            result[pixel][channel] =
                                    toBoxedDouble(
                                            nchw
                                                    ? getValue(
                                                    rawData,
                                                    instanceIndex,
                                                    channel,
                                                    heightIndex,
                                                    widthIndex
                                            )
                                                    : getValue(
                                                    rawData,
                                                    instanceIndex,
                                                    heightIndex,
                                                    widthIndex,
                                                    channel
                                            )
                                    );
                        }
                    }
                }

                return result;
            }

            double[][] result =
                    new double[pixels][channels];

            for (int heightIndex = 0;
                 heightIndex < height;
                 heightIndex++) {

                for (int widthIndex = 0;
                     widthIndex < width;
                     widthIndex++) {

                    int pixel =
                            heightIndex * width
                                    + widthIndex;

                    for (int channel = 0;
                         channel < channels;
                         channel++) {

                        result[pixel][channel] =
                                toPrimitiveDouble(
                                        nchw
                                                ? getValue(
                                                rawData,
                                                instanceIndex,
                                                channel,
                                                heightIndex,
                                                widthIndex
                                        )
                                                : getValue(
                                                rawData,
                                                instanceIndex,
                                                heightIndex,
                                                widthIndex,
                                                channel
                                        )
                                );
                    }
                }
            }

            return result;
        }

        Object[][] result =
                new Object[pixels][channels];

        for (int heightIndex = 0;
             heightIndex < height;
             heightIndex++) {

            for (int widthIndex = 0;
                 widthIndex < width;
                 widthIndex++) {

                int pixel =
                        heightIndex * width
                                + widthIndex;

                for (int channel = 0;
                     channel < channels;
                     channel++) {

                    result[pixel][channel] =
                            parseGenericValue(
                                    nchw
                                            ? getValue(
                                            rawData,
                                            instanceIndex,
                                            channel,
                                            heightIndex,
                                            widthIndex
                                    )
                                            : getValue(
                                            rawData,
                                            instanceIndex,
                                            heightIndex,
                                            widthIndex,
                                            channel
                                    )
                            );
                }
            }
        }

        return result;
    }

    /*
     * ---------------------------------------------------------------------
     * Labels
     * ---------------------------------------------------------------------
     */

    private Object buildLabel(
            Object rawLabels,
            int[] labelShape,
            int instanceIndex
    ) {
        if (rawLabels == null || labelShape == null) {
            return null;
        }

        if (labelShape.length == 1) {
            return parseLabelValue(
                    getRank1Value(
                            rawLabels,
                            instanceIndex
                    )
            );
        }

        int labelCount =
                labelShape[1];

        List<Object> labels =
                new ArrayList<>(
                        labelCount
                );

        /*
         * Common primitive rank-2 label paths.
         */
        if (rawLabels instanceof int[][] values) {
            for (int labelIndex = 0;
                 labelIndex < labelCount;
                 labelIndex++) {

                labels.add(
                        values[instanceIndex][labelIndex]
                );
            }

            return Collections.unmodifiableList(
                    labels
            );
        }

        if (rawLabels instanceof long[][] values) {
            for (int labelIndex = 0;
                 labelIndex < labelCount;
                 labelIndex++) {

                labels.add(
                        normalizeIntegralLabel(
                                values[instanceIndex][labelIndex]
                        )
                );
            }

            return Collections.unmodifiableList(
                    labels
            );
        }

        if (rawLabels instanceof double[][] values) {
            for (int labelIndex = 0;
                 labelIndex < labelCount;
                 labelIndex++) {

                labels.add(
                        parseLabelValue(
                                values[instanceIndex][labelIndex]
                        )
                );
            }

            return Collections.unmodifiableList(
                    labels
            );
        }

        for (int labelIndex = 0;
             labelIndex < labelCount;
             labelIndex++) {

            labels.add(
                    parseLabelValue(
                            getValue(
                                    rawLabels,
                                    instanceIndex,
                                    labelIndex
                            )
                    )
            );
        }

        return Collections.unmodifiableList(
                labels
        );
    }

    private Object getRank1Value(
            Object rawLabels,
            int instanceIndex
    ) {
        if (rawLabels instanceof int[] values) {
            return values[instanceIndex];
        }

        if (rawLabels instanceof long[] values) {
            return values[instanceIndex];
        }

        if (rawLabels instanceof double[] values) {
            return values[instanceIndex];
        }

        if (rawLabels instanceof float[] values) {
            return values[instanceIndex];
        }

        if (rawLabels instanceof short[] values) {
            return values[instanceIndex];
        }

        if (rawLabels instanceof byte[] values) {
            return values[instanceIndex];
        }

        if (rawLabels instanceof boolean[] values) {
            return values[instanceIndex];
        }

        if (rawLabels instanceof Object[] values) {
            return values[instanceIndex];
        }

        return Array.get(
                rawLabels,
                instanceIndex
        );
    }

    private Object parseLabelValue(
            Object value
    ) {
        Object normalized =
                normalizeValue(
                        value
                );

        if (normalized == null) {
            return null;
        }

        if (isRegression) {
            return toPrimitiveDouble(
                    normalized
            );
        }

        if (normalized instanceof Integer) {
            return normalized;
        }

        if (normalized instanceof Byte
                || normalized instanceof Short) {

            return ((Number) normalized).intValue();
        }

        if (normalized instanceof Long longValue) {
            return normalizeIntegralLabel(
                    longValue
            );
        }

        if (normalized instanceof Number number) {
            double valueAsDouble =
                    number.doubleValue();

            return normalizeNumericLabel(
                    valueAsDouble
            );
        }

        String token =
                normalized.toString()
                        .trim();

        if (isMissingToken(token)) {
            return null;
        }

        try {
            return Integer.parseInt(
                    token
            );
        } catch (NumberFormatException ignored) {
            try {
                return normalizeNumericLabel(
                        JavaDoubleParser.parseDouble(
                                token
                        )
                );
            } catch (NumberFormatException ignoredAgain) {
                return token;
            }
        }
    }

    private static Object normalizeIntegralLabel(
            long value
    ) {
        if (value >= Integer.MIN_VALUE
                && value <= Integer.MAX_VALUE) {

            return (int) value;
        }

        return value;
    }

    private static Object normalizeNumericLabel(
            double value
    ) {
        if (value == Math.rint(value)
                && value >= Integer.MIN_VALUE
                && value <= Integer.MAX_VALUE) {

            return (int) value;
        }

        return value;
    }

    /*
     * ---------------------------------------------------------------------
     * Typed conversion helpers
     * ---------------------------------------------------------------------
     */

    private static double[] toDoubleArray(
            float[] source
    ) {
        double[] result =
                new double[source.length];

        for (int index = 0;
             index < source.length;
             index++) {

            result[index] =
                    source[index];
        }

        return result;
    }

    private static double[] toDoubleArray(
            int[] source
    ) {
        double[] result =
                new double[source.length];

        for (int index = 0;
             index < source.length;
             index++) {

            result[index] =
                    source[index];
        }

        return result;
    }

    private static double[] toDoubleArray(
            long[] source
    ) {
        double[] result =
                new double[source.length];

        for (int index = 0;
             index < source.length;
             index++) {

            result[index] =
                    source[index];
        }

        return result;
    }

    private static double[][] toDoubleMatrix(
            float[][] source
    ) {
        double[][] result =
                new double[source.length][];

        for (int index = 0;
             index < source.length;
             index++) {

            result[index] =
                    toDoubleArray(
                            source[index]
                    );
        }

        return result;
    }

    private static double[][] toDoubleMatrix(
            int[][] source
    ) {
        double[][] result =
                new double[source.length][];

        for (int index = 0;
             index < source.length;
             index++) {

            result[index] =
                    toDoubleArray(
                            source[index]
                    );
        }

        return result;
    }

    private static double[][] toDoubleMatrix(
            long[][] source
    ) {
        double[][] result =
                new double[source.length][];

        for (int index = 0;
             index < source.length;
             index++) {

            result[index] =
                    toDoubleArray(
                            source[index]
                    );
        }

        return result;
    }

    private static double[][] transposeDouble(
            double[][] source
    ) {
        if (source.length == 0) {
            return new double[0][0];
        }

        int timeLength =
                source.length;

        int dimensionCount =
                source[0].length;

        double[][] result =
                new double[dimensionCount][timeLength];

        for (int time = 0;
             time < timeLength;
             time++) {

            validateRowLength(
                    source[time].length,
                    dimensionCount,
                    "double NTD"
            );

            for (int dimension = 0;
                 dimension < dimensionCount;
                 dimension++) {

                result[dimension][time] =
                        source[time][dimension];
            }
        }

        return result;
    }

    private static double[][] transposeFloatToDouble(
            float[][] source
    ) {
        if (source.length == 0) {
            return new double[0][0];
        }

        int timeLength =
                source.length;

        int dimensionCount =
                source[0].length;

        double[][] result =
                new double[dimensionCount][timeLength];

        for (int time = 0;
             time < timeLength;
             time++) {

            validateRowLength(
                    source[time].length,
                    dimensionCount,
                    "float NTD"
            );

            for (int dimension = 0;
                 dimension < dimensionCount;
                 dimension++) {

                result[dimension][time] =
                        source[time][dimension];
            }
        }

        return result;
    }

    private static double[][] transposeIntToDouble(
            int[][] source
    ) {
        if (source.length == 0) {
            return new double[0][0];
        }

        int timeLength =
                source.length;

        int dimensionCount =
                source[0].length;

        double[][] result =
                new double[dimensionCount][timeLength];

        for (int time = 0;
             time < timeLength;
             time++) {

            validateRowLength(
                    source[time].length,
                    dimensionCount,
                    "int NTD"
            );

            for (int dimension = 0;
                 dimension < dimensionCount;
                 dimension++) {

                result[dimension][time] =
                        source[time][dimension];
            }
        }

        return result;
    }

    private static double[][] transposeLongToDouble(
            long[][] source
    ) {
        if (source.length == 0) {
            return new double[0][0];
        }

        int timeLength =
                source.length;

        int dimensionCount =
                source[0].length;

        double[][] result =
                new double[dimensionCount][timeLength];

        for (int time = 0;
             time < timeLength;
             time++) {

            validateRowLength(
                    source[time].length,
                    dimensionCount,
                    "long NTD"
            );

            for (int dimension = 0;
                 dimension < dimensionCount;
                 dimension++) {

                result[dimension][time] =
                        source[time][dimension];
            }
        }

        return result;
    }

    private static double[][] flattenNhwcDouble(
            double[][][] source
    ) {
        int height =
                source.length;

        int width =
                height == 0
                        ? 0
                        : source[0].length;

        int channels =
                height == 0 || width == 0
                        ? 0
                        : source[0][0].length;

        double[][] result =
                new double[Math.multiplyExact(height, width)][channels];

        for (int h = 0;
             h < height;
             h++) {

            validateRowLength(
                    source[h].length,
                    width,
                    "double NHWC height"
            );

            for (int w = 0;
                 w < width;
                 w++) {

                validateRowLength(
                        source[h][w].length,
                        channels,
                        "double NHWC channels"
                );

                System.arraycopy(
                        source[h][w],
                        0,
                        result[h * width + w],
                        0,
                        channels
                );
            }
        }

        return result;
    }

    private static double[][] flattenNchwDouble(
            double[][][] source
    ) {
        int channels =
                source.length;

        int height =
                channels == 0
                        ? 0
                        : source[0].length;

        int width =
                channels == 0 || height == 0
                        ? 0
                        : source[0][0].length;

        double[][] result =
                new double[Math.multiplyExact(height, width)][channels];

        for (int channel = 0;
             channel < channels;
             channel++) {

            for (int h = 0;
                 h < height;
                 h++) {

                for (int w = 0;
                     w < width;
                     w++) {

                    result[h * width + w][channel] =
                            source[channel][h][w];
                }
            }
        }

        return result;
    }

    private static double[][] flattenNhwcFloat(
            float[][][] source
    ) {
        return flattenGenericNhwc(
                source
        );
    }

    private static double[][] flattenNchwFloat(
            float[][][] source
    ) {
        return flattenGenericNchw(
                source
        );
    }

    private static double[][] flattenNhwcInt(
            int[][][] source
    ) {
        return flattenGenericNhwc(
                source
        );
    }

    private static double[][] flattenNchwInt(
            int[][][] source
    ) {
        return flattenGenericNchw(
                source
        );
    }

    private static double[][] flattenNhwcLong(
            long[][][] source
    ) {
        return flattenGenericNhwc(
                source
        );
    }

    private static double[][] flattenNchwLong(
            long[][][] source
    ) {
        return flattenGenericNchw(
                source
        );
    }

    private static double[][] flattenGenericNhwc(
            Object source
    ) {
        int height =
                Array.getLength(
                        source
                );

        Object firstHeight =
                height == 0
                        ? null
                        : Array.get(source, 0);

        int width =
                firstHeight == null
                        ? 0
                        : Array.getLength(firstHeight);

        Object firstPixel =
                width == 0
                        ? null
                        : Array.get(firstHeight, 0);

        int channels =
                firstPixel == null
                        ? 0
                        : Array.getLength(firstPixel);

        double[][] result =
                new double[Math.multiplyExact(height, width)][channels];

        for (int h = 0;
             h < height;
             h++) {

            Object row =
                    Array.get(
                            source,
                            h
                    );

            for (int w = 0;
                 w < width;
                 w++) {

                Object pixel =
                        Array.get(
                                row,
                                w
                        );

                for (int channel = 0;
                     channel < channels;
                     channel++) {

                    result[h * width + w][channel] =
                            ((Number) Array.get(
                                    pixel,
                                    channel
                            )).doubleValue();
                }
            }
        }

        return result;
    }

    private static double[][] flattenGenericNchw(
            Object source
    ) {
        int channels =
                Array.getLength(
                        source
                );

        Object firstChannel =
                channels == 0
                        ? null
                        : Array.get(source, 0);

        int height =
                firstChannel == null
                        ? 0
                        : Array.getLength(firstChannel);

        Object firstRow =
                height == 0
                        ? null
                        : Array.get(firstChannel, 0);

        int width =
                firstRow == null
                        ? 0
                        : Array.getLength(firstRow);

        double[][] result =
                new double[Math.multiplyExact(height, width)][channels];

        for (int channel = 0;
             channel < channels;
             channel++) {

            Object channelData =
                    Array.get(
                            source,
                            channel
                    );

            for (int h = 0;
                 h < height;
                 h++) {

                Object row =
                        Array.get(
                                channelData,
                                h
                        );

                for (int w = 0;
                     w < width;
                     w++) {

                    result[h * width + w][channel] =
                            ((Number) Array.get(
                                    row,
                                    w
                            )).doubleValue();
                }
            }
        }

        return result;
    }

    private static Double[] boxDoubles(
            double[] source
    ) {
        Double[] result =
                new Double[source.length];

        for (int index = 0;
             index < source.length;
             index++) {

            result[index] =
                    source[index];
        }

        return result;
    }

    private static Object[] boxAsObjects(
            double[] source
    ) {
        Object[] result =
                new Object[source.length];

        for (int index = 0;
             index < source.length;
             index++) {

            result[index] =
                    source[index];
        }

        return result;
    }

    private static void validateRowLength(
            int actual,
            int expected,
            String role
    ) {
        if (actual != expected) {
            throw new IllegalArgumentException(
                    "Inconsistent "
                            + role
                            + " array length. Expected "
                            + expected
                            + " but found "
                            + actual
                            + "."
            );
        }
    }

    /*
     * ---------------------------------------------------------------------
     * Generic conversions
     * ---------------------------------------------------------------------
     */

    private Object parseGenericValue(
            Object value
    ) {
        Object normalized =
                normalizeValue(
                        value
                );

        if (normalized == null) {
            return null;
        }

        if (!(normalized instanceof CharSequence)) {
            return normalized;
        }

        String token =
                normalized.toString()
                        .trim();

        if (isMissingToken(token)) {
            return null;
        }

        try {
            return JavaDoubleParser.parseDouble(
                    token
            );
        } catch (NumberFormatException ignored) {
            if (token.equalsIgnoreCase("true")
                    || token.equalsIgnoreCase("false")) {

                return Boolean.parseBoolean(
                        token
                );
            }

            return token;
        }
    }

    private Double toBoxedDouble(
            Object value
    ) {
        Object normalized =
                normalizeValue(
                        value
                );

        if (normalized == null) {
            return null;
        }

        if (normalized instanceof Number number) {
            return number.doubleValue();
        }

        if (normalized instanceof Boolean booleanValue) {
            return booleanValue
                    ? 1.0
                    : 0.0;
        }

        String token =
                normalized.toString()
                        .trim();

        if (isMissingToken(token)) {
            return null;
        }

        return JavaDoubleParser.parseDouble(
                token
        );
    }

    private double toPrimitiveDouble(
            Object value
    ) {
        Object normalized =
                normalizeValue(
                        value
                );

        if (normalized == null) {
            throw new IllegalArgumentException(
                    "Encountered null numeric value, but "
                            + "hasMissingValues=false."
            );
        }

        if (normalized instanceof Number number) {
            return number.doubleValue();
        }

        if (normalized instanceof Boolean booleanValue) {
            return booleanValue
                    ? 1.0
                    : 0.0;
        }

        String token =
                normalized.toString()
                        .trim();

        if (isMissingToken(token)) {
            throw new IllegalArgumentException(
                    "Encountered missing numeric value, but "
                            + "hasMissingValues=false."
            );
        }

        return JavaDoubleParser.parseDouble(
                token
        );
    }

    private static Object normalizeValue(
            Object value
    ) {
        if (value == null) {
            return null;
        }

        if (value instanceof byte[] bytes) {
            return new String(
                    bytes,
                    StandardCharsets.UTF_8
            ).trim();
        }

        if (value instanceof Character character) {
            return character.toString();
        }

        return value;
    }

    private boolean isMissingToken(
            String token
    ) {
        if (token == null) {
            return true;
        }

        String trimmed =
                token.trim();

        if (trimmed.isEmpty()) {
            return true;
        }

        return missingIndicators.contains(
                trimmed.toUpperCase(
                        Locale.ROOT
                )
        );
    }

    private static Object getValue(
            Object array,
            int... indices
    ) {
        Object current =
                array;

        for (int index : indices) {
            current =
                    Array.get(
                            current,
                            index
                    );
        }

        return current;
    }

    /*
     * ---------------------------------------------------------------------
     * Metadata and validation
     * ---------------------------------------------------------------------
     */

    private static ReaderOptions requireOptions(
            ReaderOptions options
    ) {
        if (options == null) {
            throw new IllegalArgumentException(
                    "HDF5Reader requires non-null ReaderOptions."
            );
        }

        return options;
    }

    private static String requireNonblank(
            String value,
            String argumentName
    ) {
        if (value == null || value.isBlank()) {
            throw new IllegalArgumentException(
                    "HDF5Reader requires "
                            + argumentName
                            + "."
            );
        }

        return value.trim();
    }

    private static String normalizeNullableString(
            String value
    ) {
        if (value == null) {
            return null;
        }

        String trimmed =
                value.trim();

        if (trimmed.isEmpty()
                || trimmed.equalsIgnoreCase("None")) {

            return null;
        }

        return trimmed;
    }

    private static Hdf5Layout parseLayout(
            String value
    ) {
        String normalized =
                normalizeNullableString(
                        value
                );

        if (normalized == null) {
            return DEFAULT_LAYOUT;
        }

        try {
            return Hdf5Layout.valueOf(
                    normalized.toUpperCase(
                            Locale.ROOT
                    )
            );
        } catch (IllegalArgumentException e) {
            throw new IllegalArgumentException(
                    "Unsupported hdf5Layout: "
                            + value
                            + ". Use AUTO, NT, NDT, NTD, NHWC, or NCHW.",
                    e
            );
        }
    }

    private Path validateFile()
            throws IOException {

        Path file =
                Path.of(
                        dataFileName
                );

        if (!Files.exists(file)) {
            throw new IOException(
                    "HDF5 data file does not exist: "
                            + file
            );
        }

        if (!Files.isRegularFile(file)) {
            throw new IOException(
                    "HDF5 data path is not a regular file: "
                            + file
            );
        }

        if (!Files.isReadable(file)) {
            throw new IOException(
                    "HDF5 data file is not readable: "
                            + file
            );
        }

        return file;
    }

    private static Dataset requireDataset(
            HdfFile hdfFile,
            String datasetPath,
            String role
    ) {
        Dataset dataset =
                hdfFile.getDatasetByPath(
                        datasetPath
                );

        if (dataset == null) {
            throw new IllegalArgumentException(
                    "HDF5 "
                            + role
                            + " dataset not found: "
                            + datasetPath
            );
        }

        return dataset;
    }

    private static void validateDataShape(
            int[] shape
    ) {
        if (shape.length < 2 || shape.length > 4) {
            throw new IllegalArgumentException(
                    "HDF5Reader supports data ranks 2, 3, and 4. "
                            + "Found rank "
                            + shape.length
                            + "."
            );
        }

        for (int dimension = 0;
             dimension < shape.length;
             dimension++) {

            if (shape[dimension] < 0) {
                throw new IllegalArgumentException(
                        "Negative HDF5 dimension at index "
                                + dimension
                                + ": "
                                + shape[dimension]
                );
            }
        }
    }

    private Hdf5Layout resolveEffectiveLayout(
            int rank
    ) {
        if (hdf5Layout != Hdf5Layout.AUTO) {
            return hdf5Layout;
        }

        return switch (rank) {
            case 2 -> Hdf5Layout.NT;
            case 3 -> Hdf5Layout.NDT;
            case 4 -> Hdf5Layout.NHWC;
            default -> throw new IllegalArgumentException(
                    "Cannot resolve AUTO layout for HDF5 rank "
                            + rank
                            + "."
            );
        };
    }

    private static void validateLayoutForRank(
            Hdf5Layout layout,
            int rank
    ) {
        boolean valid =
                switch (rank) {
                    case 2 ->
                            layout == Hdf5Layout.NT;

                    case 3 ->
                            layout == Hdf5Layout.NDT
                                    || layout == Hdf5Layout.NTD;

                    case 4 ->
                            layout == Hdf5Layout.NHWC
                                    || layout == Hdf5Layout.NCHW;

                    default ->
                            false;
                };

        if (!valid) {
            throw new IllegalArgumentException(
                    "HDF5 layout "
                            + layout
                            + " is incompatible with dataset rank "
                            + rank
                            + "."
            );
        }
    }

    private static void validateLabelShape(
            int[] labelShape,
            int instanceCount
    ) {
        if (labelShape.length != 1
                && labelShape.length != 2) {

            throw new IllegalArgumentException(
                    "HDF5Reader supports label ranks 1 and 2. "
                            + "Found rank "
                            + labelShape.length
                            + "."
            );
        }

        if (labelShape[0] != instanceCount) {
            throw new IllegalArgumentException(
                    "Label dataset first dimension does not match data. "
                            + "Data instances="
                            + instanceCount
                            + ", label instances="
                            + labelShape[0]
                            + "."
            );
        }
    }

    private static int calculateCommonLength(
            int[] shape,
            Hdf5Layout layout
    ) {
        return switch (shape.length) {
            case 2 ->
                    shape[1];

            case 3 ->
                    layout == Hdf5Layout.NTD
                            ? shape[1]
                            : shape[2];

            case 4 -> {
                int height =
                        layout == Hdf5Layout.NCHW
                                ? shape[2]
                                : shape[1];

                int width =
                        layout == Hdf5Layout.NCHW
                                ? shape[3]
                                : shape[2];

                yield Math.multiplyExact(
                        height,
                        width
                );
            }

            default ->
                    0;
        };
    }

    private static int[] toIntShape(
            long[] dimensions
    ) {
        int[] result =
                new int[dimensions.length];

        for (int index = 0;
             index < dimensions.length;
             index++) {

            if (dimensions[index] > Integer.MAX_VALUE) {
                throw new IllegalArgumentException(
                        "HDF5 dimension is too large for Java array "
                                + "indexing: "
                                + dimensions[index]
                );
            }

            result[index] =
                    Math.toIntExact(
                            dimensions[index]
                    );
        }

        return result;
    }

    private static int[] toIntShape(
            int[] dimensions
    ) {
        return dimensions.clone();
    }

    private static Set<String> snapshotMissingIndicators() {
        if (AppContext.MissingStrings == null
                || AppContext.MissingStrings.isEmpty()) {

            return Set.of();
        }

        Set<String> result =
                new HashSet<>();

        for (String indicator : AppContext.MissingStrings) {
            if (indicator == null) {
                continue;
            }

            String normalized =
                    indicator.trim();

            if (!normalized.isEmpty()) {
                result.add(
                        normalized.toUpperCase(
                                Locale.ROOT
                        )
                );
            }
        }

        if (result.isEmpty()) {
            return Set.of();
        }

        return Collections.unmodifiableSet(
                result
        );
    }

    public static class ProgressLogger {

        public static void logProgress(
                int index
        ) {
            if (index % 1000 != 0) {
                return;
            }

            if (index % 100000 == 0) {
                System.out.print("\n");

                if (index % 1000000 == 0) {
                    long usedMemory =
                            AppContext.runtime.totalMemory()
                                    - AppContext.runtime.freeMemory();

                    System.out.print(
                            index
                                    + ":"
                                    + usedMemory / 1024 / 1024
                                    + "mb\n"
                    );
                }

                return;
            }

            System.out.print(".");
        }

        public static void logDuration(
                long start,
                long end
        ) {
            long elapsed =
                    end - start;

            String duration =
                    DurationFormatUtils.formatDuration(
                            (long) (elapsed / 1e6),
                            "H:m:s.SSS"
                    );

            System.out.println(
                    "finished in "
                            + duration
            );
        }
    }
}