package datasets.readers.lazy;

import java.io.Serial;
import java.io.Serializable;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Objects;

public final class LazySeriesRef implements Serializable {

    @Serial
    private static final long serialVersionUID = 1L;

    private final String readerKey;
    private final int index;
    private final String filePath;

    public LazySeriesRef(
            String readerKey,
            int index,
            Path file
    ) {
        if (readerKey == null || readerKey.isBlank()) {
            throw new IllegalArgumentException(
                    "LazySeriesRef requires a non-empty readerKey."
            );
        }

        if (file == null) {
            throw new IllegalArgumentException(
                    "LazySeriesRef requires a non-null file."
            );
        }

        this.readerKey = readerKey;
        this.index = index;

        /*
         * Store the absolute normalized path as serializable text.
         * This avoids serializing UnixPath or WindowsPath.
         */
        this.filePath =
                file.toAbsolutePath()
                        .normalize()
                        .toString();
    }

    public String getReaderKey() {
        return readerKey;
    }

    public int getIndex() {
        return index;
    }

    public Path getFile() {
        return Paths.get(filePath);
    }

    public String getFilePath() {
        return filePath;
    }

    @Override
    public boolean equals(Object other) {
        if (this == other) {
            return true;
        }

        if (!(other instanceof LazySeriesRef that)) {
            return false;
        }

        return index == that.index
                && Objects.equals(readerKey, that.readerKey)
                && Objects.equals(filePath, that.filePath);
    }

    @Override
    public int hashCode() {
        return Objects.hash(
                readerKey,
                index,
                filePath
        );
    }

    @Override
    public String toString() {
        return "LazySeriesRef{"
                + "readerKey='" + readerKey + '\''
                + ", index=" + index
                + ", filePath='" + filePath + '\''
                + '}';
    }
}