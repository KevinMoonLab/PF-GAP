package datasets.readers.interop;

import datasets.readers.api.CustomDatasetReader;
import datasets.readers.api.CustomSeriesReader;

import java.io.IOException;
import java.lang.reflect.Constructor;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Modifier;
import java.net.MalformedURLException;
import java.net.URL;
import java.net.URLClassLoader;
import java.nio.file.Files;
import java.nio.file.InvalidPathException;
import java.nio.file.Path;
import java.util.Locale;
import java.util.Objects;

/**
 * Loads user-defined custom reader implementations from external JAR files.
 *
 * <p>Canonical descriptor format:</p>
 *
 * <pre>
 * javareader:/path/to/readers.jar:my.package.MyReader
 * </pre>
 *
 * <p>The descriptor is split at its final colon so Windows drive-letter paths
 * remain valid. Plugins must be concrete classes with public no-argument
 * constructors and must implement the reader interface required by the
 * selected PFGAP reader type.</p>
 */
public final class JavaReaderLoader {

    public static final String DESCRIPTOR_PREFIX = "javareader:";
    private static final String JAR_EXTENSION = ".jar";

    private JavaReaderLoader() {
        // Utility class.
    }

    /** Loads a per-observation custom series reader. */
    public static LoadedCustomReader load(
            String descriptor
    ) throws IOException, ReflectiveOperationException {
        LoadedPlugin<CustomSeriesReader> loaded = loadTyped(
                descriptor,
                CustomSeriesReader.class,
                "series"
        );
        return new LoadedCustomReader(
                loaded.reader(),
                loaded.classLoader(),
                loaded.descriptor(),
                loaded.implementationClassName()
        );
    }

    /** Loads an eager whole-dataset custom reader. */
    public static LoadedCustomDatasetReader loadDatasetReader(
            String descriptor
    ) throws IOException, ReflectiveOperationException {
        LoadedPlugin<CustomDatasetReader> loaded = loadTyped(
                descriptor,
                CustomDatasetReader.class,
                "dataset"
        );
        return new LoadedCustomDatasetReader(
                loaded.reader(),
                loaded.classLoader(),
                loaded.descriptor(),
                loaded.implementationClassName()
        );
    }

    /** Normalizes descriptor syntax without opening the JAR. */
    public static String normalizeDescriptor(String descriptor) {
        return parseDescriptor(descriptor).normalizedDescriptor();
    }

    private static <T> LoadedPlugin<T> loadTyped(
            String descriptor,
            Class<T> requiredType,
            String readerRole
    ) throws IOException, ReflectiveOperationException {
        Objects.requireNonNull(requiredType, "Required plugin type cannot be null.");
        ParsedReaderDescriptor parsed = parseDescriptor(descriptor);
        Path jarPath = validateJarPath(parsed.jarPath());
        URL jarUrl = toUrl(jarPath);

        URLClassLoader classLoader = new URLClassLoader(
                "pfgap-custom-" + readerRole + "-reader:" + parsed.className(),
                new URL[]{jarUrl},
                requiredType.getClassLoader()
        );

        try {
            Class<?> implementationClass = Class.forName(
                    parsed.className(),
                    true,
                    classLoader
            );
            validateImplementationClass(
                    implementationClass,
                    requiredType,
                    parsed
            );
            Constructor<?> constructor = requireNoArgumentConstructor(
                    implementationClass,
                    parsed,
                    jarPath
            );
            T reader = instantiate(
                    constructor,
                    requiredType,
                    parsed,
                    jarPath
            );
            return new LoadedPlugin<>(
                    reader,
                    classLoader,
                    parsed.normalizedDescriptor(),
                    parsed.className()
            );
        } catch (ReflectiveOperationException
                 | RuntimeException | Error failure) {
            try {
                classLoader.close();
            } catch (IOException closeFailure) {
                failure.addSuppressed(closeFailure);
            }
            throw failure;
        }
    }

    private static ParsedReaderDescriptor parseDescriptor(String descriptor) {
        if (descriptor == null || descriptor.isBlank()) {
            throw new IllegalArgumentException(
                    "Custom reader descriptor cannot be null or blank. "
                            + "Expected format: " + DESCRIPTOR_PREFIX
                            + "/path/to/readers.jar:my.package.MyReader"
            );
        }

        String trimmed = descriptor.trim();
        if (!trimmed.regionMatches(
                true, 0, DESCRIPTOR_PREFIX, 0, DESCRIPTOR_PREFIX.length()
        )) {
            throw new IllegalArgumentException(
                    "Invalid custom reader descriptor prefix. Expected '"
                            + DESCRIPTOR_PREFIX + "': " + descriptor
            );
        }

        String remainder = trimmed.substring(DESCRIPTOR_PREFIX.length());
        int separator = remainder.lastIndexOf(':');
        if (separator < 0) {
            throw new IllegalArgumentException(
                    "Custom reader descriptor must contain a JAR path and "
                            + "implementation class separated by the final ':' : "
                            + descriptor
            );
        }

        String pathText = remainder.substring(0, separator).trim();
        String className = remainder.substring(separator + 1).trim();
        if (pathText.isEmpty()) {
            throw new IllegalArgumentException(
                    "Custom reader descriptor contains an empty JAR path: "
                            + descriptor
            );
        }
        if (className.isEmpty()) {
            throw new IllegalArgumentException(
                    "Custom reader descriptor contains an empty implementation "
                            + "class name: " + descriptor
            );
        }
        validateClassName(className, descriptor);

        Path jarPath;
        try {
            jarPath = Path.of(pathText).toAbsolutePath().normalize();
        } catch (InvalidPathException failure) {
            throw new IllegalArgumentException(
                    "Custom reader descriptor contains an invalid JAR path: "
                            + pathText,
                    failure
            );
        }

        return new ParsedReaderDescriptor(
                jarPath,
                className,
                DESCRIPTOR_PREFIX + jarPath + ":" + className
        );
    }

    private static Path validateJarPath(Path jarPath) throws IOException {
        Objects.requireNonNull(jarPath, "Custom reader JAR path cannot be null.");
        String fileName = jarPath.getFileName() == null
                ? ""
                : jarPath.getFileName().toString();
        if (!fileName.toLowerCase(Locale.ROOT).endsWith(JAR_EXTENSION)) {
            throw new IllegalArgumentException(
                    "Custom reader plugins must currently be supplied as JAR "
                            + "files. Received: " + jarPath
            );
        }
        if (!Files.exists(jarPath)) {
            throw new IOException("Custom reader JAR does not exist: " + jarPath);
        }
        if (!Files.isRegularFile(jarPath)) {
            throw new IOException(
                    "Custom reader JAR path is not a regular file: " + jarPath
            );
        }
        if (!Files.isReadable(jarPath)) {
            throw new IOException("Custom reader JAR is not readable: " + jarPath);
        }
        return jarPath;
    }

    private static URL toUrl(Path jarPath) throws MalformedURLException {
        return jarPath.toUri().toURL();
    }

    private static void validateImplementationClass(
            Class<?> implementationClass,
            Class<?> requiredType,
            ParsedReaderDescriptor descriptor
    ) {
        if (!requiredType.isAssignableFrom(implementationClass)) {
            throw new IllegalArgumentException(
                    "Custom reader class " + descriptor.className()
                            + " must implement " + requiredType.getName() + "."
            );
        }
        int modifiers = implementationClass.getModifiers();
        if (implementationClass.isInterface()
                || Modifier.isAbstract(modifiers)) {
            throw new IllegalArgumentException(
                    "Custom reader implementation must be a concrete class: "
                            + descriptor.className()
            );
        }
        if (implementationClass.isMemberClass()
                && !Modifier.isStatic(modifiers)) {
            throw new IllegalArgumentException(
                    "Custom reader implementation cannot be a non-static inner "
                            + "class: " + descriptor.className()
            );
        }
        if (implementationClass.isAnonymousClass()
                || implementationClass.isLocalClass()) {
            throw new IllegalArgumentException(
                    "Custom reader implementation must be a named top-level "
                            + "class or static nested class: "
                            + descriptor.className()
            );
        }
    }

    private static Constructor<?> requireNoArgumentConstructor(
            Class<?> implementationClass,
            ParsedReaderDescriptor descriptor,
            Path jarPath
    ) throws NoSuchMethodException {
        try {
            Constructor<?> constructor = implementationClass.getConstructor();
            if (!Modifier.isPublic(constructor.getModifiers())) {
                throw new IllegalArgumentException(
                        "The no-argument constructor for custom reader "
                                + descriptor.className() + " must be public."
                );
            }
            return constructor;
        } catch (NoSuchMethodException failure) {
            NoSuchMethodException focused = new NoSuchMethodException(
                    "Custom reader class " + descriptor.className()
                            + " from JAR " + jarPath
                            + " must provide a public no-argument constructor."
            );
            focused.initCause(failure);
            throw focused;
        }
    }

    private static <T> T instantiate(
            Constructor<?> constructor,
            Class<T> requiredType,
            ParsedReaderDescriptor descriptor,
            Path jarPath
    ) throws ReflectiveOperationException {
        Object instance;
        try {
            instance = constructor.newInstance();
        } catch (InvocationTargetException failure) {
            Throwable cause = failure.getCause();
            ReflectiveOperationException focused = new ReflectiveOperationException(
                    "Custom reader constructor failed for "
                            + descriptor.className() + " from JAR " + jarPath,
                    cause == null ? failure : cause
            );
            throw focused;
        }
        return requiredType.cast(instance);
    }

    private static void validateClassName(
            String className,
            String descriptor
    ) {
        String[] components = className.split("\\.", -1);
        for (String component : components) {
            if (component.isEmpty()
                    || !isValidBinaryNameComponent(component)) {
                throw new IllegalArgumentException(
                        "Invalid custom reader class name '" + className
                                + "' in descriptor: " + descriptor
                );
            }
        }
    }

    private static boolean isValidBinaryNameComponent(String component) {
        if (component.isEmpty()) {
            return false;
        }
        char first = component.charAt(0);
        if (!Character.isJavaIdentifierStart(first) && first != '$') {
            return false;
        }
        for (int index = 1; index < component.length(); index++) {
            char current = component.charAt(index);
            if (!Character.isJavaIdentifierPart(current) && current != '$') {
                return false;
            }
        }
        return true;
    }

    private record LoadedPlugin<T>(
            T reader,
            URLClassLoader classLoader,
            String descriptor,
            String implementationClassName
    ) {
        private LoadedPlugin {
            Objects.requireNonNull(reader, "Loaded plugin reader cannot be null.");
            Objects.requireNonNull(
                    classLoader,
                    "Loaded plugin class loader cannot be null."
            );
        }
    }

    private record ParsedReaderDescriptor(
            Path jarPath,
            String className,
            String normalizedDescriptor
    ) {
        private ParsedReaderDescriptor {
            Objects.requireNonNull(
                    jarPath,
                    "Parsed reader JAR path cannot be null."
            );
            Objects.requireNonNull(
                    className,
                    "Parsed reader class name cannot be null."
            );
            Objects.requireNonNull(
                    normalizedDescriptor,
                    "Normalized reader descriptor cannot be null."
            );
        }
    }
}
