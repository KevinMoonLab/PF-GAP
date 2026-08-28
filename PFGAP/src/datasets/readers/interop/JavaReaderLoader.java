package datasets.readers.interop;

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
 * Loads user-defined CustomSeriesReader implementations from external JAR
 * files.
 *
 * <p>Canonical descriptor format:</p>
 *
 * <pre>
 * javareader:/path/to/readers.jar:my.package.MyReader
 * </pre>
 *
 * <p>Windows paths are also supported:</p>
 *
 * <pre>
 * javareader:C:\path\to\readers.jar:my.package.MyReader
 * </pre>
 *
 * <p>The descriptor is split at its final colon rather than every colon.
 * This preserves drive-letter paths such as {@code C:\...}.</p>
 *
 * <p>The initial implementation intentionally supports JAR files only.
 * Requiring a JAR and a fully qualified class name avoids the classpath-root
 * ambiguity associated with loading individual packaged {@code .class}
 * files. It also permits the plugin to include helper classes and resources.</p>
 *
 * <p>The loaded implementation must:</p>
 *
 * <ul>
 *     <li>Be a concrete class</li>
 *     <li>Implement {@link CustomSeriesReader}</li>
 *     <li>Have an accessible public no-argument constructor</li>
 * </ul>
 *
 * <p>The returned {@link LoadedCustomReader} owns both the plugin instance
 * and its {@link URLClassLoader}. Callers must keep that object open for as
 * long as the custom reader may be used, then close it during application or
 * experiment cleanup.</p>
 */
public final class JavaReaderLoader {

    public static final String DESCRIPTOR_PREFIX =
            "javareader:";

    private static final String JAR_EXTENSION =
            ".jar";

    private JavaReaderLoader() {
        /*
         * Utility class.
         */
    }

    /**
     * Loads a custom series reader from a JAR descriptor.
     *
     * @param descriptor descriptor in the form
     *                   {@code javareader:path/to/plugin.jar:package.Class}
     * @return loaded reader and owning class-loader resource
     * @throws IOException if the JAR cannot be accessed or the class loader
     *                     cannot be closed after a failed load
     * @throws ReflectiveOperationException if the implementation class cannot
     *                                      be loaded or instantiated
     */
    public static LoadedCustomReader load(
            String descriptor
    ) throws IOException, ReflectiveOperationException {

        ParsedReaderDescriptor parsed =
                parseDescriptor(
                        descriptor
                );

        Path jarPath =
                validateJarPath(
                        parsed.jarPath()
                );

        URL jarUrl =
                toUrl(
                        jarPath
                );

        URLClassLoader classLoader =
                new URLClassLoader(
                        new URL[]{jarUrl},
                        CustomSeriesReader.class.getClassLoader()
                );

        boolean loadSucceeded =
                false;

        try {
            Class<?> implementationClass =
                    Class.forName(
                            parsed.className(),
                            true,
                            classLoader
                    );

            validateImplementationClass(
                    implementationClass,
                    parsed,
                    jarPath
            );

            Constructor<?> constructor =
                    requireNoArgumentConstructor(
                            implementationClass,
                            parsed,
                            jarPath
                    );

            CustomSeriesReader reader =
                    instantiate(
                            constructor,
                            parsed,
                            jarPath
                    );

            LoadedCustomReader loadedReader =
                    new LoadedCustomReader(
                            reader,
                            classLoader,
                            parsed.normalizedDescriptor(),
                            parsed.className()
                    );

            loadSucceeded =
                    true;

            return loadedReader;
        } finally {
            if (!loadSucceeded) {
                classLoader.close();
            }
        }
    }

    /**
     * Parses and validates the descriptor syntax without opening the JAR.
     *
     * @param descriptor custom-reader descriptor
     * @return parsed descriptor
     */
    private static ParsedReaderDescriptor parseDescriptor(
            String descriptor
    ) {
        if (descriptor == null || descriptor.isBlank()) {
            throw new IllegalArgumentException(
                    "Custom reader descriptor cannot be null or blank. "
                            + "Expected format: "
                            + DESCRIPTOR_PREFIX
                            + "/path/to/readers.jar:my.package.MyReader"
            );
        }

        String trimmed =
                descriptor.trim();

        if (!trimmed.regionMatches(
                true,
                0,
                DESCRIPTOR_PREFIX,
                0,
                DESCRIPTOR_PREFIX.length()
        )) {
            throw new IllegalArgumentException(
                    "Invalid custom reader descriptor prefix. "
                            + "Expected '"
                            + DESCRIPTOR_PREFIX
                            + "' but received: "
                            + descriptor
            );
        }

        String body =
                trimmed.substring(
                        DESCRIPTOR_PREFIX.length()
                );

        if (body.isBlank()) {
            throw new IllegalArgumentException(
                    "Custom reader descriptor must include a JAR path and "
                            + "implementation class. Expected format: "
                            + DESCRIPTOR_PREFIX
                            + "/path/to/readers.jar:my.package.MyReader"
            );
        }

        /*
         * Split at the final colon so a Windows drive-letter colon remains
         * part of the JAR path.
         */
        int classSeparator =
                body.lastIndexOf(':');

        if (classSeparator <= 0
                || classSeparator == body.length() - 1) {

            throw new IllegalArgumentException(
                    "Invalid custom reader descriptor. Expected format: "
                            + DESCRIPTOR_PREFIX
                            + "/path/to/readers.jar:my.package.MyReader. "
                            + "Received: "
                            + descriptor
            );
        }

        String pathText =
                body.substring(
                        0,
                        classSeparator
                ).trim();

        String className =
                body.substring(
                        classSeparator + 1
                ).trim();

        if (pathText.isEmpty()) {
            throw new IllegalArgumentException(
                    "Custom reader descriptor contains an empty JAR path: "
                            + descriptor
            );
        }

        if (className.isEmpty()) {
            throw new IllegalArgumentException(
                    "Custom reader descriptor contains an empty "
                            + "implementation class name: "
                            + descriptor
            );
        }

        validateClassName(
                className,
                descriptor
        );

        Path jarPath;

        try {
            jarPath =
                    Path.of(
                                    pathText
                            )
                            .toAbsolutePath()
                            .normalize();
        } catch (InvalidPathException e) {
            throw new IllegalArgumentException(
                    "Custom reader descriptor contains an invalid JAR path: "
                            + pathText,
                    e
            );
        }

        String normalizedDescriptor =
                DESCRIPTOR_PREFIX
                        + jarPath
                        + ":"
                        + className;

        return new ParsedReaderDescriptor(
                jarPath,
                className,
                normalizedDescriptor
        );
    }

    private static Path validateJarPath(
            Path jarPath
    ) throws IOException {

        Objects.requireNonNull(
                jarPath,
                "Custom reader JAR path cannot be null."
        );

        String fileName =
                jarPath.getFileName() == null
                        ? ""
                        : jarPath.getFileName()
                        .toString();

        if (!fileName.toLowerCase(Locale.ROOT)
                .endsWith(JAR_EXTENSION)) {

            throw new IllegalArgumentException(
                    "Custom reader plugins must currently be supplied as "
                            + "JAR files. Received: "
                            + jarPath
            );
        }

        if (!Files.exists(jarPath)) {
            throw new IOException(
                    "Custom reader JAR does not exist: "
                            + jarPath
            );
        }

        if (!Files.isRegularFile(jarPath)) {
            throw new IOException(
                    "Custom reader JAR path is not a regular file: "
                            + jarPath
            );
        }

        if (!Files.isReadable(jarPath)) {
            throw new IOException(
                    "Custom reader JAR is not readable: "
                            + jarPath
            );
        }

        return jarPath;
    }

    private static URL toUrl(
            Path jarPath
    ) {
        try {
            return jarPath.toUri()
                    .toURL();
        } catch (MalformedURLException e) {
            throw new IllegalArgumentException(
                    "Could not convert custom reader JAR path to URL: "
                            + jarPath,
                    e
            );
        }
    }

    private static void validateImplementationClass(
            Class<?> implementationClass,
            ParsedReaderDescriptor descriptor,
            Path jarPath
    ) {
        if (!CustomSeriesReader.class.isAssignableFrom(
                implementationClass
        )) {
            throw new IllegalArgumentException(
                    "Custom reader class "
                            + descriptor.className()
                            + " from JAR "
                            + jarPath
                            + " does not implement "
                            + CustomSeriesReader.class.getName()
                            + "."
            );
        }

        int modifiers =
                implementationClass.getModifiers();

        if (implementationClass.isInterface()) {
            throw new IllegalArgumentException(
                    "Custom reader implementation cannot be an interface: "
                            + descriptor.className()
            );
        }

        if (Modifier.isAbstract(modifiers)) {
            throw new IllegalArgumentException(
                    "Custom reader implementation cannot be abstract: "
                            + descriptor.className()
            );
        }

        if (!Modifier.isPublic(modifiers)) {
            throw new IllegalArgumentException(
                    "Custom reader implementation class must be public: "
                            + descriptor.className()
            );
        }

        if (implementationClass.isMemberClass()
                && !Modifier.isStatic(modifiers)) {

            throw new IllegalArgumentException(
                    "Custom reader implementation cannot be a non-static "
                            + "inner class: "
                            + descriptor.className()
            );
        }

        if (implementationClass.isAnonymousClass()
                || implementationClass.isLocalClass()) {

            throw new IllegalArgumentException(
                    "Custom reader implementation must be a named "
                            + "top-level class or static nested class: "
                            + descriptor.className()
            );
        }
    }

    private static Constructor<?> requireNoArgumentConstructor(
            Class<?> implementationClass,
            ParsedReaderDescriptor descriptor,
            Path jarPath
    ) throws NoSuchMethodException {

        Constructor<?> constructor;

        try {
            constructor =
                    implementationClass.getConstructor();
        } catch (NoSuchMethodException e) {
            throw new NoSuchMethodException(
                    "Custom reader class "
                            + descriptor.className()
                            + " from JAR "
                            + jarPath
                            + " must provide a public no-argument "
                            + "constructor."
            );
        }

        if (!Modifier.isPublic(
                constructor.getModifiers()
        )) {
            throw new IllegalArgumentException(
                    "The no-argument constructor for custom reader "
                            + descriptor.className()
                            + " must be public."
            );
        }

        return constructor;
    }

    private static CustomSeriesReader instantiate(
            Constructor<?> constructor,
            ParsedReaderDescriptor descriptor,
            Path jarPath
    ) throws ReflectiveOperationException {

        Object instance;

        try {
            instance =
                    constructor.newInstance();
        } catch (InvocationTargetException e) {
            Throwable cause =
                    e.getCause();

            String message =
                    "Constructor for custom reader "
                            + descriptor.className()
                            + " from JAR "
                            + jarPath
                            + " threw an exception";

            if (cause != null
                    && cause.getMessage() != null
                    && !cause.getMessage().isBlank()) {

                message +=
                        ": "
                                + cause.getMessage();
            }

            ReflectiveOperationException wrapped =
                    new ReflectiveOperationException(
                            message,
                            cause == null
                                    ? e
                                    : cause
                    );

            for (Throwable suppressed : e.getSuppressed()) {
                wrapped.addSuppressed(
                        suppressed
                );
            }

            throw wrapped;
        }

        return CustomSeriesReader.class.cast(
                instance
        );
    }

    /**
     * Performs a conservative syntactic check of a binary Java class name.
     *
     * <p>Both ordinary top-level names and static nested binary names such as
     * {@code package.Outer$Reader} are accepted.</p>
     */
    private static void validateClassName(
            String className,
            String descriptor
    ) {
        if (className.startsWith(".")
                || className.endsWith(".")
                || className.contains("..")) {

            throw new IllegalArgumentException(
                    "Invalid custom reader class name in descriptor: "
                            + descriptor
            );
        }

        String[] components =
                className.split(
                        "\\.",
                        -1
                );

        for (String component : components) {
            if (component.isEmpty()
                    || !isValidBinaryNameComponent(component)) {

                throw new IllegalArgumentException(
                        "Invalid custom reader class name '"
                                + className
                                + "' in descriptor: "
                                + descriptor
                );
            }
        }
    }

    private static boolean isValidBinaryNameComponent(
            String component
    ) {
        if (component.isEmpty()) {
            return false;
        }

        char first =
                component.charAt(0);

        if (!Character.isJavaIdentifierStart(first)
                && first != '$') {

            return false;
        }

        for (int index = 1;
             index < component.length();
             index++) {

            char current =
                    component.charAt(index);

            if (!Character.isJavaIdentifierPart(current)
                    && current != '$') {

                return false;
            }
        }

        return true;
    }

    /**
     * Immutable parsed descriptor.
     */
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