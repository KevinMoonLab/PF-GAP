package distance.interop;

import distance.api.DistanceFunction;

import java.io.File;
import java.net.URL;
import java.net.URLClassLoader;

public class JavaDistanceLoader {

    static DistanceFunction load(String descriptor) throws Exception {
        String[] parts = descriptor.split(":");
        if (parts.length < 2) {
            throw new IllegalArgumentException("Invalid descriptor format. Use javadistance:path/to/file[:ClassName]");
        }

        String path = parts[1];
        String className = parts.length == 3 ? parts[2] : inferClassName(path);

        File file = new File(path);
        if (!file.exists()) {
            throw new IllegalArgumentException("File not found: " + path);
        }

        /*File parent = file.getParentFile();
        if (parent == null) {
            // Fallback to current working directory
            parent = new File(".").getAbsoluteFile();
        }

        URL[] urls = { parent.toURI().toURL() };*/

        URL[] urls;
        if (path.endsWith(".jar")) {
            // ✅ Load directly from the .jar file
            urls = new URL[]{ file.toURI().toURL() };
        } else {
            // ✅ Load from the parent directory of the .class file
            File parent = file.getParentFile();
            if (parent == null) {
                parent = new File(".").getAbsoluteFile();
            }
            urls = new URL[]{ parent.toURI().toURL() };
        }

        /*try (URLClassLoader loader = new URLClassLoader(urls, DistanceFunction.class.getClassLoader())) {
            Class<?> clazz = loader.loadClass(className);

            if (!DistanceFunction.class.isAssignableFrom(clazz)) {
                throw new IllegalArgumentException("Class does not implement DistanceFunction");
            }

            return (DistanceFunction) clazz.getDeclaredConstructor().newInstance();
        }*/
        URLClassLoader loader = new URLClassLoader(urls, DistanceFunction.class.getClassLoader());
        Class<?> clazz = loader.loadClass(className);

        if (!DistanceFunction.class.isAssignableFrom(clazz)) {
            throw new IllegalArgumentException("Class does not implement DistanceFunction");
        }

        return (DistanceFunction) clazz.getDeclaredConstructor().newInstance();

    }

    private static String inferClassName(String path) {
        String name = new File(path).getName();
        if (name.endsWith(".class")) {
            return name.substring(0, name.length() - 6); // remove .class
        }
        throw new IllegalArgumentException("Class name must be provided for .jar files");
    }
}
