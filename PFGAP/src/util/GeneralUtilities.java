package util;

import java.io.File;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.util.List;

import org.apache.commons.lang3.time.DurationFormatUtils;

public class GeneralUtilities {

	public static String getCurrentTimeStamp(String format) {
		return LocalDateTime.now().format(DateTimeFormatter.ofPattern(format));
	}
	
	public static String formatTime(long duration, String format) {
		return DurationFormatUtils.formatDuration((long) duration, "H:m:s.SSS");
	}
	
	public static void warmUpJavaRuntime() {
		//TODO
		System.out.println("TODO doing some extra work to warm up jvm..."
				+ "this helps to measure time more accurately for short experiments");
	}


	/**
	 * Writes a List<Object> to a file using the original format.
	 * Each Object is either Object[], Object[][], or their primitive equivalents.
	 *
	 * @param data The data to write
	 * @param filePath The output file path
	 * @param firstSeparator Separator between top-level objects (array separator)
	 * @param secondSeparator Separator within nested arrays (entry separator)
	 * @throws IOException If writing fails
	 */
	public static void writeDelimitedData(List<Object> data, String filePath, String firstSeparator, String secondSeparator) throws IOException {
		// switching the arguments above is a "cheap" and confusing fix...
		File file = new File(filePath);
		File parentDir = file.getParentFile();
		if (parentDir != null && !parentDir.exists()) {
			parentDir.mkdirs(); // Create missing directories
		}

		try (BufferedWriter writer = new BufferedWriter(new FileWriter(filePath))) {
			for (Object obj : data) {
				if (obj instanceof Object[][]) {
					Object[][] matrix = (Object[][]) obj;
					for (Object[] row : matrix) {
						writer.write(join(row, secondSeparator));
						writer.write(firstSeparator);
					}
				} else if (obj instanceof double[][]) {
					double[][] matrix = (double[][]) obj;
					for (double[] row : matrix) {
						writer.write(join(row, secondSeparator));
						writer.write(firstSeparator);
					}
				} else if (obj instanceof Object[]) {
					writer.write(join((Object[]) obj, secondSeparator));
					//writer.write(firstSeparator);
				} else if (obj instanceof double[]) {
					writer.write(join((double[]) obj, secondSeparator));
					//writer.write(firstSeparator);
				} else if (obj instanceof Double[]) {
					writer.write(join((Double[]) obj, secondSeparator));
					//writer.write(firstSeparator);
				} else {
					writer.write(obj.toString());
					//writer.write(firstSeparator);
				}
				writer.newLine();
			}
		}
	}

	private static String join(Object[] array, String sep) {
		StringBuilder sb = new StringBuilder();
		for (int i = 0; i < array.length; i++) {
			sb.append(array[i]);
			if (i < array.length - 1) sb.append(sep);
		}
		return sb.toString();
	}

	private static String join(double[] array, String sep) {
		StringBuilder sb = new StringBuilder();
		for (int i = 0; i < array.length; i++) {
			sb.append(array[i]);
			if (i < array.length - 1) sb.append(sep);
		}
		return sb.toString();
	}

	private static String join(Double[] array, String sep) {
		StringBuilder sb = new StringBuilder();
		for (int i = 0; i < array.length; i++) {
			sb.append(array[i]);
			if (i < array.length - 1) sb.append(sep);
		}
		return sb.toString();
	}
	
}
