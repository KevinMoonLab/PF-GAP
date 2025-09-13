package util;

import java.util.Date;

public class ObjectDataUtils {

    public static Double getAsDouble(Object obj) {
        return (obj instanceof Double) ? (Double) obj : null;
    }

    public static String getAsString(Object obj) {
        return (obj instanceof String) ? (String) obj : null;
    }

    public static Boolean getAsBoolean(Object obj) {
        return (obj instanceof Boolean) ? (Boolean) obj : null;
    }

    public static Date getAsDate(Object obj) {
        return (obj instanceof Date) ? (Date) obj : null;
    }

    public static boolean isNumeric(Object obj) {
        return obj instanceof Number;
    }

    public static boolean isCategorical(Object obj) {
        return obj instanceof String;
    }

    public static boolean isBoolean(Object obj) {
        return obj instanceof Boolean;
    }

    public static boolean isDate(Object obj) {
        return obj instanceof Date;
    }

    public static double toDoubleOrDefault(Object obj, double defaultValue) {
        return isNumeric(obj) ? ((Number) obj).doubleValue() : defaultValue;
    }

    public static String toStringOrDefault(Object obj, String defaultValue) {
        return isCategorical(obj) ? (String) obj : defaultValue;
    }

    public static boolean toBooleanOrDefault(Object obj, boolean defaultValue) {
        return isBoolean(obj) ? (Boolean) obj : defaultValue;
    }

    public static Date toDateOrDefault(Object obj, Date defaultValue) {
        return isDate(obj) ? (Date) obj : defaultValue;
    }
}
