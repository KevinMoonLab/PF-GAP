package imputation.initial;

import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.util.Arrays;
import java.util.Objects;

/**
 * Invokes a distance after retaining only positions observed in both inputs.
 * Numeric inputs support matching primitive double/float 1D or 2D arrays;
 * categorical inputs support Object arrays with null missing values.
 */
public final class MaskedDistance {
    private final Object distanceInstance;
    private final Method distanceMethod;

    public MaskedDistance(Object distanceInstance) {
        this.distanceInstance = Objects.requireNonNull(distanceInstance, "Distance instance cannot be null.");
        try {
            this.distanceMethod = distanceInstance.getClass().getMethod(
                    "distance",
                    Object.class,
                    Object.class
            );
        } catch (NoSuchMethodException exception) {
            throw new IllegalArgumentException(
                    "Distance class must expose distance(Object, Object): "
                            + distanceInstance.getClass().getName(),
                    exception
            );
        }
    }

    public double compute(Object first, Object second) {
        FilteredPair filtered = filter(first, second);
        if (filtered.empty()) return Double.POSITIVE_INFINITY;
        try {
            Object result = distanceMethod.invoke(
                    distanceInstance,
                    filtered.first(),
                    filtered.second()
            );
            if (!(result instanceof Number number)) {
                throw new IllegalStateException("Distance method did not return a numeric value.");
            }
            return number.doubleValue();
        } catch (IllegalAccessException exception) {
            throw new IllegalStateException("Cannot access distance(Object, Object).", exception);
        } catch (InvocationTargetException exception) {
            Throwable cause = exception.getCause();
            if (cause instanceof RuntimeException runtime) throw runtime;
            throw new IllegalStateException("Distance invocation failed.", cause);
        }
    }

    private static FilteredPair filter(Object first, Object second) {
        if (first instanceof double[] a && second instanceof double[] b) return filterDouble1D(a,b);
        if (first instanceof float[] a && second instanceof float[] b) return filterFloat1D(a,b);
        if (first instanceof double[][] a && second instanceof double[][] b) return filterDouble2D(a,b);
        if (first instanceof float[][] a && second instanceof float[][] b) return filterFloat2D(a,b);
        if (first instanceof Object[] a && second instanceof Object[] b) return filterObject1D(a,b);
        if (first instanceof Object[][] a && second instanceof Object[][] b) return filterObject2D(a,b);
        throw new IllegalArgumentException(
                "MaskedDistance requires matching primitive numeric or Object-array inputs. Received "
                        + type(first) + " and " + type(second) + "."
        );
    }

    private static FilteredPair filterDouble1D(double[] a,double[] b){
        requireSameLength(a.length,b.length); int count=0;
        for(int i=0;i<a.length;i++)if(!Double.isNaN(a[i])&&!Double.isNaN(b[i]))count++;
        double[] af=new double[count],bf=new double[count]; int out=0;
        for(int i=0;i<a.length;i++)if(!Double.isNaN(a[i])&&!Double.isNaN(b[i])){af[out]=a[i];bf[out++]=b[i];}
        return new FilteredPair(af,bf,count==0);
    }
    private static FilteredPair filterFloat1D(float[] a,float[] b){
        requireSameLength(a.length,b.length); int count=0;
        for(int i=0;i<a.length;i++)if(!Float.isNaN(a[i])&&!Float.isNaN(b[i]))count++;
        float[] af=new float[count],bf=new float[count]; int out=0;
        for(int i=0;i<a.length;i++)if(!Float.isNaN(a[i])&&!Float.isNaN(b[i])){af[out]=a[i];bf[out++]=b[i];}
        return new FilteredPair(af,bf,count==0);
    }
    private static FilteredPair filterObject1D(Object[] a,Object[] b){
        requireSameLength(a.length,b.length); int count=0;
        for(int i=0;i<a.length;i++)if(a[i]!=null&&b[i]!=null)count++;
        Object[] af=new Object[count],bf=new Object[count]; int out=0;
        for(int i=0;i<a.length;i++)if(a[i]!=null&&b[i]!=null){af[out]=a[i];bf[out++]=b[i];}
        return new FilteredPair(af,bf,count==0);
    }

    private static FilteredPair filterDouble2D(double[][] a,double[][] b){
        requireSameLength(a.length,b.length); double[][] af=new double[a.length][],bf=new double[b.length][]; int total=0;
        for(int d=0;d<a.length;d++){FilteredPair p=filterDouble1D(Objects.requireNonNull(a[d]),Objects.requireNonNull(b[d]));af[d]=(double[])p.first();bf[d]=(double[])p.second();total+=af[d].length;}
        return new FilteredPair(af,bf,total==0);
    }
    private static FilteredPair filterFloat2D(float[][] a,float[][] b){
        requireSameLength(a.length,b.length); float[][] af=new float[a.length][],bf=new float[b.length][]; int total=0;
        for(int d=0;d<a.length;d++){FilteredPair p=filterFloat1D(Objects.requireNonNull(a[d]),Objects.requireNonNull(b[d]));af[d]=(float[])p.first();bf[d]=(float[])p.second();total+=af[d].length;}
        return new FilteredPair(af,bf,total==0);
    }
    private static FilteredPair filterObject2D(Object[][] a,Object[][] b){
        requireSameLength(a.length,b.length); Object[][] af=new Object[a.length][],bf=new Object[b.length][]; int total=0;
        for(int d=0;d<a.length;d++){FilteredPair p=filterObject1D(Objects.requireNonNull(a[d]),Objects.requireNonNull(b[d]));af[d]=(Object[])p.first();bf[d]=(Object[])p.second();total+=af[d].length;}
        return new FilteredPair(af,bf,total==0);
    }

    private static void requireSameLength(int a,int b){if(a!=b)throw new IllegalArgumentException("Masked inputs must have matching lengths.");}
    private static String type(Object value){return value==null?"null":value.getClass().getTypeName();}
    private record FilteredPair(Object first,Object second,boolean empty){}
}
