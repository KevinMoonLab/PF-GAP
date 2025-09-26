package distance.elastic;

import java.io.Serializable;

public class Manhattan implements Serializable {
    public Manhattan() {

    }

    //public synchronized double distance(double[] s, double[] t, double bsf){
    public synchronized double distance(Object S, Object T, double bsf){

        double[] t = (double[]) T;
        double[] s = (double[]) S;

        int i = 0;
        double total = 0;
        //Integer[] plist = new Integer[]{1,3,4};

        //assume s.length == t.length for this implementation
        //TODO note <=, if bsf = 0, < will cause problems when early abandoning
        for (i = 0; i < s.length & total <= bsf; i++){
            total += Math.abs(s[i] - t[i]);
        }

//		System.out.println("Euclidean: early abandon after: " + i + " from: " + s.length);

//		return Math.sqrt(total);
        return total;
    }
}
