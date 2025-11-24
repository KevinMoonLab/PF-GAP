package distance.elastic;

import java.io.Serializable;

public class Cosine implements Serializable {
    public Cosine() {

    }

    private synchronized double dotProd(double[] s, double[] t, double bsf) {

        int i = 0;
        double total = 0;

        //assume s.length == t.length for this implementation
        //TODO note <=, if bsf = 0, < will cause problems when early abandoning
        for (i = 0; i < s.length & total <= bsf; i++){
            total += s[i] * t[i];
        }

        return total;
    }

    public synchronized double distance(Object S, Object T, double bsf){

        double[] s = (double[]) S;
        double[] t = (double[]) T;

        double numer = dotProd(s, t, bsf);
        double denom = Math.sqrt(dotProd(s, s, bsf) * dotProd(t, t, bsf));

        return numer/denom;
    }

    public synchronized double distance(Object S, Object T){ //for knn imputation

        double[] s = (double[]) S;
        double[] t = (double[]) T;
        int bsf = Math.min(t.length, s.length);

        double numer = dotProd(s, t, bsf);
        double denom = Math.sqrt(dotProd(s, s, bsf) * dotProd(t, t, bsf));

        return numer/denom;
    }
}