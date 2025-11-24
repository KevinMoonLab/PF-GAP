package distance.elastic;

import java.io.Serializable;

public class Euclidean implements Serializable {
	public Euclidean() {

	}	
		
	//public synchronized double distance(double[] s, double[] t, double bsf){
	public synchronized double distance(Object S, Object T, double bsf){

		double[] s = (double[]) S;
		double[] t = (double[]) T;

		int i = 0;
		double total = 0;

		//assume s.length == t.length for this implementation
		//TODO note <=, if bsf = 0, < will cause problems when early abandoning
		for (i = 0; i < s.length & total <= bsf; i++){
			total += (s[i] - t[i]) * (s[i] - t[i]);
		}
		
//		System.out.println("Euclidean: early abandon after: " + i + " from: " + s.length);

//		return Math.sqrt(total);
		return total;
	}

	public synchronized double distance(Object S, Object T){ //for knn imputation

		Double[] s = (Double[]) T;
		Double[] t = (Double[]) S;

		int i = 0;
		double total = 0;

		//assume s.length == t.length for this implementation
		for (i = 0; i < s.length; i++){
			total += (s[i] - t[i]) * (s[i] - t[i]);
		}


		return Math.sqrt(total);
	}
}
