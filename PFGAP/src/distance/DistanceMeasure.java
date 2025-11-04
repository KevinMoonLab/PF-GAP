package distance;

import java.io.IOException;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import core.AppContext;
import core.contracts.ObjectDataset;
import distance.api.DistanceFunction;
import distance.elastic.*;
import distance.graph.*;
import distance.interop.*;
import distance.meta.*;
import distance.multiTS.*;

public class DistanceMeasure implements Serializable {
	
	public final MEASURE distance_measure;

	private Euclidean euc;
	private DTW dtw;
	private DTW dtwcv;
	private DDTW ddtw;
	private DDTW ddtwcv;
	private WDTW wdtw;
	private WDDTW wddtw;
	private LCSS lcss;
	private MSM msm;
	private ERP erp;
	private TWE twe;
	private MapleDistance maple;
	private PythonDistance python;
	private Manhattan manhattan;
	private DistanceFunction distanceFunction;
	private ShapeHoG1dDTW shapeHoG1dDTW;
	private DTW_I dtw_i;
	private DTW_D dtw_d;

	private DDTW_I ddtw_i;
	private WDTW_I wdtw_i;
	private WDDTW_I wddtw_i;
	private TWE_I twe_i;
	private ERP_I erp_i;
	private Euclidean_I euclidean_i;
	private LCSS_I lcss_i;
	private MSM_I msm_i;
	private Manhattan_I manhattan_i;
	private CID_I cid_i;
	private SBD_I sbd_i;
	private ShapeHoGDTW shapeHoGdtw;

	private DDTW_D ddtw_d;
	private WDTW_D wdtw_d;
	private WDDTW_D wddtw_d;
	private ShapeHoGDTW shapeHoGdtw_d;
	//private Euclidean_D euclidean_d;
	//private Manhattan_D manhattan_d;

	//graph-based distances
	private ApproximateGraphEditDistance approximateGraphEditDistance;
	private GraphletDistance graphletDistance;
	private GraphEditDistance graphEditDistance;
	private HammingDistance hammingDistance;
	private ShortestPathDistance shortestPathDistance;
	private WLDistance wlDistance;
	private WLDistance2 wlDistance2;

	//meta-based distances
	private MetaClassMatchDistance meta_classmatch;
	private MetaFileClassMatchDistance meta_file_classmatch;
	private MetaRegressionDistance meta_regression;
	private MetaFileRegressionDistance meta_file_regression;

	
	public int windowSizeDTW =-1,
			windowSizeDDTW=-1, 
			windowSizeLCSS=-1,
			windowSizeERP=-1;
	public double epsilonLCSS = -1.0,
			gERP=-1.0,
			nuTWE,
			lambdaTWE,
			cMSM,
			weightWDTW,
			weightWDDTW;

	public DistanceMeasure (MEASURE m, String... descriptor) throws Exception{
		this.distance_measure = m;
		initialize(m, descriptor);
	}
	
	public void initialize (MEASURE m, String... descriptor) throws Exception{
		switch (m) {
			case euclidean:
			case shifazEUCLIDEAN:
				euc = new Euclidean();
				break;
			case erp:
			case shifazERP:
				erp = new ERP();
				break;
			case lcss:
			case shifazLCSS:
				lcss = new LCSS();
				break;
			case msm:
			case shifazMSM:
				msm = new MSM();
				break;
			case twe:
			case shifazTWE:
				twe = new TWE();
				break;
			case wdtw:
			case shifazWDTW:
				wdtw = new WDTW();
				break;
			case wddtw:
			case shifazWDDTW:
				wddtw = new WDDTW();
				break;
			case dtw:
			case shifazDTW:
				dtw = new DTW();
				break;
			case dtwcv:
			case shifazDTWCV:
				dtwcv = new DTW();
				break;
			case ddtw:
			case shifazDDTW:
				ddtw  = new DDTW();
				break;
			case ddtwcv:
			case shifazDDTWCV:
				ddtwcv = new DDTW();
				break;
			case maple:
				maple = new MapleDistance(descriptor[0]);
				break;
			case python:
				python = new PythonDistance(descriptor[0]); //PythonDistance();
				break;
			case javadistance:
				distanceFunction = new JavaDistance(descriptor[0]).getDistanceFunction();
				break;
			case meta_classmatch:
				meta_classmatch = new MetaClassMatchDistance(descriptor[0]);
				break;
			case meta_file_classmatch:
				meta_file_classmatch = new MetaFileClassMatchDistance(descriptor[0]);
				break;
			case meta_regression:
				meta_regression = new MetaRegressionDistance(descriptor[0]);
				break;
			case meta_file_regression:
				meta_file_regression = new MetaFileRegressionDistance(descriptor[0]);
				break;
			case manhattan:
				manhattan = new Manhattan();
				break;
			case shapeHoG1dDTW:
				shapeHoG1dDTW = new ShapeHoG1dDTW();
				break;
			case dtw_i:
				dtw_i = new DTW_I();
				break;
			case dtw_d:
				dtw_d = new DTW_D();
				break;
			case ddtw_i:
			case shifazDDTW_I:
				ddtw_i = new DDTW_I();
				break;
			case wdtw_i:
			case shifazWDTW_I:
				wdtw_i = new WDTW_I();
				break;
			case wddtw_i:
			case shifazWDDTW_I:
				wddtw_i = new WDDTW_I();
				break;
			case twe_i:
			case shifazTWE_I:
				twe_i = new TWE_I();
				break;
			case erp_i:
			case shifazERP_I:
				erp_i = new ERP_I();
				break;
			case euclidean_i:
			case shifazEUCLIDEAN_I:
				euclidean_i = new Euclidean_I();
				break;
			case lcss_i:
			case shifazLCSS_I:
				lcss_i = new LCSS_I();
				break;
			case msm_i:
			case shifazMSM_I:
				msm_i = new MSM_I();
				break;
			case manhattan_i:
			case shifazMANHATTAN_I:
				manhattan_i = new Manhattan_I();
				break;
			case cid_i:
			case shifazCID_I:
				cid_i = new CID_I();
				break;
			case sbd_i:
			case shifazSBD_I:
				sbd_i = new SBD_I();
				break;
			case shapeHoGdtw:
			case shifazShapeHoGDTW:
				shapeHoGdtw = new ShapeHoGDTW();
				break;

			case ddtw_d:
				ddtw_d = new DDTW_D();
				break;
			case wdtw_d:
				wdtw_d = new WDTW_D();
				break;
			case wddtw_d:
				wddtw_d = new WDDTW_D();
				break;
			case shapeHoGdtw_d:
				shapeHoGdtw_d = new ShapeHoGDTW();
				break;
			//case euclidean_d:
			//	euclidean_d = new Euclidean_D();
			//	break;
			//case manhattan_d:
			//	manhattan_d = new Manhattan_D();
			//	break;
			case approximateGraphEditDistance:
				approximateGraphEditDistance = new ApproximateGraphEditDistance();
				break;
			case graphEditDistance:
				graphEditDistance = new GraphEditDistance();
				break;
			case graphletDistance:
				graphletDistance = new GraphletDistance();
				break;
			case hammingDistance:
				hammingDistance = new HammingDistance();
				break;
			case shortestPathDistance:
				shortestPathDistance = new ShortestPathDistance();
				break;
			case wlDistance:
				wlDistance = new WLDistance();
				break;
			case wlDistance2:
				wlDistance2 = new WLDistance2();
				break;
			default:
				throw new Exception("Unknown distance measure");
//				break;
		}
		
	}

	public Object getDistanceInstance(MEASURE measure) {
		switch (measure) {
			case euclidean:
			case shifazEUCLIDEAN:
				return euc;
			case erp:
			case shifazERP:
				return erp;
			case lcss:
			case shifazLCSS:
				return lcss;
			case msm:
			case shifazMSM:
				return msm;
			case twe:
			case shifazTWE:
				return twe;
			case wdtw:
			case shifazWDTW:
				return wdtw;
			case wddtw:
			case shifazWDDTW:
				return wddtw;
			case dtw:
			case shifazDTW:
				return dtw;
			case dtwcv:
			case shifazDTWCV:
				return dtwcv;
			case ddtw:
			case shifazDDTW:
				return ddtw;
			case ddtwcv:
			case shifazDDTWCV:
				return ddtwcv;
			case maple:
				return maple;
			case python:
				return python;
			case javadistance:
				return distanceFunction;
			case manhattan:
				return manhattan;
			case shapeHoG1dDTW:
				return shapeHoG1dDTW;
			case dtw_i:
				return dtw_i;
			case dtw_d:
				return dtw_d;
			case ddtw_i:
			case shifazDDTW_I:
				return ddtw_i;
			case wdtw_i:
			case shifazWDTW_I:
				return wdtw_i;
			case wddtw_i:
			case shifazWDDTW_I:
				return wddtw_i;
			case twe_i:
			case shifazTWE_I:
				return twe_i;
			case erp_i:
			case shifazERP_I:
				return erp_i;
			case euclidean_i:
			case shifazEUCLIDEAN_I:
				return euclidean_i;
			case lcss_i:
			case shifazLCSS_I:
				return lcss_i;
			case msm_i:
			case shifazMSM_I:
				return msm_i;
			case manhattan_i:
			case shifazMANHATTAN_I:
				return manhattan_i;
			case cid_i:
			case shifazCID_I:
				return cid_i;
			case sbd_i:
			case shifazSBD_I:
				return sbd_i;
			case shapeHoGdtw:
			case shifazShapeHoGDTW:
				return shapeHoGdtw;
			case ddtw_d:
				return ddtw_d;
			case wdtw_d:
				return wdtw_d;
			case wddtw_d:
				return wddtw_d;
			case shapeHoGdtw_d:
				return shapeHoGdtw_d;
			case approximateGraphEditDistance:
				return approximateGraphEditDistance;
			case graphEditDistance:
				return graphEditDistance;
			case graphletDistance:
				return graphletDistance;
			case hammingDistance:
				return hammingDistance;
			case shortestPathDistance:
				return shortestPathDistance;
			case wlDistance:
				return wlDistance;
			case wlDistance2:
				return wlDistance2;
			case meta_classmatch:
				return meta_classmatch;
			case meta_file_classmatch:
				return meta_file_classmatch;
			case meta_regression:
				return meta_regression;
			case meta_file_regression:
				return meta_file_regression;
			default:
				throw new IllegalArgumentException("Unsupported measure: " + measure);
		}
	}

	public void select_random_params(ObjectDataset d, Random r) {
		switch (this.distance_measure) {
		case euclidean:
		case shifazEUCLIDEAN:

			break;
		case erp:
		case shifazERP:
			this.gERP = erp.get_random_g(d, r);
			this.windowSizeERP =  erp.get_random_window(d, r);
			break;
		case lcss:
		case shifazLCSS:
			this.epsilonLCSS = lcss.get_random_epsilon(d, r);
			this.windowSizeLCSS = lcss.get_random_window(d, r);
			break;
		case msm:
		case shifazMSM:
			this.cMSM = msm.get_random_cost(d, r);
			break;
		case twe:
		case shifazTWE:
			this.lambdaTWE = twe.get_random_lambda(d, r);
			this.nuTWE = twe.get_random_nu(d, r);
			break;
		case wdtw:
		case shifazWDTW:
			this.weightWDTW = wdtw.get_random_g(d, r);
			break;
		case wddtw:
		case shifazWDDTW:
			this.weightWDDTW = wddtw.get_random_g(d, r);
			break;
		case dtw:
		case shifazDTW:
			this.windowSizeDTW = d.length();	
			break;
		case dtwcv:
		case shifazDTWCV:
			this.windowSizeDTW = dtwcv.get_random_window(d, r);
			break;
		case ddtw:
		case shifazDDTW:
			this.windowSizeDDTW = d.length();	
			break;
		case shapeHoG1dDTW:
			this.windowSizeDDTW = d.length();
			break;
		case ddtwcv:
		case shifazDDTWCV:
			this.windowSizeDDTW = ddtwcv.get_random_window(d, r);
			break;
			case wdtw_i:
			case shifazWDTW_I:
				this.weightWDTW = wdtw_i.get_random_g(d, r);
				break;
			case wddtw_i:
			case shifazWDDTW_I:
				this.weightWDDTW = wddtw_i.get_random_g(d, r);
				break;
			case twe_i:
			case shifazTWE_I:
				this.lambdaTWE = twe_i.get_random_lambda(d, r);
				this.nuTWE = twe_i.get_random_nu(d, r);
				break;
			case erp_i:
			case shifazERP_I:
				this.gERP = erp_i.get_random_g(d, r);
				this.windowSizeERP = erp_i.get_random_window(d, r);
				break;
			case lcss_i:
			case shifazLCSS_I:
				this.epsilonLCSS = lcss_i.get_random_epsilon(d, r);
				this.windowSizeLCSS = lcss_i.get_random_window(d, r);
				break;
			case msm_i:
			case shifazMSM_I:
				this.cMSM = msm_i.get_random_cost(d, r);
				break;
			case shapeHoGdtw:
			case shifazShapeHoGDTW:
				this.windowSizeDDTW = d.length();
				break;
		default:
//			throw new Exception("Unknown distance measure");
//			break;
		}
	}

	public double distance(Object s, Object t) throws IOException, InterruptedException {
		return this.distance(s, t, Double.POSITIVE_INFINITY);
	}

	public double distance(Object s, Object t, double bsf) throws IOException, InterruptedException {
		double distance = Double.POSITIVE_INFINITY;
		
		switch (this.distance_measure) {
		case euclidean:
		case shifazEUCLIDEAN:
			distance = euc.distance(s, t, bsf);
			break;
		case erp:
		case shifazERP:
			distance = 	erp.distance(s, t, bsf, this.windowSizeERP, this.gERP);
			break;
		case lcss:
		case shifazLCSS:
			distance = lcss.distance(s, t, bsf, this.windowSizeLCSS, this.epsilonLCSS);
			break;
		case msm:
		case shifazMSM:
			distance = msm.distance(s, t, bsf, this.cMSM);
			break;
		case twe:
		case shifazTWE:
			distance = twe.distance(s, t, bsf, this.nuTWE, this.lambdaTWE);
			break;
		case wdtw:
		case shifazWDTW:
			distance = wdtw.distance(s, t, bsf, this.weightWDTW);
			break;
		case wddtw:
		case shifazWDDTW:
			distance = wddtw.distance(s, t, bsf, this.weightWDDTW);
			break;
		case dtw:
		case shifazDTW:
			distance = dtw.distance(s, t, bsf, ((double[]) s).length);
			break;
		case dtwcv:
		case shifazDTWCV:
			distance = 	dtwcv.distance(s, t, bsf, this.windowSizeDTW);
			break;
		case ddtw:
		case shifazDDTW:
			distance = ddtw.distance(s, t, bsf, ((double[]) s).length);
			break;
		case ddtwcv:
		case shifazDDTWCV:
			distance = ddtwcv.distance(s, t, bsf, this.windowSizeDDTW);
			break;
		case maple:
			//distance = MapleDistance.distance(s,t,dfile[0]);
			distance = maple.distance(s,t);
			//distance = MapleDistance.distance(s,t);
			break;
		case python:
			//distance = PythonDistance.distance(s,t,dfile[0]);
			distance = python.distance(s,t);
			//distance = PythonDistance.distance(s,t);
			break;
		case javadistance:
			distance = distanceFunction.compute(s,t);
			break;
		case manhattan:
			distance = manhattan.distance(s,t,bsf);
			break;
		case shapeHoG1dDTW:
			distance = shapeHoG1dDTW.distance(s,t,bsf,((double[]) s).length);
			break;
		case dtw_i:
			distance = dtw_i.distance(s,t,bsf,((double[][]) s).length);
			break;
		case dtw_d:
			distance = dtw_d.distance(s,t,bsf,((double[][]) s).length);
			break;
			case ddtw_i:
			case shifazDDTW_I:
				distance = ddtw_i.distance(s, t, bsf, ((double[][]) s).length);
				break;
			case wdtw_i:
			case shifazWDTW_I:
				distance = wdtw_i.distance(s, t, bsf, this.weightWDTW);
				break;
			case wddtw_i:
			case shifazWDDTW_I:
				distance = wddtw_i.distance(s, t, bsf, this.weightWDDTW);
				break;
			case twe_i:
			case shifazTWE_I:
				distance = twe_i.distance(s, t, bsf, this.nuTWE, this.lambdaTWE);
				break;
			case erp_i:
			case shifazERP_I:
				distance = erp_i.distance(s, t, bsf, this.windowSizeERP, this.gERP);
				break;
			case euclidean_i:
			case shifazEUCLIDEAN_I:
				distance = euclidean_i.distance(s, t, bsf);
				break;
			case lcss_i:
			case shifazLCSS_I:
				distance = lcss_i.distance(s, t, bsf, this.windowSizeLCSS, this.epsilonLCSS);
				break;
			case msm_i:
			case shifazMSM_I:
				distance = msm_i.distance(s, t, bsf, this.cMSM);
				break;
			case manhattan_i:
			case shifazMANHATTAN_I:
				distance = manhattan_i.distance(s, t, bsf);
				break;
			case cid_i:
			case shifazCID_I:
				distance = cid_i.distance(s, t, bsf);
				break;
			case sbd_i:
			case shifazSBD_I:
				distance = sbd_i.distance(s, t);
				break;
			case shapeHoGdtw:
			case shifazShapeHoGDTW:
				distance = shapeHoGdtw.distance(s, t, bsf, ((double[][]) s).length);
				break;

			case ddtw_d:
				distance = ddtw_d.distance(s, t, bsf, ((double[][]) s).length);
				break;
			case wdtw_d:
				distance = wdtw_d.distance(s, t, bsf, this.weightWDTW);
				break;
			case wddtw_d:
				distance = wddtw_d.distance(s, t, bsf, this.weightWDDTW);
				break;
			case shapeHoGdtw_d:
				distance = shapeHoGdtw_d.distance(s, t, bsf, ((double[][]) s).length);
				break;
			//case euclidean_d:
			//	distance = euclidean_d.distance(s, t, bsf);
			//	break;
			//case manhattan_d:
			//	distance = manhattan_d.distance(s, t, bsf);
			case approximateGraphEditDistance:
				distance = approximateGraphEditDistance.compute(s,t);
				break;
			case graphEditDistance:
				distance = approximateGraphEditDistance.compute(s,t);
				break;
			case graphletDistance:
				distance = graphletDistance.compute(s,t);
				break;
			case hammingDistance:
				distance = hammingDistance.compute(s,t);
				break;
			case shortestPathDistance:
				distance = shortestPathDistance.compute(s,t);
				break;
			case wlDistance:
				distance = wlDistance.compute(s,t);
				break;
			case wlDistance2:
				distance = wlDistance2.compute(s,t);
				break;
			case meta_classmatch:
				distance = meta_classmatch.distance(s,t);
				break;
			case meta_file_classmatch:
				distance = meta_file_classmatch.distance(s,t);
				break;
			case meta_regression:
				distance = meta_regression.distance(s,t);
				break;
			case meta_file_regression:
				distance = meta_file_regression.distance(s,t);
				break;

			default:
//			throw new Exception("Unknown distance measure");
//			break;
		}
		if (distance == Double.POSITIVE_INFINITY) {
			System.out.println("error ***********");
		}
		
		return distance;
	}
	
//	public double distance(int q, int c, double bsf, DMResult result){
////		return dm.distance(s, t, bsf, result);
//		return 0.0;
//	}	
	
	public String toString() {
		return this.distance_measure.toString(); //+ " [" + dm.toString() + "]";
	}
	
	//setters and getters
	
//	public void set_param(String key, Object val) {
//		this.dm.set_param(key, val);
//	}
//	
//	public Object get_param(String key) {
//		return this.dm.get_param(key);
//	}
	
	public void setWindowSizeDTW(int w){
		this.windowSizeDTW = w;
	}
	
	public void setWindowSizeDDTW(int w){
		this.windowSizeDDTW = w;
	}
	
	public void setWindowSizeLCSS(int w){
		this.windowSizeLCSS = w;
	}
	
	public void setWindowSizeERP(int w){
		this.windowSizeERP = w;
	}
	
	public void setEpsilonLCSS(double epsilon){
		this.epsilonLCSS = epsilon;
	}
	
	public void setGvalERP(double g){
		this.gERP= g;
	}
	
	public void setNuTWE(double nuTWE){
		this.nuTWE = nuTWE;
	}
	public void setLambdaTWE(double lambdaTWE){
		this.lambdaTWE = lambdaTWE;
	}
	public void setCMSM(double c){
		this.cMSM = c;
	}
	
	public void setWeigthWDTW(double g){
		this.weightWDTW = g;
	}
	
	public void setWeigthWDDTW(double g){
		this.weightWDDTW = g;
	}
	
	
	//just to reuse this data structure
	List<Integer> closest_nodes = new ArrayList<Integer>();
	
	//public int find_closest_node(
	//		double[] query,
	//		double[][] exemplars,
	//		boolean train,
	//		String... dfile) throws Exception{
	public int find_closest_node(
			Object query,
			Object[] exemplars,
			boolean train,
			String... dfile) throws Exception{
		closest_nodes.clear();
		double dist = Double.POSITIVE_INFINITY;
		double bsf = Double.POSITIVE_INFINITY;		

		for (int i = 0; i < exemplars.length; i++) {
			//double[] exemplar = exemplars[i];	//TODO indices must match
			Object exemplar = exemplars[i];	//TODO indices must match

			if (AppContext.config_skip_distance_when_exemplar_matches_query && exemplar == query) {
				return i;
			}
							
			dist = this.distance(query, exemplar);
			
			if (dist < bsf) {
				bsf = dist;
				closest_nodes.clear();
				closest_nodes.add(i);
			}else if (dist == bsf) {
//				if (distance == min_distance) {
//					System.out.println("min distances are same " + distance + ":" + min_distance);
//				}
				bsf = dist;
				closest_nodes.add(i);
			}
		}
		
		int r = AppContext.getRand().nextInt(closest_nodes.size());
		return closest_nodes.get(r);
	}
	
	

}
