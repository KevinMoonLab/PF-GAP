package distance;

import java.io.IOException;
import java.io.Serializable;
//import java.util.ArrayList;
//import java.util.List;
import java.util.Random;
//import java.util.Arrays;
import java.util.Objects;

import core.AppContext;
import datasets.readers.lazy.LazySeriesRef;
import core.contracts.ObjectDataset;
import distance.api.DistanceFunction;
import distance.api.LazyDistanceFunction;
import distance.elastic.*;
import distance.graph.*;
import distance.interop.*;
import distance.meta.*;
import distance.missing.*;
import distance.multiTS.*;

public class DistanceMeasure implements Serializable {
	
	public final MEASURE distance_measure;
	private final String[] descriptors;

	public String[] getDescriptors() {
		return descriptors.clone();
	}

	public boolean hasDescriptors() {
		return descriptors.length > 0;
	}

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
	private Cosine cosine;
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

	// missing-compatible distances
	private NaNEuclidean nan_euclidean;
	private NaNEuclidean_I nan_euclidean_i;
	private DTWAROW dtwarow;
	private DTWAROW_I dtwarow_i;
	private DTWAROW_D dtwarow_d;

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

	//public DistanceMeasure (MEASURE m, String... descriptor) throws Exception{
		//this.distance_measure = m;
		//initialize(m, descriptor);
	//}

	public DistanceMeasure(
			MEASURE measure,
			String... descriptor
	) throws Exception {
		this.distance_measure =
				Objects.requireNonNull(
						measure,
						"DistanceMeasure requires a non-null measure."
				);

		this.descriptors =
				descriptor == null
						? new String[0]
						: descriptor.clone();

		initialize(
				measure,
				this.descriptors
		);
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
			case cosine:
				cosine = new Cosine();
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
			case nan_euclidean_i:
				nan_euclidean_i = new NaNEuclidean_I();
				break;
			case nan_euclidean:
				nan_euclidean = new NaNEuclidean();
				break;
			case dtwarow:
				dtwarow = new DTWAROW();
				break;
			case dtwarow_i:
				dtwarow_i = new DTWAROW_I();
				break;
			case dtwarow_d:
				dtwarow_d = new DTWAROW_D();
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
			case cosine:
				return cosine;
			case shapeHoG1dDTW:
				return shapeHoG1dDTW;
			case dtw_i:
				return dtw_i;
			case dtw_d:
				return dtw_d;
			case nan_euclidean:
				return nan_euclidean;
			case nan_euclidean_i:
				return nan_euclidean_i;
			case dtwarow:
				return dtwarow;
			case dtwarow_i:
				return dtwarow_i;
			case dtwarow_d:
				return dtwarow_d;
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
		// sometimes we can't get random parameters on a lazy dataset
		// so we need a safe fallback in case.
		/*if (AppContext.isLazyDataset) {
			selectLazyCompatibleParams(r);
			return;
		}*/

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
			this.windowSizeDTW = -1; //d.length();
			break;
		case dtwcv:
		case shifazDTWCV:
			this.windowSizeDTW = dtwcv.get_random_window(d, r);
			break;
		case ddtw:
		case shifazDDTW:
			this.windowSizeDDTW = -1; //d.length();
			break;
		case shapeHoG1dDTW:
			this.windowSizeDDTW = -1; //d.length();
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
				this.windowSizeDDTW = -1; //d.length();
				break;
		default:
//			throw new Exception("Unknown distance measure");
//			break;
		}
	}

	private void selectLazyCompatibleParams(
			Random r
	) {
		switch (this.distance_measure) {
			case euclidean:
			case shifazEUCLIDEAN:
			case dtw:
			case shifazDTW:
			case ddtw:
			case shifazDDTW:
			case dtw_i:
			case dtw_d:
			case ddtw_i:
			case shifazDDTW_I:
			case ddtw_d:
			case shapeHoG1dDTW:
			case shapeHoGdtw:
			case shifazShapeHoGDTW:
			case shapeHoGdtw_d:
				/*
				 * These use no data-derived random parameter in the current
				 * full-window configuration.
				 */
				windowSizeDTW = -1;
				windowSizeDDTW = -1;
				break;

			default:
				throw new UnsupportedOperationException(
						"Distance measure "
								+ distance_measure
								+ " currently requires parameter selection from "
								+ "materialized dataset data and is not yet supported "
								+ "with lazy datasets."
				);
		}
	}

	//public double distance(Object s, Object t) throws IOException, InterruptedException {
	//	return this.distance(s, t, Double.POSITIVE_INFINITY);
	//}

	/**
	 * Computes a distance between stored or materialized series.
	 *
	 * <p>Lazy references are resolved independently. Eager objects pass through
	 * unchanged.</p>
	 */
	public double distance(
			Object s,
			Object t
	) throws IOException, InterruptedException {
		return distance(
				s,
				t,
				Double.POSITIVE_INFINITY
		);
	}

	/**
	 * Computes a distance between stored or materialized series.
	 *
	 * <p>This compatibility entry point resolves ordinary lazy references and
	 * then delegates to distanceResolved(). Custom Java distances implementing
	 * LazyDistanceFunction retain control over their own resolution behavior.</p>
	 */
	public double distance(
			Object first,
			Object second,
			double bestSoFar
	) throws IOException, InterruptedException {

		if (first == null || second == null) {
			throw new IllegalArgumentException(
					"Distance inputs cannot be null."
			);
		}

		/*
		 * Preserve the existing LazyDistanceFunction contract. Such a custom
		 * Java distance receives the original stored representations and decides
		 * when to resolve them.
		 */
		if (distance_measure == MEASURE.javadistance
				&& distanceFunction
				instanceof LazyDistanceFunction lazyDistance) {

			return lazyDistance.compute(
					first,
					second,
					AppContext::readLazySeries
			);
		}

		Object resolvedFirst =
				resolveSeries(
						first
				);

		Object resolvedSecond =
				resolveSeries(
						second
				);

		return distanceResolved(
				resolvedFirst,
				resolvedSecond,
				bestSoFar
		);
	}

	/**
	 * Computes a distance between already materialized series.
	 *
	 * <p>This method never invokes a lazy reader. It is intended for hot loops
	 * where the caller has deliberately chosen the resolution lifetime, such as
	 * one query-to-exemplars comparison or one candidate split.</p>
	 */
	public double distanceResolved(
			Object s,
			Object t
	) throws IOException, InterruptedException {
		return distanceResolved(
				s,
				t,
				Double.POSITIVE_INFINITY
		);
	}

	/**
	 * Computes a distance between already materialized series.
	 *
	 * @param s materialized first series
	 * @param t materialized second series
	 * @param bsf current best distance for early abandoning
	 * @return distance value
	 */
	public double distanceResolved(
			Object s,
			Object t,
			double bsf
	) throws IOException, InterruptedException {

		if (s == null || t == null) {
			throw new IllegalArgumentException(
					"Resolved distance inputs cannot be null."
			);
		}

		if (s instanceof LazySeriesRef
				|| t instanceof LazySeriesRef) {

			throw new IllegalArgumentException(
					"distanceResolved() received a LazySeriesRef. "
							+ "Resolve the inputs with resolveSeries() first."
			);
		}

		double distance =
				Double.POSITIVE_INFINITY;

		switch (distance_measure) {
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
			//distance = dtw.distance(s, t, bsf, ((double[]) s).length);
			distance = dtw.distance(s, t, bsf, resolveWindowSize(s, t, this.windowSizeDTW));
			break;
		case dtwcv:
		case shifazDTWCV:
			distance = 	dtwcv.distance(s, t, bsf, this.windowSizeDTW);
			break;
		case ddtw:
		case shifazDDTW:
			//distance = ddtw.distance(s, t, bsf, ((double[]) s).length);
			distance = ddtw.distance(s, t, bsf, resolveWindowSize(s, t, this.windowSizeDDTW));
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
		/*case javadistance:
			//distance = distanceFunction.compute(s,t);
			distance = computeJavaDistance(s,t);
			break;*/
		case javadistance:
			if (distanceFunction == null) {
				throw new IllegalStateException(
						"DistanceFunction is null for javadistance measure."
				);
			}

			/*
			 * distanceResolved() guarantees that both inputs have already been
			 * materialized. Do not ask a LazyDistanceFunction to resolve them again.
			 */
			distance =
					distanceFunction.compute(s,t);

			break;
		case manhattan:
			distance = manhattan.distance(s,t,bsf);
			break;
		case cosine:
			distance = cosine.distance(s,t,bsf);
			break;
		case shapeHoG1dDTW:
			distance = shapeHoG1dDTW.distance(s,t,bsf,((double[]) s).length);
			break;
		case nan_euclidean:
			distance = nan_euclidean.distance(s,t,bsf);
			break;
		case nan_euclidean_i:
			distance = nan_euclidean_i.distance(s,t,bsf);
			break;
		case dtwarow:
			distance = dtwarow.distance(s,t,bsf);
			break;
		case dtwarow_i:
			distance = dtwarow_i.distance(s,t,bsf);
			break;
		case dtwarow_d:
			distance = dtwarow_d.distance(s,t,bsf);
			break;
		case dtw_i:
			//distance = dtw_i.distance(s,t,bsf,((double[][]) s).length);
			distance = dtw_i.distance(s,t,bsf, resolveWindowSize(s,t,this.windowSizeDTW));
			break;
		case dtw_d:
			//distance = dtw_d.distance(s,t,bsf,((double[][]) s).length);
			distance = dtw_d.distance(s,t,bsf, resolveWindowSize(s,t,this.windowSizeDTW));
			break;
		case ddtw_i:
		case shifazDDTW_I:
			//distance = ddtw_i.distance(s, t, bsf, ((double[][]) s).length);
			distance = ddtw_i.distance(s,t,bsf, resolveWindowSize(s,t,this.windowSizeDDTW));
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
			//distance = shapeHoGdtw.distance(s, t, bsf, ((double[][]) s).length);
			distance = shapeHoGdtw.distance(s,t,bsf, resolveWindowSize(s,t,this.windowSizeDTW));
			break;

		case ddtw_d:
			//distance = ddtw_d.distance(s, t, bsf, ((double[][]) s).length);
			distance = ddtw_d.distance(s,t,bsf, resolveWindowSize(s,t,this.windowSizeDDTW));
			break;
		case wdtw_d:
			distance = wdtw_d.distance(s, t, bsf, this.weightWDTW);
			break;
		case wddtw_d:
			distance = wddtw_d.distance(s, t, bsf, this.weightWDDTW);
			break;
		case shapeHoGdtw_d:
			//distance = shapeHoGdtw_d.distance(s, t, bsf, ((double[][]) s).length);
			distance = shapeHoGdtw_d.distance(s,t,bsf, resolveWindowSize(s,t,this.windowSizeDDTW));
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
//		throw new Exception("Unknown distance measure");
//		break;
		}
		if (Double.isNaN(distance)) {
			throw new IllegalStateException(
					"Distance measure "
							+ distance_measure
							+ " returned NaN."
			);
		}

		/*
		 * Positive infinity may be a legitimate early-abandoning result when
		 * bestSoFar is finite. Do not print from the distance hot path.
		 */
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
	//List<Integer> closest_nodes = new ArrayList<Integer>();
	
	//public int find_closest_node(
	//		double[] query,
	//		double[][] exemplars,
	//		boolean train,
	//		String... dfile) throws Exception{
	/**
	 * Compatibility nearest-node method for stored or materialized inputs.
	 *
	 * <p>The query is resolved once and every exemplar is resolved once before
	 * the comparison loop begins.</p>
	 */
	public int find_closest_node(
			Object query,
			Object[] exemplars,
			boolean train,
			String... distanceFiles
	) throws Exception {

		Object resolvedQuery =
				resolveSeries(
						query
				);

		Object[] resolvedExemplars =
				resolveSeriesArray(
						exemplars
				);

		return findClosestResolvedNode(
				resolvedQuery,
				resolvedExemplars,
				AppContext.getRand()
		);
	}

	/**
	 * Finds the closest exemplar when every input has already been materialized.
	 *
	 * <p>All mutable nearest-node state is method-local, so separate calls may
	 * execute concurrently when they use independent DistanceMeasure instances.
	 * The current best distance is passed through to compatible distances for
	 * early abandoning.</p>
	 */
	public int findClosestResolvedNode(
			Object resolvedQuery,
			Object[] resolvedExemplars,
			Random random
	) throws IOException, InterruptedException {

		if (resolvedQuery == null) {
			throw new IllegalArgumentException(
					"Resolved query cannot be null."
			);
		}

		if (resolvedQuery instanceof LazySeriesRef) {
			throw new IllegalArgumentException(
					"Resolved query cannot be a LazySeriesRef."
			);
		}

		if (resolvedExemplars == null
				|| resolvedExemplars.length == 0) {

			throw new IllegalArgumentException(
					"At least one resolved exemplar is required."
			);
		}

		Objects.requireNonNull(
				random,
				"Nearest-node selection requires a Random instance."
		);

		double bestDistance =
				Double.POSITIVE_INFINITY;

		int[] tiedBranches =
				new int[resolvedExemplars.length];

		int tieCount =
				0;

		for (int branch = 0;
			 branch < resolvedExemplars.length;
			 branch++) {

			Object exemplar =
					resolvedExemplars[branch];

			if (exemplar == null) {
				throw new IllegalArgumentException(
						"Resolved exemplar is null at branch "
								+ branch
								+ "."
				);
			}

			if (exemplar instanceof LazySeriesRef) {
				throw new IllegalArgumentException(
						"Resolved exemplar is still a LazySeriesRef at branch "
								+ branch
								+ "."
				);
			}

			if (AppContext
					.config_skip_distance_when_exemplar_matches_query
					&& exemplar == resolvedQuery) {

				return branch;
			}

			double currentDistance =
					distanceResolved(
							resolvedQuery,
							exemplar,
							bestDistance
					);

			if (currentDistance < bestDistance) {
				bestDistance =
						currentDistance;

				tiedBranches[0] =
						branch;

				tieCount =
						1;
			} else if (Double.compare(
					currentDistance,
					bestDistance
			) == 0) {
				tiedBranches[tieCount++] =
						branch;
			}
		}

		if (tieCount == 0) {
			throw new IllegalStateException(
					"No closest branch was found for distance measure "
							+ distance_measure
							+ "."
			);
		}

		if (tieCount == 1) {
			return tiedBranches[0];
		}

		return tiedBranches[
				random.nextInt(
						tieCount
				)
				];
	}

	// for lazy datasets: resolve before passing to distances
	/*private boolean isLazyObject(Object obj) {
		return obj instanceof LazySeriesRef;
	}

	private Object resolveIfLazy(Object obj) {
		if (obj instanceof LazySeriesRef ref) {

			//return AppContext.lazySeriesReader.read(ref);
			return AppContext
					.getLazySeriesReader(ref.getReaderKey())
					.read(ref);
		}

		return obj;
	}*/

	/**
	 * Resolves one stored series representation.
	 *
	 * <p>A LazySeriesRef is materialized through its registered reader. An eager
	 * object is returned unchanged. This supports all mixed combinations:</p>
	 *
	 * <pre>
	 * eager query + eager exemplar
	 * lazy query  + eager exemplar
	 * eager query + lazy exemplar
	 * lazy query  + lazy exemplar
	 * </pre>
	 *
	 * @param series stored or already materialized series
	 * @return materialized series
	 */
	public Object resolveSeries(
			Object series
	) {
		if (series == null) {
			throw new IllegalArgumentException(
					"Cannot resolve a null series."
			);
		}

		if (series instanceof LazySeriesRef reference) {
			return AppContext
					.getLazySeriesReader(
							reference.getReaderKey()
					)
					.read(
							reference
					);
		}

		return series;
	}

	/**
	 * Resolves every element of a stored exemplar array exactly once.
	 *
	 * @param series stored or materialized series representations
	 * @return newly allocated array containing materialized series
	 */
	public Object[] resolveSeriesArray(
			Object[] series
	) {
		if (series == null) {
			throw new IllegalArgumentException(
					"Cannot resolve a null series array."
			);
		}

		Object[] resolved =
				new Object[series.length];

		for (int index = 0;
			 index < series.length;
			 index++) {

			resolved[index] =
					resolveSeries(
							series[index]
					);
		}

		return resolved;
	}

	/**
	 * Returns whether the supplied object is a lazy series reference.
	 */
	public static boolean isLazySeries(
			Object series
	) {
		return series instanceof LazySeriesRef;
	}

	/*private double computeJavaDistance(
			Object first,
			Object second
	) {
		if (distanceFunction == null) {
			throw new IllegalStateException(
					"DistanceFunction is null for javadistance measure."
			);
		}

		if (distanceFunction
				instanceof LazyDistanceFunction lazyDistance) {

			return lazyDistance.compute(
					first,
					second,
					AppContext::readLazySeries
			);
		}

		Object resolvedFirst =
				resolveSeries(
						first
				);

		Object resolvedSecond =
				resolveSeries(
						second
				);

		return distanceFunction.compute(
				resolvedFirst,
				resolvedSecond
		);
	}*/


	private int timeLengthOf(Object series) {
		if (series instanceof double[] x) {
			return x.length;
		}

		if (series instanceof Double[] x) {
			return x.length;
		}

		if (series instanceof double[][] x) {
			return x.length == 0 ? 0 : x[0].length;
		}

		if (series instanceof Double[][] x) {
			return x.length == 0 ? 0 : x[0].length;
		}

		if (series instanceof Object[][] x) {
			return x.length == 0 ? 0 : x[0].length;
		}

		if (series instanceof Object[] x) {
			return x.length;
		}

		throw new IllegalArgumentException(
				"Cannot infer time length from series type: "
						+ series.getClass().getName()
		);
	}

	private int dimensionCountOf(Object series) {
		if (series instanceof double[][] x) {
			return x.length;
		}

		if (series instanceof Double[][] x) {
			return x.length;
		}

		if (series instanceof Object[][] x) {
			return x.length;
		}

		return 1;
	}

	private int fullWindow(Object s, Object t) {
		return Math.max(
				timeLengthOf(s),
				timeLengthOf(t)
		);
	}

	private int resolveWindowSize(
			Object s,
			Object t,
			int configuredWindow
	) {
		if (configuredWindow > 0) {
			return configuredWindow;
		}

		return -1;
	}

	/**
	 * Creates an independent worker-local evaluator with the same selected
	 * parameters.
	 *
	 * <p>The new DistanceMeasure reconstructs its concrete implementation from
	 * the original measure and descriptors. Selected candidate parameters are
	 * copied rather than randomized again.</p>
	 */
	public DistanceMeasure copyForEvaluation()
			throws Exception {

		DistanceMeasure copy =
				new DistanceMeasure(
						distance_measure,
						descriptors
				);

		copy.windowSizeDTW =
				windowSizeDTW;

		copy.windowSizeDDTW =
				windowSizeDDTW;

		copy.windowSizeLCSS =
				windowSizeLCSS;

		copy.windowSizeERP =
				windowSizeERP;

		copy.epsilonLCSS =
				epsilonLCSS;

		copy.gERP =
				gERP;

		copy.nuTWE =
				nuTWE;

		copy.lambdaTWE =
				lambdaTWE;

		copy.cMSM =
				cMSM;

		copy.weightWDTW =
				weightWDTW;

		copy.weightWDDTW =
				weightWDDTW;

		return copy;
	}
	
	

}
