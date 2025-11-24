//package distance.elastic;
package distance;

import distance.graph.*;

public enum MEASURE {
	basicDTW, 
	dtwDistance, 
	dtwDistanceEfficient, 
	erp, lcss, msm, pdtw, scdtw, twe, wdtw, francoisDTW, smoothDTW, dtw, dca_dtw, euclidean, equality, dtwcv, ddtw, ddtwcv, wddtw
	,shifazDTW, shifazDTWCV, shifazDDTW, shifazDDTWCV, shifazWDTW, shifazWDDTW
	,shifazEUCLIDEAN, shifazERP, shifazMSM, shifazLCSS, shifazTWE
	,manhattan
	,shapeHoG1dDTW
	,cosine,

	// interop distance
	python, maple, javadistance,

	// Multivariate Independent (_I) distances
	dtw_i,
	ddtw_i, shifazDDTW_I,
	wdtw_i, shifazWDTW_I,
	wddtw_i, shifazWDDTW_I,
	twe_i, shifazTWE_I,
	erp_i, shifazERP_I,
	euclidean_i, shifazEUCLIDEAN_I,
	lcss_i, shifazLCSS_I,
	msm_i, shifazMSM_I,
	manhattan_i, shifazMANHATTAN_I,
	cid_i, shifazCID_I,
	sbd_i, shifazSBD_I,

// Multivariate Dependent (_D) distances
	dtw_d,
	ddtw_d,
	wdtw_d,
	wddtw_d,
	shapeHoGdtw_d,
//	euclidean_d,
//	manhattan_d,

// Shape-based multivariate DTW
	shapeHoGdtw, shifazShapeHoGDTW,

	// graph-based distances
	approximateGraphEditDistance,
	graphletDistance,
	graphEditDistance,
	hammingDistance,
	shortestPathDistance,
	wlDistance,
	wlDistance2,

	// meta distances - pretrained models
	meta_classmatch,
	meta_file_classmatch,
	meta_regression,
	meta_file_regression,
}
