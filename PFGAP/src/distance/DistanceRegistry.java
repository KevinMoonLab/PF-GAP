package distance;

import imputation.MaskedDistance;

import java.util.HashMap;
import java.util.Map;

public class DistanceRegistry {

    private static final Map<String, MEASURE> registry = new HashMap<>();

    static {
        // univariate time series--mostly came with the original PF implementation
        registry.put("basicDTW", MEASURE.basicDTW);
        registry.put("dtwDistance", MEASURE.dtwDistance);
        registry.put("dtwDistanceEfficient", MEASURE.dtwDistanceEfficient);
        registry.put("erp", MEASURE.erp);
        registry.put("lcss", MEASURE.lcss);
        registry.put("msm", MEASURE.msm);
        registry.put("pdtw", MEASURE.pdtw);
        registry.put("scdtw", MEASURE.scdtw);
        registry.put("twe", MEASURE.twe);
        registry.put("wdtw", MEASURE.wdtw);
        registry.put("francoisDTW", MEASURE.francoisDTW);
        registry.put("smoothDTW", MEASURE.smoothDTW);
        registry.put("dtw", MEASURE.dtw);
        registry.put("dca_dtw", MEASURE.dca_dtw);
        registry.put("equality", MEASURE.equality);
        registry.put("dtwcv", MEASURE.dtwcv);
        registry.put("ddtwcv", MEASURE.ddtwcv);
        registry.put("wddtw", MEASURE.wddtw);
        registry.put("ddtw", MEASURE.ddtw);
        registry.put("shifazDTW", MEASURE.shifazDTW);
        registry.put("shifazDTWCV", MEASURE.shifazDTWCV);
        registry.put("shifazDDTW", MEASURE.shifazDDTW);
        registry.put("shifazDDTWCV", MEASURE.shifazDDTWCV);
        registry.put("shifazWDTW", MEASURE.shifazWDTW);
        registry.put("shifazWDDTW", MEASURE.shifazWDDTW);
        registry.put("shifazERP", MEASURE.shifazERP);
        registry.put("shifazMSM", MEASURE.shifazMSM);
        registry.put("shifazLCSS", MEASURE.shifazLCSS);
        registry.put("shifazTWE", MEASURE.shifazTWE);
        registry.put("shapeHoG1dDTW", MEASURE.shapeHoG1dDTW);

        // Inter-op
        registry.put("maple", MEASURE.maple);
        registry.put("python", MEASURE.python);
        registry.put("javadistance", MEASURE.javadistance);

        // Designed for tabular numeric (also for time series)
        registry.put("euclidean", MEASURE.euclidean);
        registry.put("manhattan", MEASURE.manhattan);
        registry.put("shifazEUCLIDEAN", MEASURE.shifazEUCLIDEAN);

        // Multivariate Independent (_I)
        registry.put("dtw_i", MEASURE.dtw_i);
        registry.put("ddtw_i", MEASURE.ddtw_i);
        registry.put("shifazDDTW_I", MEASURE.shifazDDTW_I);
        registry.put("wdtw_i", MEASURE.wdtw_i);
        registry.put("shifazWDTW_I", MEASURE.shifazWDTW_I);
        registry.put("wddtw_i", MEASURE.wddtw_i);
        registry.put("shifazWDDTW_I", MEASURE.shifazWDDTW_I);
        registry.put("twe_i", MEASURE.twe_i);
        registry.put("shifazTWE_I", MEASURE.shifazTWE_I);
        registry.put("erp_i", MEASURE.erp_i);
        registry.put("shifazERP_I", MEASURE.shifazERP_I);
        registry.put("euclidean_i", MEASURE.euclidean_i);
        registry.put("shifazEUCLIDEAN_I", MEASURE.shifazEUCLIDEAN_I);
        registry.put("lcss_i", MEASURE.lcss_i);
        registry.put("shifazLCSS_I", MEASURE.shifazLCSS_I);
        registry.put("msm_i", MEASURE.msm_i);
        registry.put("shifazMSM_I", MEASURE.shifazMSM_I);
        registry.put("manhattan_i", MEASURE.manhattan_i);
        registry.put("shifazMANHATTAN_I", MEASURE.shifazMANHATTAN_I);
        registry.put("cid_i", MEASURE.cid_i);
        registry.put("shifazCID_I", MEASURE.shifazCID_I);
        registry.put("sbd_i", MEASURE.sbd_i);
        registry.put("shifazSBD_I", MEASURE.shifazSBD_I);

        // Multivariate Dependent (_D)
        registry.put("dtw_d", MEASURE.dtw_d);
        registry.put("ddtw_d", MEASURE.ddtw_d);
        registry.put("wdtw_d", MEASURE.wdtw_d);
        registry.put("wddtw_d", MEASURE.wddtw_d);
        registry.put("shapeHoGdtw_d", MEASURE.shapeHoGdtw_d);
        //measuresByName.put("euclidean_d", MEASURE.euclidean_d);
        //measuresByName.put("manhattan_d", MEASURE.manhattan_d);

        // Shape-based multivariate DTW
        registry.put("shapeHoGdtw", MEASURE.shapeHoGdtw);
        registry.put("shifazShapeHoGDTW", MEASURE.shifazShapeHoGDTW);
    }

    public static MEASURE get(String name) {
        return registry.get(name);
    }

    public static boolean contains(String name) {
        return registry.containsKey(name);
    }

    public static Map<String, MEASURE> getAll() {
        return registry;
    }


    public static MaskedDistance getMaskedDistance(MEASURE measure, String... descriptors) {
        try {
            DistanceMeasure dm = new DistanceMeasure(measure, descriptors);
            //dm.initialize(measure); // optionally pass descriptors if needed
            Object distanceInstance = dm.getDistanceInstance(measure);
            return new MaskedDistance(distanceInstance);
        } catch (Exception e) {
            throw new RuntimeException("Failed to initialize distance for measure: " + measure, e);
        }
    }


}