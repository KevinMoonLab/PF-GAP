import core.CSVReader;
import datasets.ListDataset;
import distance.elastic.DistanceMeasure;
import distance.elastic.MEASURE;
import distance.elastic.MapleDistance;
import org.apache.commons.lang3.ArrayUtils;
import trees.ProximityForest;
import trees.ProximityTree;
//import java.util.concurrent.ThreadLocalRandom;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Scanner;

// This class was created for the sole purpose of debugging and is not needed for the application.
public class pfgap{

    public static void main(String[] args) throws Exception {

        //first, we'll get user input for the parameters of the model and dataset name.
        Scanner myObj = new Scanner(System.in);
        System.out.println("Enter name of Dataset: ");
        String dataset = myObj.nextLine();
        System.out.println("Dataset is: " + dataset);
        System.out.println("Enter the number of trees to use: ");
        String treenum = myObj.nextLine();
        System.out.println("The number of trees is: " + treenum);
        //System.out.println("Enter the name of the Distance to use: ");
        //String dist = myObj.nextLine();
        //System.out.println("The selected distance is: " + dist);

        int Treenum = Integer.parseInt(treenum);
        //Class<DistanceMeasure> distance = Class.forName(dist);
        //String input_distance = dist;


        //ProximityForest PF = new ProximityForest(Treenum);
        //List<MEASURE> myMeasures = new List<MEASURE>() {MEASURE.maple};
        MEASURE[] mything = {MEASURE.euclidean}; //{MEASURE.python}; //{MEASURE.maple};
        String[] myfile = {"DistS2.mpl"};
        //ProximityForest PF = new ProximityForest(Treenum,myfile);
        //ProximityForest PF = new ProximityForest(Treenum);
        ProximityForest PF = new ProximityForest(Treenum,mything);
        //ProximityForest PF = new ProximityForest(Treenum, MEASURE.maple);
        String train_file = System.getProperty("user.dir") + "/Data/" + dataset + "_TRAIN.csv";
        String test_file = System.getProperty("user.dir") + "/Data/" + dataset + "_TEST.csv";
        ListDataset train_data = CSVReader.readCSVToListDataset(train_file,false, true,"\t");
        // Here is the test set, if testing is desired.
        //String test_file = "/home/ben/Documents/SchoolGithub/Math_Dissertation/Data_Science/PFGAP/Data/GunPoint_TEST.csv";
        ListDataset test_data = CSVReader.readCSVToListDataset(test_file,false, true,"\t");
        System.out.println("Training...");
        double t1 = System.currentTimeMillis();
        PF.train(train_data);
        double t2 = System.currentTimeMillis();
        System.out.println("Training complete.");
        System.out.print("Training Time: ");
        System.out.println(t2-t1);
        // We may want to test later.
        System.out.println("Testing...");
        double t3 = System.currentTimeMillis();
        PF.test(test_data);
        double t4 = System.currentTimeMillis();
        System.out.println("Testing Complete.");
        System.out.print("Testing time: ");
        System.out.println(t4-t3);
        System.out.print("Accuracy Score: ");
        System.out.println(PF.getResultSet().accuracy);
        //System.out.println(ForestProximity(6,2, PF));
        System.out.println("Computing Forest Proximities...");
        double t5 = System.currentTimeMillis();
        Double[][] PFGAP = new Double[train_data.size()][train_data.size()];
        for (Integer i=0; i< train_data.size(); i++){
            for (Integer j=0; j< train_data.size(); j++){
                Double prox = ForestProximity(i,j,PF);
                PFGAP[i][j] = prox;
            }
        }
        double t6 = System.currentTimeMillis();
        System.out.println("Done Computing Forest Proximities.");
        //Process process = Runtime.getRuntime().exec("mkdir dummy_folder");
        //Double testMapleDist = MapleDistance.distance(new double[]{10.0,2.0},new double[]{20.0,4.0},"DistS2.mpl");
        //System.out.println(testMapleDist);
        System.out.print("Computation time: ");
        System.out.println(t6-t5);
        //System.out.println(ArrayUtils.toString(PFGAP));
        //Now it's time to put the results in a text file so that we can read them into python.
        PrintWriter writer = new PrintWriter("Results/Proximities/" + dataset + "_Proximities.txt", "UTF-8");
        writer.print(ArrayUtils.toString(PFGAP));
        writer.close();
        Integer[] ytrain = new Integer[train_data.size()];
        for(Integer i=0; i< train_data.size(); i++){
            ytrain[i] = train_data.get_class(i);
        }
        PrintWriter writer2 = new PrintWriter("Results/ytrain/" + dataset + "_ytrain.txt", "UTF-8");
        writer2.print(ArrayUtils.toString(ytrain));
        writer2.close();


    }

    public static ArrayList<ProximityTree> getSi(Integer i, ProximityForest pf){
        ArrayList<ProximityTree> Si = new ArrayList<ProximityTree>();
        ProximityTree[] trees = pf.getTrees();
        for(ProximityTree tree:trees){
            ArrayList<Integer> oob = tree.getRootNode().getOutOfBagIndices();
            if(oob.contains(i)){
                Si.add(tree);
            }
        }
        return Si;
    }

    //public static ProximityTree.Node getJiLeaf(Integer i, ProximityTree t){
    public static ArrayList<Integer> getJi(Integer i, ProximityTree t){
        ArrayList<Integer> Ji = new ArrayList<>();
        ArrayList<ProximityTree.Node> leaves = t.getLeaves();
        //System.out.println(leaves.size());
        for (ProximityTree.Node leaf : leaves){
            ArrayList<Integer> inbags = leaf.getInBagIndices();
            ArrayList<Integer> oob = leaf.getOutOfBagIndices();
            //System.out.println(oob);
            //System.out.println(i);
            if(oob.contains(i)){
                Ji = inbags;
                //System.out.print(Ji);
            }
        }
        return Ji;
    }

    public static Double ForestProximity(Integer i, Integer j, ProximityForest pf){
        ArrayList<ProximityTree> Si = getSi(i,pf);
        //Double[] terms = new Double[]{};
        ArrayList<Double> terms = new ArrayList<>();
        for (ProximityTree t : Si){
            Integer cj = t.getRootNode().getMultiplicities().get(j);
            ArrayList<Integer> Mi = getJi(i,t);
            if (Mi.contains(j)){
                double Cj = (double) cj;
                terms.add((Cj/Mi.size())/ Si.size());
            }
            else{
                terms.add((double) 0);
            }
        }
        double sum = 0;
        for (Double term : terms){
            sum += term;
        }
        return sum;
    }
}
