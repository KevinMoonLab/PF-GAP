package distance.elastic;

import org.apache.commons.lang3.ArrayUtils;
import java.io.*;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Scanner;

public class MapleDistance implements Serializable {
    public MapleDistance() {

    }

    public static double distance(double[] t1, double[] t2, String mapledistfile) throws IOException, InterruptedException {
        // first, we build a maple file that:
            //1. reads the actual script file
            //2. calls the distance function
        //System.out.println("Calling Maple Distance...");
        Integer key = (int)(Math.random()*1000000);
        String tempdir = "_temp" + key; //for parallelization, we want to prevent accidentally using the wrong distance.
        String tempname = tempdir + "/_temp_helper.txt";
        String tempMname = tempdir + "/_temp_helper.mpl";

        String mapleCommands =
                "restart;\n" +
                        "read \"" + mapledistfile + "\";\n" +
                        "t1:=" + Arrays.toString(t1).replace("{","[").replace("}","]") + ";\n" +
                        "t2:=" + Arrays.toString(t2).replace("{","[").replace("}","]") + ";\n" +
                        "theDir:= \"" + tempdir + "\";\n" +
                        "Distance(t1,t2,theDir);\n"; //The maple procedure must be called "Distance"

        Process preprocess = Runtime.getRuntime().exec(new String[]{"/bin/bash", "-c", "mkdir " + tempdir});
        preprocess.waitFor();
        try (FileWriter writer = new FileWriter(tempname)) {
            writer.write(mapleCommands);
            //System.out.println("Maple script '" + tempname + "' created successfully.");
        } catch (IOException e) {
            System.err.println("Error writing to file: " + e.getMessage());
        }

        File oldfile = new File(tempname);
        oldfile.renameTo(new File(tempMname));

        Process process = Runtime.getRuntime().exec("maple " + tempMname);
        process.waitFor();
        // At this point, a new file has been created which stores the answer. Now we need to read it in.

        String fileName = tempdir + "/distanceanswer.txt"; // Assuming data.txt contains decimal numbers
        ArrayList<Double> distance_answer = new ArrayList<>();
        try {
            File file = new File(fileName);
            Scanner scanner = new Scanner(file);

            while (scanner.hasNextDouble()) {
                double dist = scanner.nextDouble();
                //System.out.println("Read: " + dist);
                distance_answer.add(dist); //there should only be one, actually.
            }
            scanner.close();
        } catch (FileNotFoundException e) {
            System.err.println("Distance answer file not found: " + fileName);
        }

        // clean up the temp files... I don't think we have to wait on this one.
        Runtime.getRuntime().exec(new String[]{"/bin/bash", "-c", "rm -r " + tempdir});
        //System.out.println(distance_answer.get(0));
        return distance_answer.get(0);
    }
}