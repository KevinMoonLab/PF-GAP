package distance.elastic;

import org.apache.commons.lang3.ArrayUtils;
import java.io.*;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Scanner;

public class PythonDistance implements Serializable{
    public PythonDistance(){

    }

    public static double distance(double[] t1, double[] t2, String pythondistfile) throws IOException, InterruptedException {
        // first, we build a maple file that:
        //1. reads the actual script file
        //2. calls the distance function
        Integer key = (int)(Math.random()*1000000);
        String tempdir = "_temp" + key; //for parallelization, we want to prevent accidentally using the wrong distance.
        String tempname = tempdir + "/_temp_helper.txt";
        String tempMname = tempdir + "/_temp_helper.py";

        String pythonCommands =
                "import sys \n" +
                        //"import os\n" +
                        "from pathlib import Path\n" +
                        "parent_dir = Path(__file__).resolve().parent.parent\n" +
                        "sys.path.insert(0, str(parent_dir))\n" +
                //"current_directory = os.getcwd()\n" +
                        //"sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) \n" +
                "import " + pythondistfile.substring(0,14) + " \n" +
                "t1 = " + Arrays.toString(t1).replace("{","[").replace("}","]") + "\n" +
                "t2 = " + Arrays.toString(t2).replace("{","[").replace("}","]") + "\n" +
                "theDir = \"" + tempdir + "\"\n" +
                        //pythondistfile.substring(0,14) + ".Distance(t1,t2,theDir);\n"; //The python function must be called "Distance"
                pythondistfile.substring(0,14) + ".Distance(t1,t2,theDir);\n" + //The python function must be called "Distance"
                        //"os.chdir(current_directory)";
                "sys.path.remove(str(parent_dir))";

        Process fileprocess = Runtime.getRuntime().exec(new String[]{"/bin/bash", "-c", "mkdir " + tempdir});
        fileprocess.waitFor();
        try (FileWriter writer = new FileWriter(tempname)) {
            writer.write(pythonCommands);
            //System.out.println("Python script '" + tempname + "' created successfully.");
        } catch (IOException e) {
            System.err.println("Error writing to file: " + e.getMessage());
        }

        File oldfile = new File(tempname);
        oldfile.renameTo(new File(tempMname));

        Process process = Runtime.getRuntime().exec("python3 " + tempMname);
        process.waitFor();
        //oldfile.delete();
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
            //file.delete();
        } catch (FileNotFoundException e) {
            System.err.println("Distance answer file not found: " + fileName);
        }

        // clean up the temp files...
        Runtime.getRuntime().exec(new String[]{"/bin/bash", "-c", "rm -r " + tempdir});
        //System.out.println(distance_answer.get(0));
        return distance_answer.get(0);
    }
}