package Test1;

import java.io.*;
import java.util.zip.*;
import java.lang.Byte;
import org.jblas.*;

public class MNIST_Test {
	public static void main(String[] args){
        InputStream img_in = null;
        InputStream label_in = null;
        File pictures = new File("images.bruh");
        File labels = new File("labels.bruh");
        try{
            img_in = new FileInputStream(pictures);
            label_in = new FileInputStream(labels);
        }
        catch(Exception e){
            e.printStackTrace();
        }
        byte[] buffer = new byte[784*20];
        byte[] buffer_label = new byte[20];
        try{
            img_in.read(buffer, 0, 16);
            label_in.read(buffer, 0, 8);
        }
        catch(Exception e){
            e.printStackTrace();
        }
        int numRead;
        Network n = new Network(new int[] {784,32,10},0.01);
        int numBackProps = 0;
        int numCorrect = 0;
        int numTests = 0;
        try{
            while((numRead = img_in.read(buffer)) != -1){
                label_in.read(buffer_label);
                int count = 0;
                for(int i = 0; i < numRead/784; i++){
                	DoubleMatrix xMat = new DoubleMatrix(784,1);
                    int xMatInc = 0;
                	int[][] image = new int[28][28];
                    for(int x = 0; x < 28; x++){
                        for(int y = 0; y < 28; y++){
                        	xMat.put(xMatInc,0,Byte.toUnsignedInt(buffer[count]));
                            xMatInc++;
                            count++;
                        }
                    }
                    DoubleMatrix yMat = DoubleMatrix.zeros(10,1);
                    yMat.put(Byte.toUnsignedInt(buffer_label[i]),0,1);
                    if (numBackProps >= 1000) {
                    	numTests++;
	                    DoubleMatrix result = n.feedForward(xMat);
	                    int actual = Byte.toUnsignedInt(buffer_label[i]);
	                    int netGuess = getNetworkPrediction(result);
	                    boolean netCorrect = (actual == netGuess);
	                    if (netCorrect) {
	                    	numCorrect++;
	                    }
	                    double percentCorrect = (double)numCorrect/(double)numTests;
	                    System.out.println("Actual: " + actual + " Network Output: " + netGuess + " Correct: " + netCorrect + " Percent Correct: " + (100*percentCorrect) + "%");
                    }
                    n.backProp(xMat, yMat);
                    if (numBackProps%1000==0) {
                    	numCorrect = 0;
                    	numTests = 0;
                    }
                    numBackProps++;
                    //Thread.sleep(500);
                }
            }
        }
        catch(Exception e){
            e.printStackTrace();
        }   
    }
	private static int getNetworkPrediction(DoubleMatrix y) { 
		double largest = y.get(0,0);
		int highestIndex = 0;
		for(int i=1;i<10;i++) {
			if(y.get(i,0) > largest) {
				largest = y.get(i,0);
				highestIndex = i;
			}
		}
		return highestIndex;
	}
}
