package Test1;

import java.io.File;
import java.io.FileInputStream;
import java.io.InputStream;

import javax.swing.JFrame;
import javax.swing.WindowConstants;

import org.jblas.DoubleMatrix;

public class EMNIST_TEST {
	public static void main(String[] args){
        InputStream img_in = null;
        InputStream label_in = null;
        File pictures = new File("data/EMNIST/bymerge/train-images");
        File labels = new File("data/EMNIST/bymerge/train-labels");
        int training_rate_halfed = 1;
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
            label_in.read(buffer_label, 0, 8);
        }
        catch(Exception e){
            e.printStackTrace();
        }
        int numRead;
	        Network n = new Network(new int[] {784, 128, 64, 47},0.2, new Sigmoid());
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
                    for(int x = 0; x < 28; x++){
                        for(int y = 0; y < 28; y++){
                        	//xMat.put(xMatInc,0,Byte.toUnsignedInt(buffer[count]));
                        	if(Byte.toUnsignedInt(buffer[count]) > 50) {
                        		xMat.put(xMatInc,0,1);
                        	}
                        	else {
                        		xMat.put(xMatInc,0,0);
                        	}
                            xMatInc++;
                            count++;
                        }
                    }
                    DoubleMatrix yMat = DoubleMatrix.zeros(47,1);
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
                    }
                    n.backProp(xMat, yMat);
                    if (numBackProps%1000==0) {
                    	double percentCorrect = (double)numCorrect/(double)numTests;
                    	System.out.println("Percent Correct: "+100*percentCorrect);
                    	if(training_rate_halfed < 1 && percentCorrect > 68) {
                    		//n.setLearningRate(0.01);
                    		training_rate_halfed = 1;
                    	}
                    	else if(training_rate_halfed < 2 && percentCorrect > 80){
                    		//n.setLearningRate(0.005);
                    		training_rate_halfed = 2;
                    	}
                    	numCorrect = 0;
                    	numTests = 0;
                    }
                    numBackProps++;
                    //Thread.sleep(500);
                }
            }
            img_in.close();
            label_in.close();
            
            //TESTING PART WITH DISPLAY
            
            JFrame frame = new JFrame("images");
            Display screen = new Display();
            
            frame.add(screen);
            Mouse mouse = new Mouse(screen);
            frame.addMouseListener(mouse);
            frame.setBounds(0,0,570,700);
            frame.setDefaultCloseOperation(WindowConstants.EXIT_ON_CLOSE);
            frame.setVisible(true);
            pictures = new File("data/EMNIST/bymerge/test-images");
            labels = new File("data/EMNIST/bymerge/test-labels");
            img_in = new FileInputStream(pictures);
            label_in = new FileInputStream(labels);
            try {
            	img_in.read(buffer, 0, 16);
            	label_in.read(buffer_label, 0, 8);
            	numRead = 0;
            	while((numRead = img_in.read(buffer)) != -1){
                    label_in.read(buffer_label);
                    int count = 0;
                    for(int i = 0; i < numRead/784; i++){
                    	DoubleMatrix xMat = new DoubleMatrix(784,1);
                        int xMatInc = 0;
                        int[][] image = new int[28][28];
                        for(int x = 0; x < 28; x++){
                            for(int y = 0; y < 28; y++){
                                image[y][x] = Byte.toUnsignedInt(buffer[count]);
                                if(Byte.toUnsignedInt(buffer[count]) > 50) {
                            		xMat.put(xMatInc,0,1);
                            	}
                            	else {
                            		xMat.put(xMatInc,0,0);
                            	}
                                xMatInc++;
                                count++;
                            }
                        }
                        DoubleMatrix result = n.feedForward(xMat);
                        int netGuess = getNetworkPrediction(result);
                        screen.update_image(image, Byte.toUnsignedInt(buffer_label[i]), netGuess);
                        screen.draw();
                        while(screen.pause == true) {
                        	Thread.sleep(1);
                        }
                        screen.pause = true;
                    }
                }
            }
            catch(Exception e) {
            	e.printStackTrace();
            }
            img_in.close();
            label_in.close();
            
        }
        catch(Exception e){
            e.printStackTrace();
        }   
        
    }
	private static int getNetworkPrediction(DoubleMatrix y) { 
		double largest = y.get(0,0);
		int highestIndex = 0;
		for(int i=1;i<47;i++) {
			if(y.get(i,0) > largest) {
				largest = y.get(i,0);
				highestIndex = i;
			}
		}
		return highestIndex;
	}
	public static char getCharFromInt(int i) {
		char[] table = new char[] {'0','1','2','3','4','5','6','7','8','9','A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z','a','b','d','e','f','g','h','n','q','r','t'};
		return table[i];
	}
}
