package Test1;

import org.jblas.*;

public class CompareWithC {
	public static void main(String[] args) {
		//this class&method exists to compare the working network model in this program to the errors in a more complicated one
		int[] layers = new int[] {2,5,6,2};
		Network n = new Network(layers, 0.1, new Sigmoid());
		DoubleMatrix[] weights = new DoubleMatrix[3];
		DoubleMatrix[] biases = new DoubleMatrix[3];
		double[][] wMat1 = new double[][] {{0.2,0.3},{0.4,0.5},{0.6,0.7},{0.8,0.9},{1.0,1.1}};
		double[][] wMat2 = new double[][] {{1.2,1.3,1.4,1.5,1.6},{1.7,1.8,1.9,2.0,2.1},{2.2,2.3,2.4,2.5,2.6},{2.7,2.8,2.9,3.0,3.1},{3.2,3.3,3.4,3.5,3.6},{3.7,3.8,3.9,4.0,4.1}};
		double[][] wMat3 = new double[][] {{4.2,4.3,4.4,4.5,4.6,4.7},{4.8,4.9,5.0,5.1,5.2,5.3}};
		double[] bMat1 = new double[] {0.2,-0.4,0.6,-0.8,1.0};
		double[] bMat2 = new double[] {-1.2,1.4,-1.6,1.8,-2.0,2.2}; 
		double[] bMat3 = new double[] {-2.4,2.6};
		weights[0] = new DoubleMatrix(wMat1);
		weights[1] = new DoubleMatrix(wMat2);
		weights[2] = new DoubleMatrix(wMat3);
		biases[0] = new DoubleMatrix(bMat1);
		biases[1] = new DoubleMatrix(bMat2);
		biases[2] = new DoubleMatrix(bMat3);
		n.setWMatrix(weights);
		n.setBMatrix(biases);
		DoubleMatrix x = new DoubleMatrix(new double[] {0.1,0.8});
		DoubleMatrix y = new DoubleMatrix(new double[] {0,1});
		n.backProp(x, y);
		System.out.println();
		Network.printMatrix(n.feedForward(new DoubleMatrix(new double[] {0.2,0.5})));
		x = new DoubleMatrix(new double[] {0.7,0.4});
		y = new DoubleMatrix(new double[] {1,0});
		n.backProp(x, y);
		System.out.println();
		Network.printMatrix(n.feedForward(new DoubleMatrix(new double[] {0.6,0.2})));
		
	}
}
