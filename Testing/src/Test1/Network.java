package Test1;

import org.jblas.*;

public class Network {
	public DoubleMatrix[] wMatrixArr;
	public DoubleMatrix[] bMatrixArr;
	protected String activationFunc = "sigmoid";
	protected int[] layers;
	protected double lRate;
	
	public Network(int[] layers, double learningRate) {
		this.layers = layers;
		this.lRate = learningRate;
		wMatrixArr = new DoubleMatrix[layers.length-1];
		bMatrixArr = new DoubleMatrix[layers.length-1];
		for(int i=0;i<layers.length-1;i++) {
			DoubleMatrix tempWMatrix = new DoubleMatrix(layers[i+1],layers[i]);
			DoubleMatrix tempBMatrix = new DoubleMatrix(layers[i+1],1);
			randomizeMatrix(tempWMatrix, -1, 1);
			randomizeMatrix(tempBMatrix, -1, 1);
			wMatrixArr[i] = tempWMatrix; 
			bMatrixArr[i] = tempBMatrix;
		}
	}
	public DoubleMatrix feedForward(DoubleMatrix x) {
		for(int i=0;i<layers.length-1;i++) {
				x = sigmoidMatrix((wMatrixArr[i].mmul(x)).add(bMatrixArr[i]));
		}
		return x;
	}
	public void backProp(DoubleMatrix x, DoubleMatrix y) {
		//hard-coded using quadratic cost function
		DoubleMatrix[] nodes = new DoubleMatrix[layers.length-1];
		DoubleMatrix[] activations = new DoubleMatrix[layers.length-1];
		DoubleMatrix[] layerError = new DoubleMatrix[layers.length-1];
		//forward pass
		nodes[0] = (wMatrixArr[0].mmul(x)).add(bMatrixArr[0]);
		activations[0] = sigmoidMatrix(nodes[0]);
		
		for(int i=1;i<layers.length-1;i++) {
			nodes[i] = (wMatrixArr[i].mmul(activations[i-1])).add(bMatrixArr[i]);
			activations[i] = sigmoidMatrix(nodes[i]);
		}
		//calculate output error
		layerError[layerError.length-1] = (activations[activations.length-1].sub(y)).mul(sigmoidPrimeMatrix(nodes[nodes.length-1]));
		//backward pass
		for(int i=layerError.length-2;i>=0;i--) {
			layerError[i] = (wMatrixArr[i+1].transpose().mmul(layerError[i+1])).mul(sigmoidPrimeMatrix(nodes[i]));
		}
		//perform gradient descent
		wMatrixArr[0] = wMatrixArr[0].sub((layerError[0].mmul(x.transpose())).mul(lRate));
		bMatrixArr[0] = bMatrixArr[0].sub(layerError[0].mul(lRate));
		for(int i=1;i<wMatrixArr.length;i++) {
			wMatrixArr[i] = wMatrixArr[i].sub((layerError[i].mmul(activations[i-1].transpose())).mul(lRate));
			bMatrixArr[i] = bMatrixArr[i].sub(layerError[i].mul(lRate));
		}
	}
	protected static void randomizeMatrix(DoubleMatrix m, double low, double high) {
		for(int r=0;r<m.getRows();r++) {
			for(int c=0;c<m.getColumns();c++) {
				m.put(r,c,low+(Math.random()*(high-low)));
			}
		}
	}
	protected static DoubleMatrix sigmoidMatrix(DoubleMatrix x) {
		if (x.getColumns() != 1) {
			System.err.println("WRONG NUMBER OF COLUMNS IN SIGMOID MATRIX COMPUTATION, SHOULD BE 1, HAS " + x.getColumns());
			return null;
		}
		DoubleMatrix y = new DoubleMatrix(x.getRows(),x.getColumns());
		for(int i=0;i<y.getRows();i++) {
			y.put(i,0,sigmoid(x.get(i,0)));
		}
		return y;
	}
	protected static DoubleMatrix sigmoidPrimeMatrix(DoubleMatrix x) {
		if (x.getColumns() != 1) {
			System.err.println("WRONG NUMBER OF COLUMNS IN SIGMOID MATRIX COMPUTATION, SHOULD BE 1, HAS " + x.getColumns());
			return null;
		}
		DoubleMatrix y = new DoubleMatrix(x.getRows(),x.getColumns());
		for(int i=0;i<y.getRows();i++) {
			y.put(i,0,sigmoidPrime(x.get(i,0)));
		}
		return y;
	}
	protected static double sigmoid(double x) {
		return (1.0)/(1.0+(Math.exp(-1*x)));
	}
	protected static double sigmoidPrime(double x) {
		double temp = sigmoid(x);
		return temp*(1-temp);
	}
}