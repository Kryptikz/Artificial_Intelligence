package Test1;

import org.jblas.*;

public class Network {
	public DoubleMatrix[] wMatrixArr;
	public DoubleMatrix[] bMatrixArr;
	protected ActivationFunction af;
	protected int[] layers;
	protected double lRate;
	private GPUMath gpu = new GPUMath();
	
	public Network(int[] layers, double learningRate, ActivationFunction af) {
		this.layers = layers;
		this.lRate = learningRate;
		this.af = af;
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
				x = af.evalMatrix((wMatrixArr[i].mmul(x)).add(bMatrixArr[i]));
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
		activations[0] = af.evalMatrix(nodes[0]);
		for(int i=1;i<layers.length-1;i++) {
			nodes[i] = (wMatrixArr[i].mmul(activations[i-1])).add(bMatrixArr[i]);
			activations[i] = af.evalMatrix(nodes[i]);
		}
		//calculate output error
		layerError[layerError.length-1] = (activations[activations.length-1].sub(y)).mul(af.evalPrimeMatrix(nodes[nodes.length-1]));
		//System.out.println(layerError[layerError.length-1].mean()); //prints loss
		//backward pass
		for(int i=layerError.length-2;i>=0;i--) {
			layerError[i] = (wMatrixArr[i+1].transpose().mmul(layerError[i+1])).mul(af.evalPrimeMatrix(nodes[i]));
		}
		//perform gradient descent
		wMatrixArr[0] = wMatrixArr[0].sub((layerError[0].mmul(x.transpose())).mul(lRate));
		bMatrixArr[0] = bMatrixArr[0].sub(layerError[0].mul(lRate));
		for(int i=1;i<wMatrixArr.length;i++) {
			wMatrixArr[i] = wMatrixArr[i].sub((layerError[i].mmul(activations[i-1].transpose())).mul(lRate));
			bMatrixArr[i] = bMatrixArr[i].sub(layerError[i].mul(lRate));
		}
	}
	public void backPropGPU(DoubleMatrix x, DoubleMatrix y) {
		//hard-coded using quadratic cost function
		DoubleMatrix[] nodes = new DoubleMatrix[layers.length-1];
		DoubleMatrix[] activations = new DoubleMatrix[layers.length-1];
		DoubleMatrix[] layerError = new DoubleMatrix[layers.length-1];
		//forward pass
		nodes[0] = gpu.mmulGPU(wMatrixArr[0],x).add(bMatrixArr[0]);
		activations[0] = af.evalMatrix(nodes[0]);
		for(int i=1;i<layers.length-1;i++) {
			nodes[i] = gpu.mmulGPU(wMatrixArr[i], activations[i-1]).add(bMatrixArr[i]);
			activations[i] = af.evalMatrix(nodes[i]);
		}
		//calculate output error
		layerError[layerError.length-1] = (activations[activations.length-1].sub(y)).mul(af.evalPrimeMatrix(nodes[nodes.length-1]));
		//System.out.println(layerError[layerError.length-1].mean()); //prints loss
		//backward pass
		for(int i=layerError.length-2;i>=0;i--) {
			layerError[i] = gpu.mmulGPU(wMatrixArr[i+1].transpose(),layerError[i+1]).mul(af.evalPrimeMatrix(nodes[i]));
		}
		//perform gradient descent
		System.out.printf("wMatrix size: (%d,%d) solution size: (%d,%d)",wMatrixArr[0].getRows(),wMatrixArr[0].getColumns(),(gpu.mmulGPU(layerError[0],x.transpose()).mul(lRate)).getRows(),(gpu.mmulGPU(layerError[0],x.transpose()).mul(lRate)).getColumns());
		wMatrixArr[0] = wMatrixArr[0].sub(gpu.mmulGPU(layerError[0],x.transpose()).mul(lRate));
		bMatrixArr[0] = bMatrixArr[0].sub(layerError[0].mul(lRate));
		for(int i=1;i<wMatrixArr.length;i++) {
			wMatrixArr[i] = wMatrixArr[i].sub(gpu.mmulGPU(layerError[i],activations[i-1].transpose()).mul(lRate));
			bMatrixArr[i] = bMatrixArr[i].sub(layerError[i].mul(lRate));
		}
	}
	public double getLearningRate() {
		return lRate;
	}
	public void setLearningRate(double lRate) {
		this.lRate = lRate;
	}
	protected static void randomizeMatrix(DoubleMatrix m, double low, double high) {
		for(int r=0;r<m.getRows();r++) {
			for(int c=0;c<m.getColumns();c++) {
				m.put(r,c,low+(Math.random()*(high-low)));
			}
		}
	}
}