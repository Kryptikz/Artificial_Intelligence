package Test1;

import org.jblas.*;

public class ConvolutionalLayer {
	DoubleMatrix[] kernels;
	int depth;
	int padding;
	int stride;
	public ConvolutionalLayer(int depth, int padding, int stride) {
		this.depth = depth;
		this.padding = padding;
		this.stride = stride;
	}
	public DoubleMatrix feedForward(DoubleMatrix x) {
		//add padding
		return null;
	}
	private DoubleMatrix padMatrix(DoubleMatrix x) {
		//DoubleMatrix ret =
		return null;
	}
	
}
