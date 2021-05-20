package Test1;

import org.jblas.*;

public abstract class ActivationFunction {
	protected String type;
	public ActivationFunction(String type) {
		this.type = type;
	}
	public String getType() {
		return type;
	}
	public boolean isType(String otype) {
		return type.equals(otype);
	}
	public DoubleMatrix evalMatrix(DoubleMatrix x) {
		validateInputMatrix(x);
		DoubleMatrix y = new DoubleMatrix(x.getRows(),1);
		for(int i=0;i<y.getRows();i++) {
			y.put(i,0,eval(x.get(i,0)));
		}
		return y;
	}
	public DoubleMatrix evalPrimeMatrix(DoubleMatrix x) {
		validateInputMatrix(x);
		DoubleMatrix y = new DoubleMatrix(x.getRows(),1);
		for(int i=0;i<y.getRows();i++) {
			y.put(i,0,evalPrime(x.get(i,0)));
		}
		return y;
	}
	private static boolean validateInputMatrix(DoubleMatrix x) {
		if (x.getColumns() != 1) {
			System.err.println("WRONG NUMBER OF COLUMNS IN SIGMOID MATRIX COMPUTATION, SHOULD BE 1, HAS " + x.getColumns());
			return false;
		}
		return true;
	}
	public abstract double eval(double x);
	public abstract double evalPrime(double x);
}