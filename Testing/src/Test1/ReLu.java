package Test1;

public class ReLu extends ActivationFunction {
	public ReLu() {
		super("ReLu");
	}
	public double eval(double x) {
		if (x>0) {
			return x;
		} else {
			return 0;
		}
	}
	public double evalPrime(double x) {
		if (x>0) {
			return 1;
		} else {
			return 0;
		}
	}
}
