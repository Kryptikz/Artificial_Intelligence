package Test1;

public class HyperbolicTangent extends ActivationFunction {
	public HyperbolicTangent() {
		super("hyperbolicTangent");
	}
	public double eval(double x) {
		double epx = Math.exp(x);
		double enx = Math.exp(-x);
		return (epx-enx)/(epx+enx);
	}
	public double evalPrime(double x) {
		double htx = eval(x);
		return 1-(htx*htx);
	}
}
