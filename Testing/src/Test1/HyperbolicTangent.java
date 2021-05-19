package Test1;

public class HyperbolicTangent extends ActivationFunction {
	public HyperbolicTangent() {
		super("hyperbolicTangent");
	}
	public double eval(double x) {
		return (Math.exp(2*x)-1)/(Math.exp(2*x)+1);
	}
	public double evalPrime(double x) {
		double temp = eval(x);
		return temp*(1-temp);
	}
}
