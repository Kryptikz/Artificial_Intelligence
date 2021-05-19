package Test1;

public class Sigmoid extends ActivationFunction {
	public Sigmoid() {
		super("sigmoid");
	}
	public double eval(double x) {
		return (1.0)/(1.0+(Math.exp(-1*x)));
	}
	public double evalPrime(double x) {
		double temp = eval(x);
		return temp*(1-temp);
	}
}