package Test1;

public class GPUMath_test {
	public static void main(String[] args) {
		GPUMath gpu = new GPUMath();
		int arrSize = 2000000;
		int[] arrOne = new int[arrSize];
		int[] arrTwo = new int[arrSize];
		int[] arrRes = new int[arrSize];
		for(int i = 0; i < arrSize; i++) {
			arrOne[i] = (int)(Math.random()*10) + 1;
			arrTwo[i] = (int)(Math.random()*10) + 1;
		}
		long start = System.nanoTime();
		for(int i = 0; i < arrSize; i++) {
			arrRes[i] = arrOne[i] + arrTwo[i];
		}
		long end = System.nanoTime();
		System.out.println("CPU Time taken: " + (end-start));
		start = System.nanoTime();
	    arrRes = gpu.addMatrices(arrOne, arrTwo);
	    end = System.nanoTime();
	    System.out.println("GPU Time taken with overhead: " + (end-start));
	}
}
