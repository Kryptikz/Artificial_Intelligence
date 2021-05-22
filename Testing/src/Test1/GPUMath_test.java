package Test1;

import org.jblas.DoubleMatrix;

public class GPUMath_test {
	public static void main(String[] args) {
		GPUMath gpu = new GPUMath();
		/*
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
			arrRes[i] = arrOne[i] * arrTwo[i];
		}
		long end = System.nanoTime();
		System.out.println("CPU Time taken: " + (end-start));
		start = System.nanoTime();
	    arrRes = gpu.addMatrices(arrOne, arrTwo);
	    end = System.nanoTime();
	    System.out.println("GPU Time taken with overhead: " + (end-start));
	    */
		//double[][] arrOne = new double[][] {{81, 5}, {4, 5}};
		//double[][] arrTwo = new double[][] {{7, 8}, {9, 10}};
		
		double[][] arrOne = new double[1000][1000];
		double[][] arrTwo = new double[1000][1];
		for(int r=0;r<arrOne.length;r++) {
			for(int c=0;c<arrOne[0].length;c++) {
				arrOne[r][c] = (Math.random()*10)-5;
			}
		}
		for(int r=0;r<arrTwo.length;r++) {
			for(int c=0;c<arrTwo[0].length;c++) {
				arrTwo[r][c] = (Math.random()*10)-5;
			}
		}
		
		int M = arrOne.length;
		int N = arrTwo[0].length;
		int K = arrTwo.length;
		//arrTwo = new double[][] {{7,9,11},{8,10,12}};
		long start = System.nanoTime();
		double[][] result = gpu.dotProduct(M, N, K, arrOne, arrTwo);
		long end = System.nanoTime();
	    System.out.println("GPU Time taken with overhead: " + (double)(end-start)/(long)(Math.pow(10,9)));
		DoubleMatrix A = new DoubleMatrix(arrOne);
		DoubleMatrix B = new DoubleMatrix(arrTwo);
		start = System.nanoTime();
		DoubleMatrix C = A.mmul(B);
		end = System.nanoTime();
		System.out.println("CPU Time taken with overhead: " + (double)(end-start)/(long)(Math.pow(10,9)));
		
	    /*for(double[] da : result) {
			for(double d : da) {
				System.out.print(d + " ");
			}
			System.out.println();
		}*/
	    gpu.close();
	}
}
