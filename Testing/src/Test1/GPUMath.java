package Test1;
import static org.jocl.CL.*;
import org.jocl.*;
import org.jblas.*;
public class GPUMath {
	cl_context context;
	cl_device_id devices[];
	cl_command_queue commandQueue;
	cl_kernel multKernel;
	cl_program multProgram;
    private String addProgram = 
			"__kernel void " + 
		    "addKernel(__global const int *a, __global const int *b, __global int *c){" + 
			"int gid = get_global_id(0);" +
		    "c[gid] = a[gid] * b[gid];" + 
			"c[gid] = c[gid] * 5;}";
    private String matMul = 
    		"__kernel void matMulKernel( const int M, const int N, const int K, "+ 
    		"__global const double *A, __global const double *B, __global double *C){" +
    	    "const int globalRow = get_global_id(0);"
    	    + "const int globalCol = get_global_id(1);"
    	    + "float value = 0.0f;"
    	    + "for(int k = 0; k < K; k++){"
    	    + "   value += A[k*M + globalRow] * B[globalCol*K + k];"
    	    + "   /*printf(\"\\nglobalRow=%d globalCol=%d M=%d N=%d K=%d m1: %f m2: %f \",globalRow,globalCol,M,N,K,A[k*M + globalRow],B[globalCol*K + k]);*/"
    	    + "}"
    	    + "C[globalCol*M + globalRow] = value;"
    	    + "}";
    public GPUMath() {
    	final long deviceType = CL_DEVICE_TYPE_ALL;
    	final int platformIndex = 0;
    	final int deviceIndex = 0;
    		
    	CL.setExceptionsEnabled(true);
    	
    	//creates the context properties
    	int numPlatformsArray[] = new int[1];
        clGetPlatformIDs(0, null, numPlatformsArray);
        int numPlatforms = numPlatformsArray[0];
    	cl_platform_id platforms[] = new cl_platform_id[numPlatforms];
        clGetPlatformIDs(platforms.length, platforms, null);
        cl_platform_id platform = platforms[platformIndex];
        cl_context_properties contextProperties = new cl_context_properties();
        contextProperties.addProperty(CL_CONTEXT_PLATFORM, platform);
        
        //create the context
        int numDevicesArray[] = new int[1];
        clGetDeviceIDs(platform, deviceType, 0, null, numDevicesArray);
        int numDevices = numDevicesArray[0];
        devices = new cl_device_id[numDevices]; //array of the GPUs, will be changed for the bitcoin miner
        //clGetDeviceIDs(platform, deviceType, numDevices, devices, null);
        
        //cl_device_id devices[] = new cl_device_id[numDevices];
        clGetDeviceIDs(platform, deviceType, numDevices, devices, null);
        for (int i=0; i<numDevices; i++)
        {
            String deviceName = getString(devices[i], CL_DEVICE_NAME);
            System.out.println("Device "+i+" of "+numDevices+": "+deviceName);
        }
        cl_device_id device = devices[0];
        //System.out.println(getString(device, CL_DEVICE_NAME));
        /*for(char c : value) {
        	System.out.print(c);
        }*/
        
        context = clCreateContext(contextProperties, 1, new cl_device_id[]{device}, null, null, null); //context for a specific GPU
        
        //create command queue
        cl_queue_properties properties = new cl_queue_properties();
        //commandQueue = clCreateCommandQueueWithProperties(context, devices[0], properties, null);
        commandQueue = clCreateCommandQueue(context, devices[0], CL_QUEUE_PROFILING_ENABLE, null);
        
        //create the program and kernels
        String[] matMulStr = new String[] { matMul };
        cl_program program = clCreateProgramWithSource(context, 1, matMulStr, null, null);
        clBuildProgram(program, 0, null, null, null, null);
        multKernel = clCreateKernel(program, "matMulKernel", null);
    }
	//[row][col] 
    //m = num row in A 
    //n = num col in B
    //k = num col in A or num row in B
    public DoubleMatrix mmulGPU(DoubleMatrix A, DoubleMatrix B) {
    	return new DoubleMatrix(dotProduct(A.getRows(),B.getColumns(),B.getRows(),B.toArray2(),A.toArray2()));
    }
    public double[][] dotProduct(final int M, final int N, final int K, double[][] B, double[][] A) {
    	//COMPUTE DOT PRODUCT A TIMES B, just put matrices in correct order!!!!
    	long buf_length = M * N;
    	long start = System.nanoTime();
    	
    	cl_mem memA = clCreateBuffer(context, CL_MEM_READ_ONLY, Sizeof.cl_double * M * K, null, null);
        cl_mem memB = clCreateBuffer(context, CL_MEM_READ_ONLY, Sizeof.cl_double * K * N, null, null);
        cl_mem memR = clCreateBuffer(context, CL_MEM_READ_WRITE, Sizeof.cl_double * M * N, null, null);
        
        double[] aConverted = squishMatrix(A);
        double[] bConverted = squishMatrix(B);
        
        //long end = System.nanoTime();
        //System.out.println("Flatten Time: " + (double)(end-start)/(long)(Math.pow(10,9)));
        
        double[] C = new double[M*N];
        
        clEnqueueWriteBuffer(commandQueue, memA, CL_TRUE, 0, Sizeof.cl_double * M * K, Pointer.to(aConverted), 0, null, null);
        clEnqueueWriteBuffer(commandQueue, memB, CL_TRUE, 0, Sizeof.cl_double * K * N, Pointer.to(bConverted), 0, null, null);
        clEnqueueWriteBuffer(commandQueue, memR, CL_TRUE, 0, Sizeof.cl_double * M * N, Pointer.to(C), 0, null, null);
        
        int[] in1 = new int[] {M};
        int[] in2 = new int[] {N};
        int[] in3 = new int[] {K};
        
        clSetKernelArg(multKernel, 0, Sizeof.cl_int, Pointer.to(in1));
        clSetKernelArg(multKernel, 1, Sizeof.cl_int, Pointer.to(in2));
        clSetKernelArg(multKernel, 2, Sizeof.cl_int, Pointer.to(in3));
        clSetKernelArg(multKernel, 3, Sizeof.cl_mem, Pointer.to(memA));
        clSetKernelArg(multKernel, 4, Sizeof.cl_mem, Pointer.to(memB));
        clSetKernelArg(multKernel, 5, Sizeof.cl_mem, Pointer.to(memR));
        
        long global[] = new long[]{M, N};
        //long local[] = new long[]{2, 2};
        long local[] = new long[] {1,1};
        
        cl_event event = new cl_event();
        
        //start = System.nanoTime();
        clEnqueueNDRangeKernel(commandQueue, multKernel, 2, null, global, local, 0, null, event);
        
        clWaitForEvents(1,new cl_event[] {event});
        
        
        clEnqueueReadBuffer(commandQueue, memR, CL_TRUE, 0,buf_length * Sizeof.cl_double, Pointer.to(C), 0, null, null);
        
        clFinish(commandQueue);
        
        long time_start = 0;
        long time_end = 0;
        //clGetEventProfilingInfo(event, param_name, param_value_size, param_value, param_value_size_ret)
        clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, Sizeof.cl_long, Pointer.to(new long[] {time_start}), null);
        clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, Sizeof.cl_long, Pointer.to(new long[] {time_end}), null);

        double nanoSeconds = time_end-time_start;
        //System.out.println(time_start);
        //System.out.printf("OpenCl Execution time is: %f milliseconds \n",(double)nanoSeconds / 1000000.0);
        
        
        long end = System.nanoTime();
        
        clReleaseMemObject(memA);
        clReleaseMemObject(memB);
        clReleaseMemObject(memR);
        
        double[][] ret = unsquishMatrix(C, N);
        
	    //System.out.println("GPU Time taken without overhead: " + (double)(end-start)/(long)(Math.pow(10,9)));
        return ret;
    }
    
	public int[] addMatrices(int[] A, int[] B) {
		int buf_length = A.length;
		int[] result = new int[buf_length];
	    Pointer srcA = Pointer.to(A);
	    Pointer srcB = Pointer.to(B);
	    Pointer srcR = Pointer.to(result);
	    
	    //Allocates memory
        cl_mem srcMemA = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, Sizeof.cl_float * buf_length, srcA, null);
        cl_mem srcMemB = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, Sizeof.cl_float * buf_length, srcB, null);
        cl_mem rMem = clCreateBuffer(context, CL_MEM_READ_WRITE, Sizeof.cl_int * buf_length, null, null);
        
        //creates program and kernel
        cl_program program = clCreateProgramWithSource(context, 1, new String[]{ addProgram }, null, null);
        clBuildProgram(program, 0, null, null, null, null);
        cl_kernel kernel = clCreateKernel(program, "addKernel", null);
          
        //sets kernel arguments
        clSetKernelArg(kernel, 0, Sizeof.cl_mem, Pointer.to(srcMemA));
        clSetKernelArg(kernel, 1, Sizeof.cl_mem, Pointer.to(srcMemB));
        clSetKernelArg(kernel, 2, Sizeof.cl_mem, Pointer.to(rMem));
        
        //defines num work items and work group size
        long global_work_size[] = new long[]{buf_length};
        //long local_work_size[] = new long[]{32};
        
        //runs the command
        long start = System.nanoTime();
        clEnqueueNDRangeKernel(commandQueue, kernel, 1, null, global_work_size, null, 0, null, null);
        long end = System.nanoTime();
        //System.out.println("GPU without overhead: "+(end- start));
        //reads the data
        clEnqueueReadBuffer(commandQueue, rMem, CL_TRUE, 0,buf_length * Sizeof.cl_int, srcR, 0, null, null);
        
        //frees allocated memory
        clReleaseMemObject(srcMemA);
        clReleaseMemObject(srcMemB);
        clReleaseMemObject(rMem);
        clReleaseKernel(kernel);
        clReleaseProgram(program);
		return result;
	}
	
	public void close() {
		clReleaseKernel(multKernel);
		clReleaseContext(context);
		clReleaseCommandQueue(commandQueue);
	}
	
	public static double[] squishMatrix(double[][] a) {
		double[] b = new double[a.length*a[0].length];
		for(int i = 0; i < b.length; i++) {
			b[i] = a[i/a[0].length][i%a[0].length];
		}
		return b;
	}
	
	public static double[][] unsquishMatrix(double[] a, int numCol) {
		double[][] b = new double[a.length/numCol][numCol];
		for(int i = 0; i < a.length; i++) {
			b[i/numCol][i%numCol] = a[i];
		}
		return b;
	}	
	private static String getString(cl_platform_id platform, int paramName)
    {
        long size[] = new long[1];
        clGetPlatformInfo(platform, paramName, 0, null, size);
        byte buffer[] = new byte[(int)size[0]];
        clGetPlatformInfo(platform, paramName, 
            buffer.length, Pointer.to(buffer), null);
        return new String(buffer, 0, buffer.length-1);
    }
	private static String getString(cl_device_id device, int paramName)
    {
        long size[] = new long[1];
        clGetDeviceInfo(device, paramName, 0, null, size);
        byte buffer[] = new byte[(int)size[0]];
        clGetDeviceInfo(device, paramName, 
            buffer.length, Pointer.to(buffer), null);
        return new String(buffer, 0, buffer.length-1);
    }
}
