package Test1;
import static org.jocl.CL.*;
import org.jocl.*;
public class GPUMath {
	cl_context context;
	cl_device_id devices[];
	cl_command_queue commandQueue;
    private String addProgram = 
			"__kernel void " + 
		    "addKernel(__global const int *a, __global const int *b, __global int *c){" + 
			"int gid = get_global_id(0);" + 
		    "c[gid] = a[gid] + b[gid];}";
    
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
        clGetDeviceIDs(platform, deviceType, numDevices, devices, null);
        cl_device_id device = devices[deviceIndex];
        
        context = clCreateContext(contextProperties, 1, new cl_device_id[]{device}, null, null, null); //context for a specific GPU
        
        //create command queue
        cl_queue_properties properties = new cl_queue_properties();
        commandQueue = clCreateCommandQueueWithProperties(context, devices[0], properties, null);
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
        cl_mem rMem = clCreateBuffer(context, CL_MEM_READ_WRITE, Sizeof.cl_float * buf_length, null, null);
        
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
        
        //runs the command
        long start = System.nanoTime();
        clEnqueueNDRangeKernel(commandQueue, kernel, 1, null, global_work_size, null, 0, null, null);
        long end = System.nanoTime();
        System.out.println("GPU without overhead: "+(end- start));
        //reads the data
        clEnqueueReadBuffer(commandQueue, rMem, CL_TRUE, 0,buf_length * Sizeof.cl_float, srcR, 0, null, null);
        
        //frees allocted memory
        clReleaseMemObject(srcMemA);
        clReleaseMemObject(srcMemB);
        clReleaseMemObject(rMem);
        clReleaseKernel(kernel);
        clReleaseProgram(program);
		return result;
	}
	public void close() {
		clReleaseContext(context);
		clReleaseCommandQueue(commandQueue);
		
	}
}
