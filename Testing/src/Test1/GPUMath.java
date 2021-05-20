package Test1;
import static org.jocl.CL.*;
import org.jocl.*;
public class GPUMath {
	private static String addProgram = 
			"__kernal void " + 
		    "addKernal(__global const float *a, __global const float *b, __global const float *c){" + 
			"int gid = get_global_id(0);" + 
		    "c[gid] = a[gid] + b[gid];";
	public static double[] addMatrices(double[] A, double[] B) {
		double[] result = new double[A.length];
	    Pointer srcA = Pointer.to(A);
	    Pointer srcB = Pointer.to(B);
	    Pointer srcR = Pointer.to(result);
	    final int platformIndex = 0;
        final long deviceType = CL_DEVICE_TYPE_ALL;
        final int deviceIndex = 0;
        CL.setExceptionsEnabled(true);
        
		return result;
	}
}
