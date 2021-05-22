package Test1;

import java.util.*;
import java.io.*;
import java.awt.image.*;
import java.awt.*;
import javax.imageio.ImageIO;
import org.jblas.*;
import java.awt.*;
import javax.swing.JFrame;

public class ImageColorTest {
	public static void main(String[] args) {
		//learn to colorize black and white images to become colored
		File bw_Images = new File("data/Colorizing/bw");
		File color_Images = new File("data/Colorizing/fullcolor");
		JFrame frame = new JFrame("Colorizer Window");
        frame.setVisible(true);
        frame.setSize(600,400);
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        
        ColorizerDisplay display = new ColorizerDisplay();
        frame.add(display);
        display.setVisible(true);
         
		int[] imageResizeDims = new int[] {81,81};
		//int[] netLayers = new int[] {imageResizeDims[0]*imageResizeDims[1],128,32,3*imageResizeDims[0]*imageResizeDims[1]};
		int[] netLayers = new int[] {imageResizeDims[0]*imageResizeDims[1],5196,5196,3*imageResizeDims[0]*imageResizeDims[1]};
		Network n = new Network(netLayers,0.1,new Sigmoid());
		File[] bwImages = bw_Images.listFiles();
		//for(File f : bwImages) {
		//	System.out.println(f);
		//}
		File[] colorImages = color_Images.listFiles();
		if (bwImages.length != colorImages.length) {
			System.err.println("Different number of files in b/w and color databases, they should be equal");
			System.exit(0);
		}
		for(int i=0;i<bwImages.length;i++) {
			BufferedImage bw = null;
			BufferedImage color = null;
			BufferedImage bwResize = null;
			BufferedImage colorResize = null;
			try {
				bw = ImageIO.read(bwImages[i]);
				color = ImageIO.read(colorImages[i]);
				bwResize = resizeImage(bw,imageResizeDims[0],imageResizeDims[1]);
				colorResize = resizeImage(color,imageResizeDims[0],imageResizeDims[1]);
			} catch (Exception e) {
				e.printStackTrace();
			}
			//BufferedImage bwResize = resizeImage(bw,imageResizeDims);
			DoubleMatrix netIn = convertImageToNetworkInput(bwResize);
			DoubleMatrix netOutIdeal = convertImageToNetworkOutput(colorResize);
			BufferedImage netOutActual = convertNetworkOutputToImage(n.feedForward(netIn), imageResizeDims);
			try {
				display.setBW(resizeImage(bwResize,400,400));
				display.setColorized(resizeImage(netOutActual,400,400));
			} catch (Exception e) {
				e.printStackTrace();
			}
			n.backPropGPU(netIn, netOutIdeal);
			try {
				Thread.sleep(100);
			} catch (Exception e) {
				e.printStackTrace();
			}
		}
	}
	public static DoubleMatrix convertImageToNetworkInput(BufferedImage in) {
		//input format: every row in input is the grayscale value of the image from 0 to 1 
		DoubleMatrix input = new DoubleMatrix(in.getHeight()*in.getWidth(),1);
		int incr = 0;
		for(int r=0;r<in.getWidth();r++) {
			for(int c=0;c<in.getHeight();c++) {
				input.put(incr,0,(double)(new Color(in.getRGB(r, c)).getRed())/255.0);
				incr++;
			}
		}
		return input;
	}
	public static DoubleMatrix convertImageToNetworkOutput(BufferedImage in) {
		DoubleMatrix input = new DoubleMatrix(3*in.getHeight()*in.getWidth(),1);
		int incr = 0;
		for(int r=0;r<in.getWidth();r++) {
			for(int c=0;c<in.getHeight();c++) {
				input.put(incr,0,(double)(new Color(in.getRGB(r, c)).getRed())/255.0);
				input.put(incr+1,0,(double)(new Color(in.getRGB(r, c)).getGreen())/255.0);
				input.put(incr+2,0,(double)(new Color(in.getRGB(r, c)).getGreen())/255.0);
				incr+=3;
				
			}
		}
		return input;
	}
	public static BufferedImage convertNetworkOutputToImage(DoubleMatrix y, int[] dims) {
		BufferedImage ret = new BufferedImage(dims[0], dims[1], BufferedImage.TYPE_INT_RGB);
		int incr = 0;
		for(int r=0;r<ret.getWidth();r++) {
			for(int c=0;c<ret.getHeight();c++) {
				Color pixelColor = new Color((int)(255*y.get(incr,0))%255,(int)(255*y.get(incr+1,0))%255,(int)(255*y.get(incr+2))%255);
				ret.setRGB(r, c, pixelColor.getRGB());
				incr+=3;
			}
		}
		return ret;
	}
	/*public static BufferedImage resizeImage(BufferedImage in, int[] newDims) {
		Image tmp = in.getScaledInstance(newDims[0], newDims[1], Image.SCALE_SMOOTH);
	    return new BufferedImage(newDims[0], newDims[1], BufferedImage.TYPE_INT_ARGB);
	}*/
	public static BufferedImage resizeImage(BufferedImage originalImage, int targetWidth, int targetHeight) throws IOException {
	    BufferedImage resizedImage = new BufferedImage(targetWidth, targetHeight, BufferedImage.TYPE_INT_RGB);
	    Graphics2D graphics2D = resizedImage.createGraphics();
	    graphics2D.drawImage(originalImage, 0, 0, targetWidth, targetHeight, null);
	    graphics2D.dispose();
	    return resizedImage;
	}
}
