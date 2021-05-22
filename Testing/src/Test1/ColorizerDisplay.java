package Test1;

import java.awt.*;
import javax.swing.*;
import java.util.*;
import java.io.*;
import java.awt.image.*;
import javax.imageio.ImageIO;
public class ColorizerDisplay extends JPanel {
    BufferedImage bw;
    BufferedImage colorized;
    public ColorizerDisplay() {
    }
    public void redraw() {
        this.repaint();
    }
    public void paintComponent(Graphics g){
    	if (bw != null) {
	        try {
	            super.paintComponent(g);
	            g.drawImage(bw, 0, 0, this);
	            g.drawImage(colorized,bw.getWidth()+50,0,this);
	        } catch (Exception e) {
	            System.out.println("Failed to draw frame");
	            e.printStackTrace();
	            return;
	        }
    	}
    }
    public void setBW(BufferedImage bw) {
        this.bw = bw;
        this.redraw();
    }
    public void setColorized(BufferedImage colorized) {
    	this.colorized = colorized;
    	this.redraw();
    }
}