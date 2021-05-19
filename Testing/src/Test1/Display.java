package Test1;
import javax.swing.*;
import java.awt.*;
public class Display extends JComponent{
    private int[][] image;
    private int box_size = 20;
    private int label;
    public Display(){
        image = new int[28][28];
        label = 0;
    }
    public void draw(){
        super.repaint();
    }
    public void update_image(int[][] new_image, int l){
        image = new_image;
        label = l;
    }
    public void paintComponent(Graphics g){
        super.paintComponent(g);
        for(int x = 0; x < image.length; x++){
            for(int y = 0; y < image[0].length; y++){
                g.setColor(new Color(image[x][y], image[x][y], image[x][y]));
                g.fillRect(x*box_size, y*box_size, box_size, box_size);
            }
        }
        g.setColor(Color.BLACK);
        g.drawString("Label: "+ label,50, 600);
    }
}