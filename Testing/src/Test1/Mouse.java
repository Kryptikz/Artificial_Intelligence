package Test1;

import java.awt.event.*;
public class Mouse implements MouseListener{
    Display screen;
    public Mouse(Display screen){
        this.screen = screen;
    }
    public void mousePressed(MouseEvent e){
    	
    }
    public void mouseReleased(MouseEvent e){
    	screen.pause = false;
    }
    public void mouseEntered(MouseEvent e) {

    }
    public void mouseExited(MouseEvent e) {

    }
    public void mouseClicked(MouseEvent e) {
        
    }
}
