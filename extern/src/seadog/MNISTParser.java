package seadog;

import java.awt.image.BufferedImage;
import java.io.DataInputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.ObjectOutputStream;
import java.io.FileOutputStream;
import java.util.zip.GZIPOutputStream;
import java.io.OutputStream;
import java.util.Vector;

import javax.imageio.ImageIO;

/**
 * @author seadog
 * @see https://github.com/seadog/MNISTParser
 */
public class MNISTParser {
    private static byte[] getData(File file) throws IOException {
        DataInputStream dis = new DataInputStream(new FileInputStream(file));
        byte[] data = new byte[(int)file.length()];
        dis.readFully(data);
        dis.close();
        return data;
    }
    
    private static boolean checkMagic(byte[] data, int magic){
        int data_magic = ((data[3] & 0xFF) << 0) | ((data[2] & 0xFF) << 8) | ((data[1] & 0xFF) << 16) | ((data[0] & 0xFF) << 24);
        return magic == data_magic;
    }
    
    private static int getLength(byte[] data){
        int length = ((data[7] & 0xFF) << 0) | ((data[6] & 0xFF) << 8) | ((data[5] & 0xFF) << 16) | ((data[4] & 0xFF) << 24);
        return length;
    }
    
    private static int[] getLabels(byte[] data){
        int[] labels = new int[data.length - 8];

        for(int i = 8; i < data.length; i++){
            labels[i-8] = data[i] & 0xFF;
        }
        
        for(int i = 0; i < labels.length; i++){
            if(labels[i] > 9 || labels[i] < 0){
                System.out.println("Label greater than 9 or less than 0!");
                System.exit(1);
            }
        }
        return labels;
    }
    
    private static BufferedImage[] getImages(byte[] image_data){
        int byte_index = 8;
        int image_index = 0;
        
        BufferedImage[] images = new BufferedImage[getLength(image_data)];
        
        int rows = ((image_data[byte_index+3] & 0xFF) << 0) | ((image_data[byte_index+2] & 0xFF) << 8) | ((image_data[byte_index+1] & 0xFF) << 16) | ((image_data[byte_index+0] & 0xFF) << 24);
        byte_index += 4;
        int cols = ((image_data[byte_index+3] & 0xFF) << 0) | ((image_data[byte_index+2] & 0xFF) << 8) | ((image_data[byte_index+1] & 0xFF) << 16) | ((image_data[byte_index+0] & 0xFF) << 24);
        byte_index += 4;
        
        if(rows != 28 || cols != 28){
            System.out.println("Rows/Cols error! " + rows + "x" + cols + ", byte position: " + (byte_index-8));
            System.exit(1);
        }
        
        while(byte_index < image_data.length){
            BufferedImage image = new BufferedImage(cols, rows,  BufferedImage.TYPE_BYTE_GRAY);
            for(int i = 0; i < rows; i++){
                for(int j = 0; j < cols; j++){
                    int grey_value = image_data[byte_index] & 0xFF;
                    if(grey_value < 0 || grey_value > 255) {
                        System.out.println("Pixel value error!");
                        System.exit(1);
                    }
                    int pixel_value = ((grey_value & 0xFF) << 0) | ((grey_value & 0xFF) << 8) | ((grey_value & 0xFF) << 16) | ((255 & 0xFF) << 24);
                    image.setRGB(j, i, pixel_value);
                    byte_index++;
                }
            }
            images[image_index] = image;
            image_index++;
        }
        
        return images;
    }
    
    private static void write_images(int[] labels, BufferedImage[] images, File directory){
        for(int image_class = 0; image_class < 10; image_class++){
            int index = 0;
            for(int i = 0; i < labels.length; i++){
                if(labels[i] == image_class){
                    File file = new File(directory, image_class + "_" + String.format("%04d", index) + ".png");
                    try {
                        ImageIO.write(images[i], "png", file);
                    } catch (IOException e) {
                        e.printStackTrace();
                        System.exit(1);
                    }
                    index++;
                }
            }
        }
    }

    private static double[] imgToVector(BufferedImage img){
        double[] vector = new double[img.getWidth()*img.getHeight()];
        for(int x = 0; x < img.getWidth(); x++){
            for(int y = 0; y < img.getHeight(); y++){
                int pxl = img.getRGB(x,y);
                // normalize 0=black 1=white
                double gray = ((pxl&0xFF) + (pxl&(0xFF<<8)) + (pxl&(0xFF<<16))) / (255*3.0);
                vector[x+y*img.getWidth()] = gray;
            }
        }
        return vector;
    }

    // suppress necessary for using array with generics
    @SuppressWarnings("unchecked")
    private static void write_binary(int[] labels, BufferedImage[] images, File outputFile, boolean zip){
        Vector<double[]>[] labeledInstances = (Vector<double[]>[])new Vector[10];
        for(int i = 0; i < labeledInstances.length; i++)
            labeledInstances[i] = new Vector<double[]>();
        assert(labels.length == images.length);
        for(int i = 0; i < labels.length; i++)
            labeledInstances[labels[i]].add(imgToVector(images[i]));
        // restructure into output format now that we know the # in each instance list
        double[][][] out = new double[labeledInstances.length][][];
        for(int i = 0; i < labeledInstances.length; i++){
            out[i] = new double[labeledInstances[i].size()][];
            for(int j = 0; j < out[i].length; j++){
                out[i][j] = labeledInstances[i].get(j);
            }
        }
        labeledInstances = null;
        // write to file
        try{
            OutputStream outStream = new FileOutputStream(outputFile);
            if(zip){
                outStream = new GZIPOutputStream(outStream);
            }
            ObjectOutputStream oos = new ObjectOutputStream(outStream);
            oos.writeObject(out);
            oos.close();
        } catch (Throwable t) {
            t.printStackTrace();
            System.exit(1);
        }
    }
    
    public static void printHelpAndDie(){
        System.out.println("<program> (-b -z) <label_file> <image_file> <output_dir/file>");
        System.out.println("-b = output as binary, serialized java double[<label>][<instance>][<pixel>]");
        System.out.println("-z = output binary zipped using a GZIPOutputStream");
        System.exit(1);
    }   
    public static void main(String[] args){
        // parse args
        if(args.length < 3 || args.length > 5)
            printHelpAndDie();
        int arg  = 0;
        boolean binary = false;
        boolean zipped = false;
        if(args[arg].equals("-b")){
            binary = true;
            arg++;
            if(args[arg].equals("-z")){
                zipped = true;
                arg++;
            }
        }
        File label_file = new File(args[arg++]);
        File image_file = new File(args[arg++]);
        File output_loc = new File(args[arg++]);
        if(arg != args.length)
            printHelpAndDie();
        
        // validate args
        if(!label_file.exists() || !label_file.isFile()){
            System.out.println("Label file is either a directory or does not exist");
            System.exit(1);
        }
        if(!image_file.exists() || !image_file.isFile()){
            System.out.println("Image file is either a directory or does not exist");
            System.exit(1);
        }
        if(!binary && (!output_loc.exists() || !output_loc.isDirectory())){
            System.out.println("Output directory is either a file or does not exist");
            System.exit(1);
        }
        
        // validate files
        byte[] label = null;
        byte[] image = null;
        try {
            label = getData(label_file);
            image = getData(image_file);
        } catch(IOException e) {
            e.printStackTrace();
            System.exit(1);
        }
        if(!checkMagic(label, 2049)){
            System.out.println("Label magic failed");
            System.exit(1);
        }
        if(!checkMagic(image, 2051)){
            System.out.println("Image magic failed");
            System.exit(1);
        }
        if(getLength(label) != getLength(image)){
            System.out.println("Length mis-match");
            System.exit(1);
        }

        // process and output
        int[] labels = getLabels(label);
        BufferedImage[] images = getImages(image);
        if(binary)
            write_binary(labels, images, output_loc, zipped);
        else
            write_images(labels, images, output_loc);
    }
}
