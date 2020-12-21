package com.company;
import java.util.*;

public class Adaline
{
    //public String info_string;
    private static double threshold = 0;
    private static double modifier;

    private double[] x_vector = new double[3];
    private double[] w_vector = new double[3];

    private double y;
    public double calculated_y;
    public double error;

    public Adaline(double starting_weight_range, double alpha_modifier, Random generator)
    {
        this.modifier = alpha_modifier;

        x_vector[0] = 1;

        for(int i=0; i<w_vector.length; i++)
        {
            w_vector[i] = generator.nextDouble()*starting_weight_range;
            if (generator.nextInt(2)==0)
            {
                w_vector[i]*= -1;
            }
        }
    }

    public void setData(double x1, double x2, double y)
    {
        x1 = x1==0 ? -1 : x1;
        x2 = x2==0 ? -1 : x2;
        y = y==0 ? -1 : y;

        this.x_vector[1] = x1;
        this.x_vector[2] = x2;
        this.y = y;
    }

    public void calculate()
    {
        calculateOutput();
        //setInfoString();
    }

    public void calculateNewWeights()
    {
        for(int i=0; i<w_vector.length; i++)
        {
            w_vector[i] = w_vector[i] + 2 * x_vector[i] * error * modifier;
        }
    }

    private void calculateOutput()
    {
        double temp_y = 0;
        calculated_y = -1;

        for(int i=0; i<w_vector.length; i++)
        {
            temp_y += w_vector[i]*x_vector[i];
        }

        error = y - temp_y;

        if (temp_y >= threshold)
        {
            calculated_y = 1;
        }
    }

    /*public void setInfoString()
    {
        info_string = "";
        info_string += "bias: ["+x_vector[0]+"]\n";
        info_string += "wejscie: ["+x_vector[1]+", "+x_vector[2]+"]\n";
        info_string += "wyjscie: ["+y+"]\n";
        info_string += "wyjscie obliczone: ["+calculated_y+"]\n";
        info_string += "waga0: ["+w_vector[0]+"]\n";
        info_string += "waga1: ["+w_vector[1]+"]\n";
        info_string += "waga2: ["+w_vector[2]+"]\n";
        info_string += "error: ["+error+"]\n";
    }*/
}
