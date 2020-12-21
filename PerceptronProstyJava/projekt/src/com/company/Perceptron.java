package com.company;

import java.util.*;

public class Perceptron
{
    //public String info_string;
    private static String function_type;
    private static double threshold;
    private static double modifier;

    private double[] x_vector = new double[3];
    private double[] w_vector = new double[3];

    private double y;
    public double calculated_y;
    public double error;

    private Random generator;

    public Perceptron(double starting_weight_range, double alpha_modifier, String function_type, Random generator)
    {
        this.modifier = alpha_modifier;
        this.generator = generator;
        this.function_type = function_type;

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
        if (function_type.equals("bipolar"))
        {
            x1 = x1==0 ? -1 : x1;
            x2 = x2==0 ? -1 : x2;
            y = y==0 ? -1 : y;

            threshold = 0;
        }
        else
        {
            threshold = 0.5;
        }

        this.x_vector[1] = x1;
        this.x_vector[2] = x2;
        this.y = y;
    }

    public void calculate()
    {
        calculateOutput();
        calculateError();
        calculateNewWeights();
        //setInfoString();
    }

    public void calculateNewWeights()
    {
        for(int i=0; i<w_vector.length; i++)
        {
            w_vector[i] = w_vector[i] + x_vector[i] * error * modifier;
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

        if (temp_y >= threshold)
        {
            calculated_y = 1;
        }
    }

    private void calculateError() //Perceptron prosty
    {
        error = 0;
        if (function_type.equals("unipolar"))
        {
            if ((calculated_y==0)&&(y==1)) {error = 1;}
            else if ((calculated_y==1)&&(y==0)) {error = -1;}
        }
        else if (function_type.equals("bipolar"))
        {
            error = y - calculated_y;
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
