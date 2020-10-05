using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SieciNeuronoweZad1
{
    class Perceptron
    {
        private static Random generator;

        private bool is_bipolar; //else unipolar
        private bool is_biased;

        private int vector_length;

        private double threshold;
        private double modifier;

        private double[] x_vector;
        private double[] w_vector;

        public Perceptron(double modifier, double random_weight_range = 1.0, bool is_bipolar = false, bool is_biased = false)
        {
            this.modifier = modifier;
            this.is_bipolar = is_bipolar;
            this.is_biased = is_biased;

            threshold = 0.5;

            vector_length = is_biased ? 3 : 2;

            x_vector = new double[vector_length];
            w_vector = new double[vector_length];

            if (is_biased)
                x_vector[0] = 1;

            for (int i = 0; i < vector_length; i++)
            {
                bool minus = generator.Next(0, 2) == 0 ? true : false;
                w_vector[i] = generator.NextDouble() * random_weight_range;
                w_vector[i] = minus ? -w_vector[i] : w_vector[i];
            }
        }

        public double iteration(double[] x_vector, double y)
        {
            double temp_y = 0;

            for (int i = 0; i < vector_length; i++)
            {
                temp_y += w_vector[i] * x_vector[i];
            }

            double calculated_y = temp_y >= threshold ? 1.0 : 0.0;
            double error = y - calculated_y;

            for (int i = 0; i < vector_length; i++)
            {
                w_vector[i] = w_vector[i] + x_vector[i] * error * modifier;
            }

            return error;
        }

        public double calculate(double[] x_vector)
        {
            double temp_y = 0;

            for (int i = 0; i < vector_length; i++)
            {
                temp_y += w_vector[i] * x_vector[i];
            }

            return temp_y >= threshold ? 1.0 : 0.0;
        }

        public string getWeightsString()
        {
            string weigths_string = "[";
            for (int i = 0; i < vector_length; i++)
            {
                weigths_string += w_vector[i] + ", ";
            }

            //weigths_string = weigths_string.Substring(weigths_string.Length - 2);

            weigths_string += "]";

            return weigths_string;
        }

        public static void setGenerator(Random generator)
        {
            Perceptron.generator = generator;
        }




    }
}
