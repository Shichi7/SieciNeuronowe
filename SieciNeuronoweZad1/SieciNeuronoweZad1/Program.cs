using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SieciNeuronoweZad1
{
    class Program
    {
        static void Main(string[] args)
        {
            Random generator = new Random();
            Perceptron.setGenerator(generator);

            teachPerceptron(0.01, 1.0, false, false);
            teachPerceptron(0.05, 1.0, false, false);
            teachPerceptron(0.10, 1.0, false, false);
        }

        static void teachPerceptron(double alpha, double weight_range, bool is_bipolar, bool is_biased)
        {
            string final_text = "Alpha: [" + alpha + "]\n";
            final_text += "Zakres wag początkowych: [-" + weight_range + ", "+weight_range+"]\n";

            Perceptron perceptron = new Perceptron(alpha, weight_range, is_bipolar, is_biased);
            final_text += "Wektor wag początkowych: " + perceptron.getWeightsString() + "\n";

            int max_iterations = 10000;
            int iterations = 0;
            double error = 1;
            bool success = true;

            while (error != 0)
            {
                iterations++;
                error = 0;
                error += perceptron.iteration(new double[] {0, 0}, 0);
                error += perceptron.iteration(new double[] {0, 1}, 0);
                error += perceptron.iteration(new double[] {1, 0}, 0);
                error += perceptron.iteration(new double[] {1, 1}, 1);

                if (iterations > max_iterations)
                {
                    error = 0;
                    success = false;
                }
            }

            final_text += success ? "Wyuczono w: [" + iterations + "] iteracji\nWektor wag ostatecznych: "+perceptron.getWeightsString() : "Nie wyuczono";
            final_text += "\n\n";
            Console.WriteLine(final_text);
            Console.ReadKey();
        }
    }
}
