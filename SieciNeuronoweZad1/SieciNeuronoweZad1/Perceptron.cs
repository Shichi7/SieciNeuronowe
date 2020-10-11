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

        private PerceptronSettings settings;

        private List<double> w_vector;
        private List<double> x_vector;

        public Perceptron(PerceptronSettings settings)
        {
            this.settings = settings;

            w_vector = new List<double>();

            for (int i = 0; i < settings.vector_len; i++)
            {
                bool minus = generator.Next(0, 2) == 0 ? true : false;
                double new_weight = generator.NextDouble() * settings.starting_weight_range;
                new_weight = minus ? -new_weight : new_weight;
                w_vector.Add(new_weight);
            }
        }

        public double iteration(Entry entry)
        {
            int predicted_y = predictY(entry);

            int error = entry.output - predicted_y;

            for (int i = 0; i < settings.vector_len; i++)
            {
                w_vector[i] = w_vector[i] + x_vector[i] * error * settings.modifier;
            }

            return error;
        }

        public int predictY(Entry entry)
        {
            double sum = 0;

            x_vector = new List<double>();

            if (settings.is_biased)
                x_vector.Add(1.0);

            foreach (double input in entry.inputs)
            {
                x_vector.Add(input);
            }

            for (int i = 0; i < settings.vector_len; i++)
            {
                sum += w_vector[i] * x_vector[i];
            }

            return (sum >= settings.threshold) ? settings.active_value : settings.inactive_value;
        }

        public string getWeightsString()
        {
            string weigths_string = "[";
            for (int i = 0; i < settings.vector_len; i++)
            {
                weigths_string += w_vector[i] + ", ";
            }

            weigths_string = weigths_string.Substring(0, weigths_string.Length - 2) + "]";

            return weigths_string;
        }

        public static void setGenerator(Random generator)
        {
            Perceptron.generator = generator;
        }
    }
}
