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

        private List<Entry> training_dataset;

        private string problem_name;

        public string teaching_log;
        public string testing_log;
        public bool teaching_successfull;

        public Perceptron(PerceptronSettings settings, string problem_name)
        {
            teaching_successfull = false;

            this.settings = settings;
            this.problem_name = problem_name;

            w_vector = new List<double>();

            training_dataset = DataSets.getTrainingDataset(problem_name, settings.is_bipolar);
            if (training_dataset.Count > 0)
            {
                this.settings.setVectorLen(training_dataset[0].inputs.Count);
            }

            for (int i = 0; i < settings.vector_len; i++)
            {
                bool minus = generator.Next(0, 2) == 0 ? true : false;
                double new_weight = generator.NextDouble() * settings.starting_weight_range;
                new_weight = minus ? -new_weight : new_weight;
                w_vector.Add(new_weight);
            }
        }

        public void teach()
        {
            teaching_log = "";
            teaching_log += string.Format("Nazwa danych: [{0}]\n", problem_name);
            teaching_log += settings.dumpSettingsString();

            if (training_dataset.Count > 0)
            {
                teaching_log += string.Format("Wektor wag początkowych: {0}\n\n", getWeightsString());

                int iterations = 0;

                bool run_loop = true;

                while (run_loop)
                {
                    double error = 0;

                    foreach (Entry entry in training_dataset)
                    {
                        error += Math.Pow(iteration(entry), 2);
                    }
                    error /= training_dataset.Count;

                    if ((error == 0) || (iterations > PerceptronSettings.MAX_ITERATIONS))
                    {
                        run_loop = false;
                    }

                    if ((settings.is_adaline)&&(error < PerceptronSettings.LMS_THRESHOLD))
                    {
                        teaching_log += string.Format("Ostateczny LMS: [{0}]\n", error);
                        run_loop = false;
                    }

                    iterations++;
                }

                if (iterations <= PerceptronSettings.MAX_ITERATIONS)
                {
                    teaching_successfull = true;
                    teaching_log += string.Format("Wyuczono w: [{0}] iteracji\nWektor wag ostatecznych: {1}\n\n", iterations, getWeightsString());
                }
                else
                {
                    teaching_successfull = false;
                    teaching_log += "Nie wyuczono\n\n";
                }
            }
            else
            {
                teaching_successfull = false;
                teaching_log += string.Format("Nie ma danych o nazwie[{0}]\n\n", problem_name); ;
            }
        }

        public void test()
        {
            testing_log = "";

            List<Entry> test_dataset = DataSets.getTestingDataset(problem_name, settings.is_bipolar);
            if (test_dataset.Count > 0)
            {
                testing_log += "Testy:\n";
                foreach (Entry entry in test_dataset)
                    testing_log += entry.dumpEntryString() + string.Format("Input otrzymany: [{0}]\n", predictY(entry));
            }
        }

        private double iteration(Entry entry)
        {
            double predicted_y;

            if (settings.is_adaline)
                predicted_y = predictYNoThreshold(entry);
            else
                predicted_y = predictY(entry);

            double error = entry.output - predicted_y;

            for (int i = 0; i < settings.vector_len; i++)
            {
                double modification = x_vector[i] * error * settings.modifier;
                w_vector[i] = w_vector[i] + modification;
            }

            return error;
        }

        public double predictYNoThreshold(Entry entry)
        {
            double y = 0;

            x_vector = new List<double>();

            if (settings.is_biased)
                x_vector.Add(1.0);

            foreach (double input in entry.inputs)
            {
                x_vector.Add(input);
            }

            for (int i = 0; i < settings.vector_len; i++)
            {
                y += w_vector[i] * x_vector[i];
            }

            return y;
        }
        public double predictY(Entry entry)
        {
            return predictYNoThreshold(entry) >= settings.threshold ? settings.active_value : settings.inactive_value;
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
