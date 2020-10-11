using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SieciNeuronoweZad1
{
    class Program
    {
        private const int MAX_ITERATIONS = 100000;
        private static int experiment_number = 1;

        static void Main(string[] args)
        {
            Random generator = new Random();
            Perceptron.setGenerator(generator);

            teachPerceptron("AND", new PerceptronSettings(0.01, 1.0, true, true));
            teachPerceptron("OR", new PerceptronSettings(0.01, 1.0, true, true));
        }

        static void teachPerceptron(string problem_name, PerceptronSettings settings)
        {
            string final_text = string.Format("EKSPERYMENT [{0}]\n\n", experiment_number);
            final_text += string.Format("Nazwa danych: [{0}]\n", problem_name);
            final_text += settings.dumpSettingsString();

            List<Entry> dataset = DataSets.getTrainingDataset(problem_name, settings.is_bipolar);

            if (dataset.Count>0)
            {
                settings.setVectorLen(dataset[0].vector_length);

                Perceptron perceptron = new Perceptron(settings);

                final_text += string.Format("Wektor wag początkowych: {0}\n\n", perceptron.getWeightsString());

                int iterations = 0;

                bool run_loop = true;

                while (run_loop)
                {
                    double error = 0;

                    foreach (Entry entry in dataset)
                        error += Math.Pow(perceptron.iteration(entry), 2);

                    if ((error == 0)||(iterations > MAX_ITERATIONS))
                    {
                        run_loop = false;
                    }
    
                    iterations++;
                }

                if (iterations <= MAX_ITERATIONS)
                {
                    final_text += string.Format("Wyuczono w: [{0}] iteracji\nWektor wag ostatecznych: {1}\n\n", iterations, perceptron.getWeightsString());
                    List<Entry> test_dataset = DataSets.getTestingDataset(problem_name, settings.is_bipolar);
                    if (test_dataset.Count>0)
                    {
                        final_text += "Testy:\n";
                        foreach (Entry entry in test_dataset)
                            final_text += entry.dumpEntryString() + string.Format("Input otrzymany: [{0}]\n", perceptron.predictY(entry));
                    }
                }
                else
                {
                    final_text += "Nie wyuczono\n\n";
                }
            }
            else
            {
                final_text += string.Format("Nie ma danych o nazwie[{0}]\n\n", problem_name); ;
            }

            experiment_number++;

            Console.WriteLine(final_text);
            Console.ReadKey();
        }
    }
}
