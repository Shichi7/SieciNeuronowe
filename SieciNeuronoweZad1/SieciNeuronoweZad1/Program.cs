using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SieciNeuronoweZad1
{
    class Program
    {
        static int experiment_number = 1;
        static void Main(string[] args)
        {
            Random generator = new Random();
            Perceptron.setGenerator(generator);

            //perceptronWagesExperiment();
            //perceptronThresholdExperiment();
            perceptronAlphasExperiment();
            //perceptronFunctionExperiment();

            //manager("AND", new PerceptronSettings(100, 0.05, false, false, false));
            manager("AND", new PerceptronSettings(0.01, 0.001, true, true, true));
        }

        static void perceptronThresholdExperiment()
        {
            List<double> thresholds = new List<double> { -0.5, -0.25, 0, 0.25, 0.5, 0.75, 0.9};

            foreach (double threshold in thresholds)
            {
                customExperiment(new PerceptronSettings(0.01, 1.0, false, false, false, threshold), "OR", 100);
            }

            Console.ReadKey();
        }

        static void perceptronWagesExperiment()
        {
            List<double> wages = new List<double> { 1, 0.8, 0.6, 0.4, 0.2, 0.1, 0.001};

            foreach (double wage in wages)
            {
                customExperiment(new PerceptronSettings(0.01, wage, false, true, true), "AND", 100);
            }

            Console.ReadKey();
        }

        static void perceptronAlphasExperiment()
        {
            List<double> alphas = new List<double> { 0.01, 0.02, 0.03, 0.05, 0.055, 0.060, 1};

            foreach (double alpha in alphas)
            {
                customExperiment(new PerceptronSettings(alpha, 0.5, true, true, true), "AND", 100);
            }

            Console.ReadKey();
        }
        static void perceptronFunctionExperiment()
        {
            customExperiment(new PerceptronSettings(0.01, 0.5, false, false, true), "AND", 100);
            customExperiment(new PerceptronSettings(0.01, 0.5, false, true, true), "AND", 100);
            customExperiment(new PerceptronSettings(0.2, 0.01, false, false, true), "AND", 100);
            customExperiment(new PerceptronSettings(0.2, 0.01, false, true, true), "AND", 100);

            Console.ReadKey();
        }

        static void customExperiment(PerceptronSettings settings, string data_name, int reps)
        {
            double mean_iterations = 0;
            for (int i = 0; i<reps; i++)
            {
                mean_iterations += manager(data_name, settings, false);
            }
            mean_iterations /= reps;

            Console.WriteLine("Eksperyment {0} - średnia liczba iteracji: [{1}]", experiment_number++, mean_iterations);
        }

        static int manager(string problem_name, PerceptronSettings settings, bool log_results = true)
        {
            string log = string.Format("EKSPERYMENT START~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n\n");

            Perceptron perceptron = new Perceptron(settings, problem_name);

            int iterations = perceptron.teach();
            log += perceptron.teaching_log;

            if (perceptron.teaching_successfull)
            {
                perceptron.test();
                log += perceptron.testing_log;
            }

            if (log_results)
            {
                Console.WriteLine(log);
                Console.ReadKey();
            }

            return iterations;
        }
    }
}
