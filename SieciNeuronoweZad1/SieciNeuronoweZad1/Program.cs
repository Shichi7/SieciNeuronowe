using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SieciNeuronoweZad1
{
    class Program
    {
        private static int experiment_number = 1;

        static void Main(string[] args)
        {
            Random generator = new Random();
            Perceptron.setGenerator(generator);

            manager("AND", new PerceptronSettings(0.005, 1.0, true, false, true));
            manager("OR", new PerceptronSettings(0.005, 1.0, true, false, true));
            manager("AND", new PerceptronSettings(0.005, 1.0, false, false, true));
            manager("OR", new PerceptronSettings(0.005, 1.0, false, true, true));

        }

        static void manager(string problem_name, PerceptronSettings settings)
        {
            string log = string.Format("EKSPERYMENT [{0}]\n\n", experiment_number);

            Perceptron perceptron = new Perceptron(settings, problem_name);

            perceptron.teach();
            log += perceptron.teaching_log;

            if (perceptron.teaching_successfull)
            {
                perceptron.test();
                log += perceptron.testing_log;
            }

            experiment_number++;

            Console.WriteLine(log);
            Console.ReadKey();
        }
    }
}
