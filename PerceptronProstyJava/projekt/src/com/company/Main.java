package com.company;
import java.text.DecimalFormat;
import java.util.*;

public class Main
{
    public static double LMS_threshold = 0.4;

    public static Perceptron perceptron;
    public static Adaline adaline;
    public static Random generator = new Random();
    private static String problem_name = "OR";

    public static void main(String[] args)
    {
        neuronDifferentStartingWeightsTest("PERCEPTRON");
        neuronDifferentAlphasTest("PERCEPTRON");

        neuronDifferentStartingWeightsTest("ADALINE");
        neuronDifferentAlphasTest("ADALINE");

        neuronDifferentFunctionTypeTest();
    }

    public static void neuronDifferentFunctionTypeTest()
    {
        String neuron_type = "PERCEPTRON";
        DecimalFormat double_format = new DecimalFormat("#0.00");
        double starting_weight_range = 0.8;
        double alpha_modifier = 0.1;

        print("");
        print("~~~~~~"+neuron_type+" DIFFERENT STARTING WEIGHTS TEST~~~~~~");

        String function_type = "bipolar";
        print("Function type: ("+function_type+")");
        customTest(neuron_type, starting_weight_range, alpha_modifier, function_type);

        function_type = "unipolar";
        print("Function type: ("+function_type+")");
        customTest(neuron_type, starting_weight_range, alpha_modifier, function_type);
    }

    public static void neuronDifferentStartingWeightsTest(String neuron_type)
    {
        DecimalFormat double_format = new DecimalFormat("#0.00");
        double[] ranges = {1.5, 1.3, 1.1, 0.9, 0.7, 0.5, 0.3, 0.1};
        double alpha_modifier = 0.01;
        String function_type = "bipolar";

        print("");
        print("~~~~~~"+neuron_type+" DIFFERENT STARTING WEIGHTS TEST~~~~~~");
        if (neuron_type.equals("ADALINE"))
        {
            print("~~~~~~TEACHING UNTIL (LMS <= "+ LMS_threshold +")~~~~~~");
        }

        for (int i=0; i<ranges.length; i++)
        {
            double starting_weight_range = ranges[i];
            print("Starting weight range: ("+ (double_format.format(-starting_weight_range))+" to " + (double_format.format(starting_weight_range))+")");

            customTest(neuron_type, starting_weight_range, alpha_modifier, function_type);
        }
    }

    public static void neuronDifferentAlphasTest(String neuron_type)
    {
        DecimalFormat double_format = new DecimalFormat("#0.000");
        double[] alphas = {0.001, 0.01, 0.02, 0.024, 0.026, 0.028, 0.029, 0.03, 1};
        double starting_weight_range = 0.8;
        String function_type = "bipolar";

        double alpha_modifier;

        print("");
        print("~~~~~~"+neuron_type+" DIFFERENT ALPHA MODIFIER TEST~~~~~~");
        if (neuron_type.equals("ADALINE"))
        {
            print("~~~~~~TEACHING UNTIL (LMS <= "+ LMS_threshold +")~~~~~~");
        }

        for (int i=0; i<alphas.length; i++)
        {
            alpha_modifier = alphas[i];
            print("Alpha modifier: ("+ (double_format.format(alpha_modifier))+")");

            customTest(neuron_type, starting_weight_range, alpha_modifier, function_type);
        }
    }

    private static void customTest(String neuron_type, double starting_weight_range, double alpha_modifier, String function_type)
    {
        double mean_iterations = 0;

        for (int j=0; j<100; j++)
        {
            if (neuron_type.equals("ADALINE"))
            {
                mean_iterations += teachAdaline(starting_weight_range, alpha_modifier);
            }
            else if (neuron_type.equals("PERCEPTRON"))
            {
                mean_iterations += teachPerceptron(starting_weight_range, alpha_modifier, function_type);
            }
        }

        mean_iterations/= 100;

        if (mean_iterations>10000)
        {
            print("Teaching was not achieved in 10000 iterations with this settings");
        }
        else
        {
            print("Mean number of iterations to teach after 100 trials: " + mean_iterations);
        }
    }

    private static int teachAdaline(double starting_weight_range, double alpha_modifier)
    {
        adaline = new Adaline(starting_weight_range, alpha_modifier, generator);
        int iterations = 0;
        double LMS = 1;

        while (LMS>LMS_threshold)
        {
            iterations++;
            LMS = 0;
            LMS += adalineTeachingIteration(0, 0);
            LMS +=adalineTeachingIteration(0, 1);
            LMS +=adalineTeachingIteration(1, 0);
            LMS +=adalineTeachingIteration(1, 1);
            LMS /= 4;

            if (iterations>10000) {LMS=0;}
        }

        return iterations;
    }

    private static int teachPerceptron(double starting_weight_range, double alpha_modifier, String function_type)
    {
        perceptron = new Perceptron(starting_weight_range, alpha_modifier, function_type, generator);
        int iterations = 0;
        double error = 1;

        while (error!=0)
        {
            iterations++;
            error = 0;
            error += perceptronTeachingIteration(0, 0);
            error += perceptronTeachingIteration(0, 1);
            error += perceptronTeachingIteration(1, 0);
            error += perceptronTeachingIteration(1, 1);

            if (iterations>10000) {error=0;}
        }

        return iterations;
    }

    private static double perceptronTeachingIteration(double x1, double x2)
    {
        double error;
        perceptron.setData(x1, x2, calculateY(x1, x2));
        perceptron.calculate();
        error = perceptron.error * perceptron.error;

        return error;
    }

    private static double adalineTeachingIteration(double x1, double x2)
    {
        double LMS;
        adaline.setData(x1, x2, calculateY(x1, x2));
        adaline.calculate();
        LMS = adaline.error * adaline.error;

        adaline.calculateNewWeights();
        return LMS;
    }

    private static double calculateY(double x1, double x2)
    {
        return (((problem_name.equals("OR"))&&(x1+x2>=1))||((problem_name.equals("AND"))&&(x1+x2==2))) ? 1 : 0;
    }

    public static void print(String to_print)
    {
        System.out.println(to_print);
    }
}
