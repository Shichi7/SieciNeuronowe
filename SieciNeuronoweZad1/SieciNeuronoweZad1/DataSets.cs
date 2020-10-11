using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SieciNeuronoweZad1
{
    class DataSets
    {
        public static List<Entry> getTrainingDataset(string name, bool bipolar = false)
        {
            List<Entry> dataset = new List<Entry>();
            if (name.Equals("AND"))
            {
                dataset.Add(new Entry(new List<double>() { 0, 0 }, 0, bipolar));
                dataset.Add(new Entry(new List<double>() { 0, 1 }, 0, bipolar));
                dataset.Add(new Entry(new List<double>() { 1, 0 }, 0, bipolar));
                dataset.Add(new Entry(new List<double>() { 1, 1 }, 1, bipolar));
            }
            else if (name.Equals("OR"))
            {
                dataset.Add(new Entry(new List<double>() { 0, 0 }, 0, bipolar));
                dataset.Add(new Entry(new List<double>() { 0, 1 }, 1, bipolar));
                dataset.Add(new Entry(new List<double>() { 1, 0 }, 1, bipolar));
                dataset.Add(new Entry(new List<double>() { 1, 1 }, 1, bipolar));
            }

            return dataset;
        }

        public static List<Entry> getTestingDataset(string name, bool bipolar = false)
        {
            List<Entry> dataset = new List<Entry>();
            if (name.Equals("AND"))
            {
                dataset.Add(new Entry(new List<double>() { 0, 0 }, 0, bipolar));
                dataset.Add(new Entry(new List<double>() { 0, 1 }, 0, bipolar));
                dataset.Add(new Entry(new List<double>() { 1, 0 }, 0, bipolar));
                dataset.Add(new Entry(new List<double>() { 1, 1 }, 1, bipolar));
                dataset.Add(new Entry(new List<double>() { 0.99, 0.99 }, 1, bipolar));
                dataset.Add(new Entry(new List<double>() { 0.11, 0.91 }, 0, bipolar));
                dataset.Add(new Entry(new List<double>() { 0.91, 0.07 }, 0, bipolar));
            }
            else if (name.Equals("OR"))
            {
                dataset.Add(new Entry(new List<double>() { 0, 0 }, 0, bipolar));
                dataset.Add(new Entry(new List<double>() { 0, 1 }, 1, bipolar));
                dataset.Add(new Entry(new List<double>() { 1, 0 }, 1, bipolar));
                dataset.Add(new Entry(new List<double>() { 1, 1 }, 1, bipolar));
            }

            return dataset;
        }
    }
}
