using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SieciNeuronoweZad1
{
    class Entry
    {
        public List<double> inputs;
        public int output;
        public int vector_length;

        public Entry(List<double> inputs, int output, bool bipolar = false)
        {
            vector_length = inputs.Count;

            if (bipolar)
            {
                output = output == 0 ? -1 : output;
                for (int i=0; i<inputs.Count; i++)
                {
                    inputs[i] = inputs[i] == 0 ? -1 : inputs[i];
                }
            }

            this.inputs = inputs;
            this.output = output;
        }

        public string dumpEntryString()
        {
            string entry_string = "Input: [";

            foreach (double input in inputs)
                entry_string += input + ", ";

            entry_string = entry_string.Substring(0, entry_string.Length - 2) + "]; ";
            entry_string += string.Format("Output: [{0}]; ", output);

            return entry_string;
        }
    }
}
