using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SieciNeuronoweZad1
{
    class PerceptronSettings
    {
        public bool is_bipolar;
        public bool is_biased;

        public int vector_len;

        public double starting_weight_range;
        public double threshold;
        public double modifier;

        public int active_value, inactive_value;

        public PerceptronSettings(double modifier, double starting_weight_range = 1.0, bool is_bipolar = false, bool is_biased = false, double custom_threshold = 0.5)
        {
            this.starting_weight_range = starting_weight_range;
            this.modifier = modifier;

            this.is_biased = is_biased;
            this.is_bipolar = is_bipolar;
            threshold = custom_threshold;

            active_value = 1;
            inactive_value = 0;

            if (is_bipolar)
                inactive_value = -1;

            if (is_biased)
                threshold = 0.0;
        }

        public void setVectorLen(int vector_len)
        {
            if (is_biased)
                vector_len++;

            this.vector_len = vector_len;
        }

        public string dumpSettingsString()
        {
            string settings_string = "";

            settings_string += string.Format("Alpha: [{0}]\n", modifier);
            settings_string += string.Format("Zakres wag początkowych: [-{0}, {0}]\n", starting_weight_range);
            settings_string += string.Format("Bipolarny: [{0}]\n", is_bipolar ? "Tak" : "Nie");
            settings_string += string.Format("Bias: [{0}]\n", is_biased ? "Tak" : "Nie");
                      
            settings_string += is_biased ? "" : string.Format("Threshold: [{0}]\n", threshold);

            settings_string += "\n";

            return settings_string;
        }
    }
}
