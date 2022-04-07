using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Text;
using System.Threading.Tasks;

namespace DecisionTree
{
    /// <summary>
    /// Class holding information for a single event
    /// </summary>
    public class DataPoint
    {
        // An array of doubles, representing the relevant variables for the event
        public double[] Variables { get; set; }

        /// <param name="nvar">The number of different variables in the point</param>
        public DataPoint(int nvar)
        {
            Variables = new double[nvar];
        }
    }
}
