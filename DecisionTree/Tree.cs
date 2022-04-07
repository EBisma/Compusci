using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DecisionTree
{
    /// <summary>
    /// A class to represent a decision tree
    /// The Leaf class does all the work
    /// </summary>
    public class Tree
    {
        /// <summary>
        /// The top of the tree, which is all that is needed - all other leaves are accessed from here
        /// </summary>
        private Leaf headnode;

        public Tree()
        {
            headnode = new Leaf();
        }

        /// <summary>
        /// Writes the tree to a binary file
        /// </summary>
        public void WriteToFile(string filename)
        {
            using (var bw = new BinaryWriter(File.Create(filename)))
            {
                headnode.Write(bw);
            }
        }

        /// <summary>
        /// Constructs the tree from a binary file
        /// </summary>
        public Tree(string filename)
        {
            using (var br = new BinaryReader(File.OpenRead(filename)))
            {
                headnode = new Leaf(br);
            }
        }

        /// <summary>
        /// Trains the tree on signal and background samples
        /// </summary>
        public void Train(DataSet signal, DataSet background)
        {
            headnode.Train(signal, background);
        }

        /// <summary>
        /// Calculates the value for a single DataPoint
        /// Assumes the tree is trained already
        /// </summary>
        public double RunDataPoint(DataPoint dp)
        {
            return headnode.RunDataPoint(dp);
        }

        public void MakeTextFile(string filename, DataSet data)
        {
            using var file = File.CreateText(filename);
            file.WriteLine("Event\tPurity");

            for (int i = 0; i < data.Points.Count; ++i)
            {
                double output = RunDataPoint(data.Points[i]);
                file.WriteLine(i + "\t" + output);
            }
        }
    }
}
