using System;
using System.Collections.Generic;
using System.IO;
using System.Text;

namespace DecisionTree
{
    static public class BinaryIOExtensions
    {
        static public void Write(this BinaryWriter bw, DataPoint dp)
        {
            foreach (var variable in dp.Variables)
            {
                bw.Write(variable);
            }
        }

        /// <summary>
        /// Reads a datapoint from a binary file
        /// Number of variables must be given externally to save on space
        /// </summary>
        /// <param name="br"></param>
        /// <param name="nvar"></param>
        /// <returns></returns>
        static public DataPoint ReadDataPoint(this BinaryReader br, int nvar)
        {
            var answer = new DataPoint(nvar);
            for (int i = 0; i < nvar; ++i)
            {
                answer.Variables[i] = br.ReadDouble();
            }

            return answer;
        }

        static public void Write(this BinaryWriter bw, DataSet ds)
        {
            if (ds.Points.Count == 0)
                return;

            // Write the number of variables
            bw.Write(ds.Points[0].Variables.Length);

            foreach (string name in ds.Names)
            {
                bw.Write(name);
            }

            bw.Write(ds.Points.Count);
            foreach (var point in ds.Points)
            {
                bw.Write(point);
            }
        }

        static public DataSet ReadDataSet(this BinaryReader br)
        {
            int nvar = br.ReadInt32();

            var names = new string[nvar];
            for (int i = 0; i < nvar; ++i)
            {
                names[i] = br.ReadString();
            }
            var dataset = new DataSet(names);

            int size = br.ReadInt32();

            for (int i = 0; i < size; ++i)
            {
                var dp = br.ReadDataPoint(nvar);
                dataset.AddDataPoint(dp);
            }

            return dataset;
        }

        internal static void Write(this BinaryWriter bw, Leaf leaf)
        {
            leaf.Write(bw);
        }

        internal static Leaf ReadLeaf(this BinaryReader br)
        {
            return new Leaf(br);
        }
    }
}
