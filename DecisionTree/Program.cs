using System;
using System.Collections.Generic;
using System.IO;
using System.Reflection;
using System.Text;

namespace DecisionTree
{
    internal static class Program
    {
        private static readonly string path = Directory.GetParent(Directory.GetCurrentDirectory()).Parent.Parent.Parent.FullName + '\\';
        static void Main()
        {
            LevelI();
            LevelIIAndBeyond();
        }

        static void LevelI()
        {
            // Load training samples
            var signal = DataSet.ReadDataSet(path + "signal.dat");
            var background = DataSet.ReadDataSet(path + "background.dat");

            // Load data sample
            var data = DataSet.ReadDataSet(path + "decisionTreeData.dat");

            int bestVariableIndex = -1;
            double bestSplitValue = 0;

            // TODO: Insert code here that calculates the proper values of bestVariableIndex and bestSplitValue





            using var file = File.CreateText(path + "decisionTreeResultsLevelI.txt");
            file.WriteLine("Event\tPurity");

            for (int i = 0; i < data.Points.Count; ++i)
            {
                // Note that you may have to change the order of the 1 and 0 here, depending on which one matches signal.
                // 1 means signal and 0 means background
                double output = data.Points[i].Variables[bestVariableIndex] > bestSplitValue ? 1 : 0;
                file.WriteLine(i + "\t" + output);
            }
        }

        static void LevelIIAndBeyond()
        {
            // Load training samples
            var signal = DataSet.ReadDataSet(path + "signal.dat");
            var background = DataSet.ReadDataSet(path + "background.dat");

            // Load data sample
            var data = DataSet.ReadDataSet(path + "decisionTreeData.dat");

            var tree = new Tree();

            // Train the tree
            tree.Train(signal, background);

            // Calculate output value for each event and write to file
            tree.MakeTextFile(path + "decisionTreeResults.txt", data);
        }
    }
}
