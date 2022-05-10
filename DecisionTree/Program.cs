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
            //LevelI();
            //LevelII();
            LevelIII();
        }

        static void LevelI()
        {
            // Load training samples
            var signal = DataSet.ReadDataSet(path + "signal.dat");
            var background = DataSet.ReadDataSet(path + "background.dat");
            List<(DataPoint, bool)> combinedData = new();
            for (int i = 0; i < signal.Points.Count(); i++)
            {
                combinedData.Add((signal.Points[i], true));
            }
            for (int i = 0; i < background.Points.Count(); i++)
            {
                combinedData.Add((background.Points[i], false));
            }

            // Load data sample
            var data = DataSet.ReadDataSet(path + "decisionTreeData.dat");

            int bestVariableIndex = 0;
            double bestSplitValue = 0;
            double bestPurityDiff = 0;
            double purityIndex = 0;
            // TODO: Insert code here that calculates the proper values of bestVariableIndex and bestSplitValue
            for (int k = 0; k < signal.Points[0].Variables.Count(); k++)
            {
                double min = 0;
                double max = 0;
                double purityDiff = 0;
                for (int i = 0; i < combinedData.Count; i++)
                {
                    if (i == 0)
                    {
                        min = combinedData[i].Item1.Variables[k];
                        max = combinedData[i].Item1.Variables[k];
                    }
                    else
                    {
                        if (combinedData[i].Item1.Variables[k] < min)
                        {
                            min = combinedData[i].Item1.Variables[k];
                        }
                        else if (combinedData[i].Item1.Variables[k] > max)
                        {
                            max = combinedData[i].Item1.Variables[k];
                        }
                    }
                }

                double chunks = 10000;
                double chunkSize = (max - min) / chunks;
                for (int i = 0; i < chunks; i++)
                {
                    double numLowSignals = 0;
                    double numHighSignals = 0;
                    double numLowBackground = 0;
                    double numHighBackground = 0;
                    var splitValue = min + i * chunkSize;
                    for (int j = 0; j < combinedData.Count; j++)
                    {
                        if (combinedData[j].Item2 == true)
                        {
                            if (combinedData[j].Item1.Variables[k] >= splitValue)
                            {
                                numHighSignals++;
                            }
                            else if (combinedData[j].Item1.Variables[k] < splitValue)
                            {
                                numLowSignals++;
                            }
                        }
                        if (combinedData[j].Item2 == false)
                        {
                            if (combinedData[j].Item1.Variables[k] >= splitValue)
                            {
                                numHighBackground++;
                            }
                            else if (combinedData[j].Item1.Variables[k] < splitValue)
                            {
                                numLowBackground++;
                            }
                        }
                    }
                    if (numHighSignals + numHighBackground > 0 && numLowSignals + numLowBackground > 0)
                    {
                        if (Math.Abs(numHighSignals / (numHighBackground + numHighSignals) - numLowSignals / (numLowBackground + numLowSignals)) > bestPurityDiff && Math.Abs(numHighSignals / (numHighBackground + numHighSignals) - numLowSignals / (numLowBackground + numLowSignals)) < 1)
                        {
                            purityDiff = Math.Abs(numHighSignals / (numHighBackground + numHighSignals) - numLowSignals / (numLowBackground + numLowSignals));
                            bestSplitValue = splitValue;
                            bestVariableIndex = k;
                            bestPurityDiff = purityDiff;
                        }
                    }
                }
            }


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

        static void LevelII()
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
            tree.MakeTextFile(path + "decisionTreeResultsLevelII.txt", data);

            //This prints effectiveness; the only thing I'm unsure if this is the result of overtraining, but my efficacy is pretty high
            double correctEvents = 0;
            foreach (var dp in signal.Points)
            {
                if (Math.Round(tree.RunDataPoint(dp)) == 1)
                {
                    correctEvents++;
                }
            }
            foreach (var dp in background.Points)
            {
                if (Math.Round(tree.RunDataPoint(dp)) == 0)
                {
                    correctEvents++;
                }
            }
            Console.WriteLine(correctEvents / (signal.Points.Count() + background.Points.Count()));
        }

        static void LevelIII()
        {
            //loads training samples
            //var signal = DataSet.ReadDataSet(path + "signal.dat");
            //var background = DataSet.ReadDataSet(path + "background.dat");
            var signal = DataSet.ReadDataSet(path + "signalOverallTrainingSample.dat");
            var background = DataSet.ReadDataSet(path + "backgroundOverallTrainingSample.dat");

            //load data sample
            //var data = DataSet.ReadDataSet(path + "decisionTreeData.dat");
            var data = DataSet.ReadDataSet(path + "project3Data.dat");

            var forest = new Forest(10);

            //train a forest
            forest.Train(signal, background);

            //forest.MakeTextFile(path + "decisionTreeResultsLevelIII.txt", data);
            //forest.MakeTextFile(path + "project3Results.txt", data);
            forest.PrintSignalIndexes(path + "Project3ResultsEthanBrazeltonAndJamesTam.txt", data);


            //this prints effectiveness, and it's in fact higher than the last, but idk if this is the result of overtraining
            double correctEvents = 0;
            foreach(var dp in signal.Points)
            {
                if(Math.Round(forest.RunDataPoint(dp))==1)
                {
                    correctEvents++;
                }
            }
            foreach(var dp in background.Points)
            {
                if(Math.Round(forest.RunDataPoint(dp))==0)
                {
                    correctEvents++;
                }
            }
            Console.WriteLine(correctEvents/(signal.Points.Count()+background.Points.Count()));
        }
    }
}
