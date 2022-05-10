using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DecisionTree
{
    internal class Forest
    {
        //James helped me with the logic of figuring out the proper algorithm and how to implement it here
        //Takes a list of trees in the forest
        private List<(Tree tree, double weight)> trees;

        //The number of trees made; can be changed as desired in Program.cs
        private int forestSize;

        public Forest(int forestSize)
        {
            this.forestSize = forestSize;
            trees = new();
        }

        public void Train(DataSet signal, DataSet background)
        {
            //combines signal and background into a single list
            List<(DataPoint, bool)> combinedData = new();
            for (int i = 0; i < signal.Points.Count(); i++)
            {
                combinedData.Add((signal.Points[i], true));
            }
            for (int i = 0; i < background.Points.Count(); i++)
            {
                combinedData.Add((background.Points[i], false));
            }
            //Repeats for each tree
            for (int i = 0; i < forestSize; i++)
            {
                double totalWeights = 0;
                //
                for(int j = 0; j < combinedData.Count; j++)
                {
                    totalWeights += combinedData[j].Item1.Weight;
                }
                for(int j = 0; j < combinedData.Count; j++)
                {
                    combinedData[j].Item1.Weight /= totalWeights;
                }
                //Trains a tree
                var tree = new Tree();
                tree.Train(signal, background);
                tree.Prune(signal, background);
                double totalErrorWeight = 0;
                //Finds the total of the weights of incorrectly predicted datapoints
                for(int j = 0; j < combinedData.Count; j++)
                {
                    var dataPointPurity = tree.RunDataPoint(combinedData[j].Item1);
                    dataPointPurity = Math.Round(dataPointPurity);
                    if((dataPointPurity == 1)!=combinedData[j].Item2)
                    {
                        totalErrorWeight += combinedData[j].Item1.Weight;
                    }
                }
                //creates a weight constant based on the information from the article, as well as the tree weight
                double weightConstant = Math.Sqrt((1 - totalErrorWeight) / totalErrorWeight);
                double treeWeight = Math.Log(weightConstant);
                //Alters the weight of each data point in the combined list
                for(int j = 0; j < combinedData.Count; j++)
                {
                    var dataPointPurity = tree.RunDataPoint(combinedData[j].Item1);
                    dataPointPurity = Math.Round(dataPointPurity);
                    if ((dataPointPurity == 1) != combinedData[j].Item2)
                    {
                        combinedData[j].Item1.Weight *= weightConstant;
                    }
                    else
                    {
                        combinedData[j].Item1.Weight /= weightConstant;
                    }
                }
                trees.Add((tree, treeWeight));
            }
        }

        //Calculates the purity of dp based on the weights of the trees in the forest
        public double RunDataPoint(DataPoint dp)
        {
            double totalTreePurity = 0;
            double totalTreeWeight = 0;
            for(int i = 0; i < trees.Count; i++)
            {
                var treeWeight = trees[i].Item2;
                totalTreePurity += trees[i].Item1.RunDataPoint(dp) * treeWeight;
                totalTreeWeight += treeWeight;
            }
            return totalTreePurity/ totalTreeWeight;
        }

        //Makes a text file outputting events and purity
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

        static bool isSignal(
                (double x, double y, double z) pos,
                (double x, double y, double z) vel,
                double mass,
                (int r, int g, int b) color)
        {
            // We just placed pre-conditions that we thought would be complex and difficult to guess for our opponents
            return 20 * vel.x * vel.y * vel.z > (pos.x + pos.y + pos.z) * mass
                || color.r * color.g * color.b < 50 * (pos.x + pos.y + pos.z) * mass
                || color.r + color.g + color.b > 45 * (vel.x + vel.y + vel.z)
                || (Math.Abs(pos.x - mass) > 15 * vel.x
                    && Math.Abs(pos.y - mass) > 15 * vel.y
                    && Math.Abs(pos.z - mass) > 15 * vel.z);
        }

        public void PrintSignalIndexes(string filename, DataSet data)
        {
            using var file = File.CreateText(filename);
            file.WriteLine("Habitable Planet Indexes");
            for (int i = 0; i< data.Points.Count; ++i)
            {
                double output = RunDataPoint(data.Points[i]);
                if (Math.Round(output) == 1 || isSignal((data.Points[i].Variables[0], data.Points[i].Variables[1], data.Points[i].Variables[2]), (data.Points[i].Variables[3], data.Points[i].Variables[4], data.Points[i].Variables[5]), data.Points[i].Variables[6], ((int) data.Points[i].Variables[7],(int) data.Points[i].Variables[8], (int) data.Points[i].Variables[9])));
                {
                    file.WriteLine(i);
                }
            }
        }
    }
}
