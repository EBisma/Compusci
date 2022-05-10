using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DecisionTree
{
    /// <summary>
    /// A leaf or branch of a tree
    /// Does most of the work of decision trees
    /// </summary>
    internal class Leaf
    {
        private bool isHidden = false;
        /// <summary>
        /// A pointer to the next leaves, if this is a branch
        /// </summary>
        private Leaf output1 = null;
        /// <summary>
        /// A pointer to the next leaves, if this is a branch
        /// </summary>
        private Leaf output2 = null;
        /// <summary>
        /// The value of the cut that is applied at this branch (unneeded if it is a leaf)
        /// </summary>
        private double split;
        /// <summary>
        /// The index of the variable which is used to make the cut (unneeded if this is a leaf)
        /// </summary>
        private int variable;

        /// <summary>
        /// The number of background training events in this leaf
        /// </summary>
        private int nBackground = 0;
        /// <summary>
        /// The number of signal training events in this leaf
        /// </summary>
        private int nSignal = 0;

        /// <summary>
        /// A default constructor is needed for some applications, but it is generally not sensible
        /// </summary>
        internal Leaf() :
            this(-1, 0) // Default values will generate an error if used
        { }

        /// <param name="variable">The index of the variable used to make the cut</param>
        /// <param name="split">The value of the cut used for the branch</param>
        public Leaf(int variable, double split)
        {
            this.variable = variable;
            this.split = split;
        }

        public void resetHidden()
        {
            this.isHidden = false;
            if(IsFinal)
            {
                return;
            }
            output1.resetHidden();
            output2.resetHidden();
        }

        public int Prune(DataSet signal, DataSet background, double alpha)
        {
            double minimizeThis;
            double error = findError(signal, background);
            int leftTerminalNodes = 0;
            int rightTerminalNodes = 0;
            double testPurity;
            if(IsFinal)
            {
                return 1;
            }
            else
            {
                var signalLeft = new DataSet(signal.Names);
                var signalRight = new DataSet(signal.Names);
                var backgroundLeft = new DataSet(background.Names);
                var backgroundRight = new DataSet(background.Names);
                foreach (var dataPoint in signal.Points)
                {
                    if (DoSplit(dataPoint))
                    {
                        signalLeft.AddDataPoint(dataPoint);
                    }
                    else
                    {
                        signalRight.AddDataPoint(dataPoint);
                    }
                }

                foreach (var dataPoint in background.Points)
                {
                    if (DoSplit(dataPoint))
                    {
                        backgroundLeft.AddDataPoint(dataPoint);
                    }
                    else
                    {
                        backgroundRight.AddDataPoint(dataPoint);
                    }
                }
                leftTerminalNodes = output1.Prune(signalLeft, backgroundLeft, alpha);
                rightTerminalNodes = output2.Prune(signalRight, backgroundRight, alpha);
                minimizeThis = error + alpha * (leftTerminalNodes + rightTerminalNodes);
                testPurity = Purity + alpha;
                if(minimizeThis < testPurity)
                {
                    return leftTerminalNodes + rightTerminalNodes;
                }
                isHidden = true;
                return 1;
            }
        }

       
        public double findError(DataSet signal, DataSet background)
        {
            double correctEvents = 0;
            foreach (var dp in signal.Points)
            {
                if (Math.Round(RunDataPoint(dp)) == 1)
                {
                    correctEvents++;
                }
            }
            foreach (var dp in background.Points)
            {
                if (Math.Round(RunDataPoint(dp)) == 0)
                {
                    correctEvents++;
                }
            }
            return 1-(correctEvents / (signal.Points.Count() + background.Points.Count()));
        }
        /// <summary>
        /// Write the leaf to a binary file
        /// </summary>
        internal void Write(BinaryWriter bw)
        {
            bw.Write(variable);
            bw.Write(split);
            bw.Write(nSignal);
            bw.Write(nBackground);

            bw.Write(IsFinal);
            if (!IsFinal)
            {
                output1.Write(bw);
                output2.Write(bw);
            }
        }

        /// <summary>
        /// Construct a leaf from a binary file
        /// </summary>
        internal Leaf(BinaryReader br)
        {
            variable = br.ReadInt32();
            split = br.ReadDouble();
            nSignal = br.ReadInt32();
            nBackground = br.ReadInt32();

            bool fin = br.ReadBoolean();
            if (!fin)
            {
                output1 = new Leaf(br);
                output2 = new Leaf(br);
            }
        }

        /// <summary>
        /// Determines if it is a leaf or a branch (true for leaves)
        /// </summary>
        public bool IsFinal => output1 == null || output2 == null;

        /// <summary>
        /// The purity of the leaf
        /// </summary>
        public double Purity => (double)nSignal / (nSignal + nBackground);

        /// <summary>
        /// Calculates the return value for a single data point, forwarding it to other leaves as needed
        /// </summary>
        public double RunDataPoint(DataPoint dataPoint)
        {
            if (IsFinal || isHidden)
            {
                return Purity;
            }

            if (DoSplit(dataPoint))
            {
                return output1.RunDataPoint(dataPoint);
            }
            else
            {
                return output2.RunDataPoint(dataPoint);
            }
        }

        /// <summary>
        /// Checks to see whether the DataPoint fails or passes the cut
        /// </summary>
        private bool DoSplit(DataPoint dataPoint)
        {
            return dataPoint.Variables[variable] <= split;
        }

        /// <summary>
        /// Trains this leaf based on input DataSets for signal and background
        /// </summary>
        public void Train(DataSet signal, DataSet background)
        {
            nSignal = signal.Points.Count;
            nBackground = background.Points.Count;

            // Determines whether this is a final leaf or if it branches
            bool branch = ChooseVariable(signal, background);

            if (branch)
            {
                // Creates a branch
                output1 = new Leaf();
                output2 = new Leaf();

                DataSet signalLeft = new DataSet(signal.Names);
                DataSet signalRight = new DataSet(signal.Names);
                DataSet backgroundLeft = new DataSet(background.Names);
                DataSet backgroundRight = new DataSet(background.Names);

                foreach (var dataPoint in signal.Points)
                {
                    if (DoSplit(dataPoint))
                    {
                        signalLeft.AddDataPoint(dataPoint);
                    }
                    else
                    {
                        signalRight.AddDataPoint(dataPoint);
                    }
                }

                foreach (var dataPoint in background.Points)
                {
                    if (DoSplit(dataPoint))
                    {
                        backgroundLeft.AddDataPoint(dataPoint);
                    }
                    else
                    {
                        backgroundRight.AddDataPoint(dataPoint);
                    }
                }

                // Trains each of the resulting leaves
                output1.Train(signalLeft, backgroundLeft);
                output2.Train(signalRight, backgroundRight);
            }
            // Do nothing more if it is not a branch
        }

        /// <summary>
        /// Chooses which variable and cut value to use
        /// </summary>
        /// <returns>True if a branch was created, false if this is a final leaf</returns>
        private bool ChooseVariable(DataSet signal, DataSet background)
        {
            // TODO set the values of variable and split here		
            // Return true if you were able to find a useful variable, 
            // and false if you were not and want to make a final leaf here

                // If you are going to branch, you should end with, for example:

                // variable = 3; // The index number of the variable you want
                // split = 2.55; // The value of the cut
                // return true;

                // Or if you cannot split usefully, you should
                // return false;
                // Make sure to do this or your code will run forever!
            //return false early if there's no signals or backgrounds cause then there's no point in splitting
            if (signal.Points.Count() == 0 || background.Points.Count() == 0)
            {
                return false;
            }

            //combines signals and background into one list for ease of accessing the list
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

            int bestVariableIndex = 0;
            double bestSplitValue = 0;
            double bestPurityDiff = 0;
            // TODO: Insert code here that calculates the proper values of bestVariableIndex and bestSplitValue
            for (int k = 0; k < signal.Points[0].Variables.Count(); k++)
            {
                double min = 0;
                double max = 0;
                double purityDiff = 0;

                //finds the min and max values of the variable it is checking so that it can test for split values starting and ending there
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

                double chunks = 100;
                //This goes through and tests different split values based on even sized chunks from the min to the max
                //Tests 100 chunks first and then checks half as many after zooming in on the lowest split value and spreading out a length of one chunk in both directions
                for (int n = 0; n < 2; n++)
                {
                    double chunkSize = (max - min) / chunks;

                    //Tests the chunks here
                    for (int i = 0; i < chunks; i++)
                    {
                        double numLowSignals = 0;
                        double numHighSignals = 0;
                        double numLowBackground = 0;
                        double numHighBackground = 0;
                        var splitValue = min + i * chunkSize;

                        //Checks the number of signals and backgrounds above and below the split value so i can calculate the purity of the leaves
                        for (int j = 0; j < combinedData.Count; j++)
                        {
                            if (combinedData[j].Item2 == true)
                            {
                                if (combinedData[j].Item1.Variables[k] >= splitValue)
                                {
                                    numHighSignals+=combinedData[j].Item1.Weight;
                                }
                                else if (combinedData[j].Item1.Variables[k] < splitValue)
                                {
                                    numLowSignals += combinedData[j].Item1.Weight;
                                }
                            }
                            if (combinedData[j].Item2 == false)
                            {
                                if (combinedData[j].Item1.Variables[k] >= splitValue)
                                {
                                    numHighBackground += combinedData[j].Item1.Weight;
                                }
                                else if (combinedData[j].Item1.Variables[k] < splitValue)
                                {
                                    numLowBackground += combinedData[j].Item1.Weight;
                                }
                            }
                        }

                        //this accounts for the fact that the purity of the split doesn't matter when the total number of data points on one side is 0, and also because if either equals 0 then the purity calcultion would break
                        if (numHighSignals + numHighBackground > 0 && numLowSignals + numLowBackground > 0)
                        {
                            //calculates the difference in purity between the leaves and if it is larger than the current difference, it's considered better since we want to reach as close to 0 and 1 respectively
                            if (Math.Abs(numHighSignals / (numHighBackground + numHighSignals) - numLowSignals / (numLowBackground + numLowSignals)) > bestPurityDiff && Math.Abs(numHighSignals / (numHighBackground + numHighSignals) - numLowSignals / (numLowBackground + numLowSignals)) < 1)
                            {
                                purityDiff = Math.Abs(numHighSignals / (numHighBackground + numHighSignals) - numLowSignals / (numLowBackground + numLowSignals));
                                bestSplitValue = splitValue;
                                bestVariableIndex = k;
                                bestPurityDiff = purityDiff;
                                variable = bestVariableIndex;
                                split = bestSplitValue;
                            }
                        }
                    }
                    min = bestSplitValue - chunkSize;
                    max = bestSplitValue + chunkSize;
                    chunks /= 2;
                }
                //throw new NotImplementedException();
            }
            //stops if the number of signals and backgrounds combined is ever less than 50 because at the point the number of points is small enough to probably not need to parse through and is a weak attempt at preventing overtraining; 50 is kind of an arbitrary number
            if (signal.Points.Count() + background.Points.Count() < 55)
            {
                return false;
            }
            return true;
        }
    }
}
