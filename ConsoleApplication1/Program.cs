using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.IO;
using System.Drawing;





namespace NeuralNetwork
{
    class Program
    {
        static void Main(string[] args)
        {
            /*
            MNIST DB = new MNIST();
            DB.LoadDB("", 30, 10000);
            NeurolAB<byte> A = new NeurolAB<byte>(new int[] { 784, 392, 10 });
            foreach (DigitImage i in DB.TrainingImages)
            {
                short[] output = new short[10] {-1,-1,-1,-1,-1,-1,-1,-1,-1,-1};
                output[i.Label] = 1;
                A.Add(i.RawImage, output);
            }
            A.Learn(1E-20d);
            Console.WriteLine(DB.TrainingImages[0].ToString());
            */

            double[][] XOR_I = new double[4][] {
                new double[2] { 0, 0 },
                new double[2] { 0, 1 },
                new double[2] { 1, 0 },
                new double[2] { 1, 1 }
            };
            double[][] XOR_O = new double[4][] {
                new double[1] { 0 },
                new double[1] { 1 },
                new double[1] { 1 },
                new double[1] { 0 }
            };
            NeuralNetworkAB<double> A = new NeuralNetworkAB<double>(new int[] { 2, 3, 1 });
            A.Input = XOR_I;
            A.Output = XOR_O;
            A.Learn(1E-10);
            
            double[] result = A.Run(XOR_I[1]);
            Console.WriteLine(string.Join(" , ", result));
            Console.ReadKey();
        }
    }
}
